/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Tests that run the `gqe_task_manager` binary as a child process and drive it over gRPC, the way
 * the node manager does.
 */

#include "../integration/flight_sql_test_utils.hpp"

#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <format>
#include <optional>
#include <string>
#include <thread>

namespace {

/** @brief Kills and reaps the child process unless the test has already reaped it. */
struct child_process {
  pid_t pid = -1;

  child_process()                                = default;
  child_process(child_process const&)            = delete;
  child_process& operator=(child_process const&) = delete;
  child_process(child_process&&)                 = delete;
  child_process& operator=(child_process&&)      = delete;

  ~child_process()
  {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
  }

  /** @brief Wait up to `timeout` for the child to end, and return its wait status. */
  std::optional<int> wait_for_exit(std::chrono::seconds timeout)
  {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      int status = 0;
      if (waitpid(pid, &status, WNOHANG) == pid) {
        pid = -1;
        return status;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return std::nullopt;
  }
};

}  // namespace

/**
 * @brief Test that a task manager whose `InitNvshmem` fails replies with the error and then exits
 * with a non-zero status on its own.
 *
 * The UID has the wrong size, so initialization fails before NVSHMEM is touched.
 */
TEST(TaskManagerProcessTest, FailedInitNvshmemRepliesThenExits)
{
  auto const port     = gqe_test::find_free_port();
  auto const binary   = GQE_PROJECT_BINARY_DIR "/src/task_manager/gqe_task_manager";
  auto const port_arg = std::to_string(port);

  child_process child;
  child.pid = fork();
  ASSERT_GE(child.pid, 0);
  if (child.pid == 0) {
    execl(binary, binary, "--grpc-port", port_arg.c_str(), nullptr);
    _exit(127);
  }

  auto channel =
    grpc::CreateChannel(std::format("127.0.0.1:{}", port), grpc::InsecureChannelCredentials());
  ASSERT_TRUE(
    channel->WaitForConnected(std::chrono::system_clock::now() + std::chrono::seconds(60)));
  auto stub = gqe::proto::TaskManagerService::NewStub(channel);

  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
  gqe::proto::InitNvshmemRequest request;
  request.set_rank(0);
  request.set_num_ranks(1);
  request.set_unique_id("malformed");
  gqe::proto::InitNvshmemResponse response;
  auto const status = stub->InitNvshmem(&ctx, request, &response);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_FALSE(response.success());
  EXPECT_NE(response.error_message().find("Invalid NVSHMEM UID size"), std::string::npos)
    << response.error_message();

  auto const wait_status = child.wait_for_exit(std::chrono::seconds(60));
  ASSERT_TRUE(wait_status.has_value()) << "The task manager did not exit after a failed bootstrap";
  ASSERT_TRUE(WIFEXITED(*wait_status)) << "wait status " << *wait_status;
  EXPECT_EQ(WEXITSTATUS(*wait_status), 1);
}
