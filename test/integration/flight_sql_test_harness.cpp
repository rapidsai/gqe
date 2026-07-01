/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "flight_sql_test_harness.hpp"

#include "flight_sql_test_utils.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace gqe_test {

TestConfig g_config;
ServerInstance g_server;

bool ServerInstance::start(int num_gpus,
                           const char* node_manager_bin,
                           const char* task_manager_bin,
                           const char* load_script,
                           const char* data_path)
{
  num_gpus_ = num_gpus;
  port_     = find_free_port();

  // Remove any stale shared memory segment from a previous test run.
  std::remove("/dev/shm/gqe_shared_memory");

  server_pid_ = fork();
  if (server_pid_ == -1) { return false; }

  if (server_pid_ == 0) {
    setenv("GQE_MAX_QUERY_MEMORY", "1073741824", /*overwrite=*/0);
    setenv("GQE_MAX_TASK_MANAGER_MEMORY", "1073741824", /*overwrite=*/0);
    auto port_str     = std::to_string(port_);
    auto num_gpus_str = std::to_string(num_gpus);
    execl(node_manager_bin,
          node_manager_bin,
          "--address",
          "127.0.0.1",
          "--port",
          port_str.c_str(),
          "--num-gpus",
          num_gpus_str.c_str(),
          "--task-manager-binary",
          task_manager_bin,
          nullptr);
    _exit(127);
  }

  if (!wait_for_port(port_)) { return false; }

  // Forward TPCH_SCHEMA to load_tpch.py when set so CI can override the script's
  // default `schema.sql` with the handcoded `ci_schema.sql`.
  std::string schema_arg;
  if (auto const* schema = std::getenv("TPCH_SCHEMA"); schema && schema[0] != '\0') {
    schema_arg = std::format(" --schema {}", schema);
  }
  auto cmd = std::format(
    "{} --server-url http://127.0.0.1:{}{} {}", load_script, port_, schema_arg, data_path);
  auto [exit_code, output] = run_command_with_output(cmd);
  if (exit_code != 0) {
    load_error_ = output;
    return false;
  }

  ready_ = true;
  return true;
}

void ServerInstance::stop()
{
  if (server_pid_ > 0) {
    kill(server_pid_, SIGTERM);
    int status;
    waitpid(server_pid_, &status, 0);
    server_pid_ = -1;
  }
  // Remove any shared memory segment left by the node manager.
  if (std::remove("/dev/shm/gqe_shared_memory") == 0) {
    std::fprintf(stderr,
                 "Warning: removed stale shared memory segment /dev/shm/gqe_shared_memory\n");
  }
  ready_ = false;
}

}  // namespace gqe_test

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  using gqe_test::env_or;
  auto& cfg = gqe_test::g_config;
  cfg.node_manager_bin =
    env_or("GQE_NODE_MANAGER", GQE_PROJECT_BINARY_DIR "/src/node_manager/gqe_node_manager");
  cfg.task_manager_bin =
    env_or("GQE_TASK_MANAGER", GQE_PROJECT_BINARY_DIR "/src/task_manager/gqe_task_manager");
  cfg.client_bin  = env_or("GQE_CLI", GQE_PROJECT_BINARY_DIR "/cargo-target/release/gqe-cli");
  cfg.load_script = env_or("GQE_LOAD_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/load_tpch.py");
  cfg.run_script  = env_or("GQE_RUN_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/run_tpch.py");
  // Export the resolved cli path so child Python processes inherit it via
  // os.environ["GQE_CLI"].
  setenv("GQE_CLI", cfg.client_bin, /*overwrite=*/0);
  cfg.data_path          = std::getenv("TPCH_DATA_PATH");
  cfg.queries_dir        = std::getenv("TPCH_QUERIES");
  cfg.physical_plans_dir = std::getenv("TPCH_PHYSICAL_PLANS");
  cfg.ref_results        = std::getenv("TPCH_REF_RESULTS");
  cfg.host_gpu_count     = gqe_test::detect_gpu_count();

  if (!cfg.data_path) { return RUN_ALL_TESTS(); }  // server cannot start; fixtures self-skip

  // Run each GPU configuration sequentially (shared memory segment is global).
  int result = 0;
  for (auto [num_gpus, suite_prefix] : {std::pair{1, std::string_view{"SingleGpu/"}},
                                        std::pair{2, std::string_view{"MultiGpu/"}}}) {
    if (num_gpus > cfg.host_gpu_count) { continue; }

    std::printf("=== Starting server with %d GPU(s) ===\n", num_gpus);
    if (!gqe_test::g_server.start(
          num_gpus, cfg.node_manager_bin, cfg.task_manager_bin, cfg.load_script, cfg.data_path)) {
      std::fprintf(stderr,
                   "Failed to start %d-GPU server: %s\n",
                   num_gpus,
                   gqe_test::g_server.load_error().c_str());
      result = 1;
      continue;
    }

    ::testing::GTEST_FLAG(filter) = std::format("{}*", suite_prefix);
    if (RUN_ALL_TESTS() != 0) { result = 1; }

    gqe_test::g_server.stop();
    std::printf("=== Stopped %d-GPU server ===\n", num_gpus);
  }
  return result;
}
