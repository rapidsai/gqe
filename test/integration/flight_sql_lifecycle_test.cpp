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

/**
 * Tests that launch their own node manager instance. These cannot share a
 * server with the main integration tests because they need exclusive access
 * to the shared memory segment (or test shutdown/crash behaviour).
 *
 * Parameterized by GPU count (1 and 2). 2-GPU tests are automatically
 * skipped on hosts with fewer than 2 GPUs.
 */

#include "flight_sql_test_utils.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>

namespace gqe_test {

// ---------------------------------------------------------------------------
// Test fixture — each test launches its own node manager.
// Parameterized by GPU count.
// ---------------------------------------------------------------------------

class FlightSqlLifecycleTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override
  {
    if (!std::getenv("TPCH_DATA_PATH") || !std::getenv("TPCH_QUERIES")) {
      GTEST_SKIP() << "TPCH_DATA_PATH and TPCH_QUERIES must be set";
    }
    // Remove any stale shared memory segment left by a previous test binary.
    if (std::remove("/dev/shm/gqe_shared_memory") == 0) {
      std::fprintf(stderr,
                   "Warning: removed stale shared memory segment /dev/shm/gqe_shared_memory\n");
    }

    nm_bin_ =
      env_or("GQE_NODE_MANAGER", GQE_PROJECT_BINARY_DIR "/src/node_manager/gqe_node_manager");
    tm_bin_ =
      env_or("GQE_TASK_MANAGER", GQE_PROJECT_BINARY_DIR "/src/task_manager/gqe_task_manager");
    client_bin_  = env_or("GQE_CLI", GQE_PROJECT_BINARY_DIR "/cargo-target/release/gqe-cli");
    load_script_ = env_or("GQE_LOAD_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/load_tpch.py");
    run_script_  = env_or("GQE_RUN_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/run_tpch.py");
    // Export the resolved cli path so child Python processes inherit it via
    // os.environ["GQE_CLI"].
    setenv("GQE_CLI", client_bin_, /*overwrite=*/0);
    data_path_   = std::getenv("TPCH_DATA_PATH");
    queries_dir_ = std::getenv("TPCH_QUERIES");

    num_gpus_ = GetParam();
    if (num_gpus_ > detect_gpu_count()) {
      GTEST_SKIP() << "Test requires " << num_gpus_ << " GPUs but host has " << detect_gpu_count();
    }
  }

  void TearDown() override
  {
    if (nm_pid_ > 0) {
      // Send SIGTERM first to allow graceful shutdown (shared memory cleanup).
      kill(nm_pid_, SIGTERM);
      int status;
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      bool exited   = false;
      while (std::chrono::steady_clock::now() < deadline) {
        if (waitpid(nm_pid_, &status, WNOHANG) > 0) {
          exited = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      if (!exited) {
        kill(nm_pid_, SIGKILL);
        waitpid(nm_pid_, &status, 0);
      }
    }
    // Safety net: if the node manager didn't clean up (e.g. after SIGKILL),
    // remove the stale segment so the next test can start fresh.
    if (std::remove("/dev/shm/gqe_shared_memory") == 0) {
      std::fprintf(stderr,
                   "Warning: removed stale shared memory segment /dev/shm/gqe_shared_memory\n");
    }
  }

  /** Launch a node manager with num_gpus_ GPUs. Sets nm_pid_ and port_. */
  void launch_node_manager()
  {
    port_ = find_free_port();

    nm_pid_ = fork();
    ASSERT_NE(nm_pid_, -1);
    if (nm_pid_ == 0) {
      auto port_str     = std::to_string(port_);
      auto num_gpus_str = std::to_string(num_gpus_);
      setenv("GQE_MAX_QUERY_MEMORY", "1073741824", /*overwrite=*/0);
      setenv("GQE_MAX_TASK_MANAGER_MEMORY", "1073741824", /*overwrite=*/0);
      execl(nm_bin_,
            nm_bin_,
            "--address",
            "127.0.0.1",
            "--port",
            port_str.c_str(),
            "--num-gpus",
            num_gpus_str.c_str(),
            "--task-manager-binary",
            tm_bin_,
            nullptr);
      _exit(127);
    }

    ASSERT_TRUE(wait_for_port(port_, 60)) << "Node manager did not start";
  }

  /** Load TPC-H data into the running server. */
  void load_data()
  {
    ASSERT_NE(load_script_, nullptr) << "GQE_LOAD_SCRIPT must be set";
    ASSERT_NE(data_path_, nullptr) << "TPCH_DATA_PATH must be set";
    auto cmd =
      std::format("{} --server-url http://127.0.0.1:{} {}", load_script_, port_, data_path_);
    auto [exit_code, output] = run_command_with_output(cmd);
    ASSERT_EQ(exit_code, 0) << "load_tpch.py failed. Output:\n" << output;
  }

  const char* nm_bin_      = nullptr;
  const char* tm_bin_      = nullptr;
  const char* client_bin_  = nullptr;
  const char* load_script_ = nullptr;
  const char* run_script_  = nullptr;
  const char* data_path_   = nullptr;
  const char* queries_dir_ = nullptr;
  pid_t nm_pid_            = -1;
  int port_                = 0;
  int num_gpus_            = 0;
};

// ---------------------------------------------------------------------------
// Instantiations
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(SingleGpu,
                         FlightSqlLifecycleTest,
                         ::testing::Values(1),
                         [](::testing::TestParamInfo<int> const& info) {
                           return std::format("{}GPU", info.param);
                         });

INSTANTIATE_TEST_SUITE_P(MultiGpu,
                         FlightSqlLifecycleTest,
                         ::testing::Values(2),
                         [](::testing::TestParamInfo<int> const& info) {
                           return std::format("{}GPU", info.param);
                         });

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/** Verify that SIGINT triggers graceful shutdown with no orphan task managers. */
TEST_P(FlightSqlLifecycleTest, GracefulShutdown)
{
  launch_node_manager();

  // Find task manager children before shutdown.
  auto children = find_child_pids(nm_pid_);
  EXPECT_FALSE(children.empty()) << "Expected at least one task manager child process";

  // Send SIGINT for graceful shutdown.
  kill(nm_pid_, SIGINT);

  int status;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  pid_t result  = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    result = waitpid(nm_pid_, &status, WNOHANG);
    if (result > 0) break;
    usleep(100'000);
  }

  if (result > 0) {
    nm_pid_ = -1;  // Prevent TearDown from killing again.
  }
  ASSERT_GT(result, 0) << "Node manager did not exit within 15 seconds after SIGINT";

  // Verify no orphan task manager processes remain.
  usleep(500'000);
  for (auto child : children) {
    int child_status;
    pid_t r = waitpid(child, &child_status, WNOHANG);
    // If waitpid returns 0, the process is still running (orphan).
    // If it returns -1 with ECHILD, the process no longer exists (good).
    if (r == 0) {
      kill(child, SIGKILL);
      waitpid(child, &child_status, 0);
      ADD_FAILURE() << "Orphan task manager (pid " << child << ") found after shutdown";
    }
  }
}

/**
 * Kill a task manager during query execution. The in-flight query should fail
 * (not hang), and a subsequent query after recovery should succeed.
 * @note Currently disabled because restart_task_managers() does not reconnect
 * the gRPC channels to the new task manager ports. The first part of the test
 * (crash detection, prompt error) works; recovery does not. Re-enable once the
 * restart logic rebuilds gRPC channels.
 */
TEST_P(FlightSqlLifecycleTest, DISABLED_TaskManagerCrash)
{
  if (!data_path_ || !queries_dir_) { FAIL() << "TPCH_DATA_PATH and TPCH_QUERIES must be set"; }

  launch_node_manager();
  load_data();

  // Find a task manager child process.
  auto children = find_child_pids(nm_pid_);
  ASSERT_FALSE(children.empty()) << "No task manager children found";

  // Kill the task manager.
  kill(children[0], SIGKILL);

  // A query issued shortly after should fail (not hang).
  auto start        = std::chrono::steady_clock::now();
  auto [rc, output] = run_tpch_query(run_script_, InputFormat::Sql, port_, queries_dir_, 6);
  auto elapsed      = std::chrono::steady_clock::now() - start;

  EXPECT_LT(elapsed, std::chrono::seconds(120))
    << "Query should complete promptly after task manager crash, not hang";

  // Wait for the node manager to restart task managers and retry.
  // The restart may take several seconds; retry up to 30s.
  bool recovered = false;
  for (int attempt = 0; attempt < 6; ++attempt) {
    usleep(5'000'000);
    auto [rc2, output2] = run_tpch_query(run_script_, InputFormat::Sql, port_, queries_dir_, 6);
    if (rc2 == 0) {
      recovered = true;
      break;
    }
  }
  EXPECT_TRUE(recovered) << "Query should succeed after recovery (tried for 30s)";
}

}  // namespace gqe_test
