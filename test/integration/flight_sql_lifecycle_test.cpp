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

#include <gqe/node_manager/exit_code.hpp>
#include <gqe/storage/shared_segment.hpp>

#include <gtest/gtest.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <sys/wait.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <future>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace gqe_test {

namespace {

/**
 * Poison the allocator lock of the node manager's shared-memory segment.
 *
 * Forks a child that maps the segment, takes the allocator lock of its first arena, and dies with
 * SIGKILL while holding it. Every later allocation or deallocation in that arena then fails. The
 * node manager creates the segment with a single arena.
 */
void poison_segment_allocator_lock()
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  // The allocator lock is reachable only in a build with the CMake option
  // `GQE_ENABLE_FAULT_INJECTION`. Each test that calls this function skips itself first
  // (`GTEST_SKIP`), so this `FAIL()` is only a guard against a call from a test that does not.
  FAIL() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#else
  pid_t const pid = fork();
  ASSERT_NE(pid, -1) << "fork failed";
  if (pid == 0) {
    try {
      // This process only touches the segment's memory. Registering it with CUDA is not needed,
      // and fails in a forked child.
      gqe::shared_segment segment(
        gqe::open_only, "gqe_shared_memory", gqe::shared_segment::registration_policy::none);
      segment.get_segment_manager()->arena_lock_for_testing(0).lock();
      // Die while the segment is still mapped. The kernel marks a dead holder's robust mutexes
      // only if it can reach them, so unmapping first would leave the lock held forever.
      kill(getpid(), SIGKILL);
    } catch (...) {
      _exit(1);
    }
    _exit(2);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
    << "The child poisoning the segment did not die from SIGKILL, status " << status;
#endif
}

/**
 * Set the node manager's shared-memory poisoned flag, as a process does after it observes a failed
 * lock in the segment.
 *
 * Poisoning a lock alone is only noticed by the next operation that takes it. This is for tests
 * that need the node manager to begin teardown while it is busy and no such operation can run.
 */
void mark_segment_poisoned()
{
  // No CUDA registration: this process only writes the flag.
  gqe::shared_segment segment(
    gqe::open_only, "gqe_shared_memory", gqe::shared_segment::registration_policy::none);
  segment.get_segment_manager()->poisoned_flag().store(1, std::memory_order_release);
}

/** Whether @p pid is a live process, as opposed to absent or a zombie. */
bool is_process_running(pid_t pid)
{
  std::ifstream stat(std::format("/proc/{}/stat", pid));
  std::string line;
  if (!std::getline(stat, line)) { return false; }
  // The state follows the parenthesized command name, which may itself contain spaces.
  auto const state_pos = line.rfind(") ");
  return state_pos != std::string::npos && state_pos + 2 < line.size() &&
         line[state_pos + 2] != 'Z';
}

/**
 * Whether @p pid is a live process running @p binary.
 *
 * Checking the executable guards against PID reuse: once a process exits, an unrelated process may
 * get its PID.
 */
bool is_process_running_binary(pid_t pid, char const* binary)
{
  if (!is_process_running(pid)) { return false; }
  std::error_code error;
  auto const executable = std::filesystem::read_symlink(std::format("/proc/{}/exe", pid), error);
  return !error && std::filesystem::equivalent(executable, binary, error) && !error;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture — each test launches its own node manager.
// Parameterized by GPU count.
// ---------------------------------------------------------------------------

class FlightSqlLifecycleTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override
  {
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

  /**
   * Start a node manager with num_gpus_ GPUs, without waiting for it. Sets nm_pid_ and port_.
   *
   * @param extra_args Further command-line arguments, such as `--query-timeout`, appended to the
   * node manager's.
   */
  void spawn_node_manager(std::vector<std::string> const& extra_args = {})
  {
    port_ = find_free_port();

    nm_pid_ = fork();
    ASSERT_NE(nm_pid_, -1);
    if (nm_pid_ == 0) {
      auto port_str     = std::to_string(port_);
      auto num_gpus_str = std::to_string(num_gpus_);
      setenv("GQE_MAX_QUERY_MEMORY", "1073741824", /*overwrite=*/0);
      setenv("GQE_MAX_TASK_MANAGER_MEMORY", "1073741824", /*overwrite=*/0);
      std::vector<char const*> argv{nm_bin_,
                                    "--address",
                                    "127.0.0.1",
                                    "--port",
                                    port_str.c_str(),
                                    "--num-gpus",
                                    num_gpus_str.c_str(),
                                    "--task-manager-binary",
                                    tm_bin_};
      for (auto const& arg : extra_args) {
        argv.push_back(arg.c_str());
      }
      argv.push_back(nullptr);
      execv(nm_bin_, const_cast<char* const*>(argv.data()));
      _exit(127);
    }
  }

  /**
   * Launch a node manager with num_gpus_ GPUs and wait for it to serve. Sets nm_pid_ and port_.
   *
   * @param extra_args Further command-line arguments, such as `--query-timeout`, appended to the
   * node manager's.
   */
  void launch_node_manager(std::vector<std::string> const& extra_args = {})
  {
    ASSERT_NO_FATAL_FAILURE(spawn_node_manager(extra_args));
    ASSERT_TRUE(wait_for_port(port_, 180)) << "Node manager did not start";
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

  /** Run @p sql against the running server and assert that it succeeds. */
  void run_sql_or_fail(std::string_view sql)
  {
    auto [exit_code, output] = run_sql(client_bin_, port_, sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
  }

  /**
   * Poison the segment allocator lock, run @p sql, and check that the node tears down.
   *
   * The statement must fail with the node-teardown error. The node manager must then exit with
   * status 101 without a signal, leaving neither the segment nor any task manager behind.
   */
  void expect_teardown_after_poisoning(std::string_view sql)
  {
    auto const children = find_child_pids(nm_pid_);
    ASSERT_FALSE(children.empty()) << "No task manager children found";

    ASSERT_NO_FATAL_FAILURE(poison_segment_allocator_lock());

    auto [exit_code, output] = run_sql(client_bin_, port_, sql);
    EXPECT_NE(exit_code, 0) << sql << " succeeded on a poisoned segment. Output:\n" << output;
    EXPECT_NE(output.find("shutting down because the shared memory is unrecoverable"),
              std::string::npos)
      << sql << " did not report the node teardown. Output:\n"
      << output;

    int status          = 0;
    pid_t result        = 0;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (result == 0 && std::chrono::steady_clock::now() < deadline) {
      result = waitpid(nm_pid_, &status, WNOHANG);
      if (result == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    }
    ASSERT_EQ(result, nm_pid_) << "Node manager did not exit within 60 seconds of the poisoning";
    nm_pid_ = -1;  // Reaped, so TearDown must not signal it.

    ASSERT_TRUE(WIFEXITED(status)) << "Node manager did not exit normally, status " << status;
    EXPECT_EQ(WEXITSTATUS(status), gqe::node_manager::exit_code::shared_memory_unrecoverable);
    EXPECT_FALSE(std::filesystem::exists("/dev/shm/gqe_shared_memory"))
      << "The node manager did not remove the shared-memory segment";
    for (auto child : children) {
      if (is_process_running_binary(child, tm_bin_)) {
        kill(child, SIGKILL);
        ADD_FAILURE() << "Task manager (pid " << child << ") survived the node teardown";
      }
    }
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
 * Kill a task manager. A query issued right after should fail (not hang), and a subsequent query
 * after the node manager has replaced the task manager group should succeed.
 */
TEST_P(FlightSqlLifecycleTest, TaskManagerCrash)
{
  if (!data_path_ || !queries_dir_) {
    GTEST_SKIP() << "TPCH_DATA_PATH and TPCH_QUERIES must be set";
  }

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
  std::string last_output;
  for (int attempt = 0; attempt < 6; ++attempt) {
    usleep(5'000'000);
    auto [rc2, output2] = run_tpch_query(run_script_, InputFormat::Sql, port_, queries_dir_, 6);
    if (rc2 == 0) {
      recovered = true;
      break;
    }
    last_output = std::move(output2);
  }
  EXPECT_TRUE(recovered) << "Query should succeed after recovery (tried for 30s). Last output:\n"
                         << last_output;
}

/** Sets an environment variable for the lifetime of the object, then unsets it. */
class scoped_env {
 public:
  scoped_env(char const* name, char const* value) : _name(name) { setenv(name, value, 1); }
  ~scoped_env() { unsetenv(_name); }
  scoped_env(scoped_env const&)            = delete;
  scoped_env& operator=(scoped_env const&) = delete;

 private:
  char const* _name;
};

/** Runs a function when the object goes out of scope, including when an `ASSERT_*` returns early.
 */
class scope_exit {
 public:
  explicit scope_exit(std::function<void()> function) : _function(std::move(function)) {}
  ~scope_exit() { _function(); }

  scope_exit(scope_exit const&)            = delete;
  scope_exit& operator=(scope_exit const&) = delete;

 private:
  std::function<void()> _function;
};

/** Removes a file when the object goes out of scope, including when an `ASSERT_*` returns early. */
class scoped_file_removal {
 public:
  explicit scoped_file_removal(std::filesystem::path path) : _path(std::move(path)) {}
  ~scoped_file_removal()
  {
    std::error_code ignored;
    std::filesystem::remove(_path, ignored);
  }
  scoped_file_removal(scoped_file_removal const&)            = delete;
  scoped_file_removal& operator=(scoped_file_removal const&) = delete;

 private:
  std::filesystem::path _path;
};

/**
 * A query that arrives while the task managers are being replaced waits for the replacement
 * instead of failing, also when the first start attempt fails.
 *
 * A task manager is killed, and every start attempt fails until the trigger file is removed. The
 * failure comes from the task manager's `GQE_TEST_FAIL_SET_PEER_INFO_FILE` hook, which fails
 * `SetPeerInfo` on the listed ranks after applying the peer info. Listing the last rank means
 * every rank has its peer info when bootstrap fails, because the node manager sends `SetPeerInfo`
 * to the ranks in order. The hook first sleeps for `attempt_delay`, so the attempt is still in
 * progress when the query arrives.
 *
 * The query must succeed, and it must have waited: it cannot have run before the replacement
 * group was ready. The node manager stays up, and the next query works.
 *
 * The hooks are compiled only with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by
 * default, and the test skips without it. To run the test locally, reconfigure the build with
 * `cmake -DGQE_ENABLE_FAULT_INJECTION=ON .` and rebuild.
 */
TEST_P(FlightSqlLifecycleTest, QueryWaitsForTaskManagerRestart)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const trigger_file =
    std::filesystem::temp_directory_path() / std::format("gqe_fail_set_peer_info_{}", getpid());
  std::filesystem::remove(trigger_file);
  scoped_file_removal remove_trigger_file(trigger_file);
  scoped_env fault_injection("GQE_TEST_FAIL_SET_PEER_INFO_FILE", trigger_file.c_str());

  launch_node_manager();

  auto const query = [&] { return run_sql(client_bin_, port_, "SELECT x FROM restart_probe"); };
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE restart_probe (x BIGINT);"));
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("SELECT x FROM restart_probe"));

  // From now on, a start attempt fails after `attempt_delay`, until the file is removed.
  constexpr auto attempt_delay = std::chrono::seconds(8);
  std::ofstream(trigger_file) << num_gpus_ - 1 << ' ' << attempt_delay.count() << '\n';

  auto children = find_child_pids(nm_pid_);
  ASSERT_FALSE(children.empty()) << "No task manager children found";
  kill(children[0], SIGKILL);

  // Let the monitor notice the exit, so that the query arrives while the group is being replaced
  // and not while the dead task manager is still the current group.
  std::this_thread::sleep_for(std::chrono::seconds(3));
  auto pending_query = std::async(std::launch::async, query);

  // The query must still be waiting when the failing attempt is over. Then let the next attempt
  // succeed, after the one in progress read the trigger file and failed.
  EXPECT_EQ(pending_query.wait_for(attempt_delay), std::future_status::timeout)
    << "The query ran before the failing attempt was over";
  std::filesystem::remove(trigger_file);

  auto [exit_code, output] = pending_query.get();
  EXPECT_EQ(exit_code, 0) << "The query did not wait for the replacement group. Output:\n"
                          << output;

  // The node manager stayed up and serves again.
  EXPECT_TRUE(is_process_running(nm_pid_));
  auto [next_exit_code, next_output] = query();
  EXPECT_EQ(next_exit_code, 0) << "A query after the restart failed. Output:\n" << next_output;
}

/**
 * A node whose task managers cannot be brought back exits with its own status instead of staying
 * up without task managers.
 *
 * A task manager is killed, and every start attempt fails, through the same hook as in
 * `QueryWaitsForTaskManagerRestart` but without a delay. After three attempts the node manager
 * tears down. A query that arrived meanwhile fails with an error that names the cause, and the
 * node manager exits with `exit_code::task_managers_unavailable`, leaving neither the segment nor
 * any task manager behind.
 */
TEST_P(FlightSqlLifecycleTest, ExitsWhenTaskManagersCannotBeRestarted)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const trigger_file =
    std::filesystem::temp_directory_path() / std::format("gqe_fail_set_peer_info_{}", getpid());
  std::filesystem::remove(trigger_file);
  scoped_file_removal remove_trigger_file(trigger_file);
  scoped_env fault_injection("GQE_TEST_FAIL_SET_PEER_INFO_FILE", trigger_file.c_str());

  launch_node_manager();

  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE restart_probe (x BIGINT);"));
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("SELECT x FROM restart_probe"));

  auto const children = find_child_pids(nm_pid_);
  ASSERT_FALSE(children.empty()) << "No task manager children found";

  // Every start attempt fails from now on.
  std::ofstream(trigger_file) << num_gpus_ - 1 << '\n';
  kill(children[0], SIGKILL);

  // The monitor notices the exit within a second. The query then waits for a group, and fails when
  // the last attempt has failed and the node tears down.
  std::this_thread::sleep_for(std::chrono::seconds(3));
  auto [exit_code, output] = run_sql(client_bin_, port_, "SELECT x FROM restart_probe");
  EXPECT_NE(exit_code, 0) << "The query succeeded although no task manager could be started. "
                             "Output:\n"
                          << output;
  EXPECT_NE(output.find("task managers could not be restarted"), std::string::npos)
    << "The query did not report why the node is shutting down. Output:\n"
    << output;
  EXPECT_EQ(output.find("Unavailable"), std::string::npos)
    << "A node that is exiting must not report Unavailable, which invites a retry. Output:\n"
    << output;

  int status          = 0;
  pid_t result        = 0;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (result == 0 && std::chrono::steady_clock::now() < deadline) {
    result = waitpid(nm_pid_, &status, WNOHANG);
    if (result == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
  }
  ASSERT_EQ(result, nm_pid_) << "Node manager did not exit within 60 seconds of the last attempt";
  nm_pid_ = -1;  // Reaped, so TearDown must not signal it.

  ASSERT_TRUE(WIFEXITED(status)) << "Node manager did not exit normally, status " << status;
  EXPECT_EQ(WEXITSTATUS(status), gqe::node_manager::exit_code::task_managers_unavailable);
  EXPECT_FALSE(std::filesystem::exists("/dev/shm/gqe_shared_memory"))
    << "The node manager did not remove the shared-memory segment";
  for (auto child : children) {
    if (is_process_running_binary(child, tm_bin_)) {
      kill(child, SIGKILL);
      ADD_FAILURE() << "Task manager (pid " << child << ") survived the node teardown";
    }
  }
}

/**
 * A node whose first task managers cannot be started exits with its own status, and never serves.
 *
 * Every start attempt fails from the first one, on the last rank, through the task manager's
 * `GQE_TEST_FAIL_SET_PEER_INFO_FILE` hook, which is compiled only with the CMake option
 * `GQE_ENABLE_FAULT_INJECTION`. The hook is armed before the node manager launches. After three
 * attempts the node manager must exit with `exit_code::task_managers_unavailable`, leaving no
 * segment behind. The restart path has its own test (`ExitsWhenTaskManagersCannotBeRestarted`).
 */
TEST_P(FlightSqlLifecycleTest, ExitsWhenTheFirstTaskManagersCannotBeStarted)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const trigger_file =
    std::filesystem::temp_directory_path() / std::format("gqe_fail_first_start_{}", getpid());
  std::filesystem::remove(trigger_file);
  scoped_file_removal remove_trigger_file(trigger_file);
  scoped_env fault_injection("GQE_TEST_FAIL_SET_PEER_INFO_FILE", trigger_file.c_str());
  std::ofstream(trigger_file) << num_gpus_ - 1 << '\n';

  ASSERT_NO_FATAL_FAILURE(spawn_node_manager());

  // Three attempts take a few seconds each, plus the 1 s and 2 s waits between them.
  int status          = 0;
  pid_t result        = 0;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (result == 0 && std::chrono::steady_clock::now() < deadline) {
    result = waitpid(nm_pid_, &status, WNOHANG);
    if (result == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
  }
  ASSERT_EQ(result, nm_pid_) << "Node manager did not exit within 120 seconds";
  nm_pid_ = -1;  // Reaped, so TearDown must not signal it.

  ASSERT_TRUE(WIFEXITED(status)) << "Node manager did not exit normally, status " << status;
  EXPECT_EQ(WEXITSTATUS(status), gqe::node_manager::exit_code::task_managers_unavailable);
  EXPECT_FALSE(std::filesystem::exists("/dev/shm/gqe_shared_memory"))
    << "The node manager did not remove the shared-memory segment";
}

/**
 * A bootstrap in which one rank fails must fail at once, not after the deadline of each rank in
 * turn. A task manager is killed and every start attempt then fails on the last rank, before that
 * rank joins the NVSHMEM collective, through the task manager's
 * `GQE_TEST_FAIL_INIT_NVSHMEM_FILE` hook. With several ranks the other ranks wait in the collective
 * for it. The node manager must act on the first failure, kill the group without a grace period
 * and, after the last attempt, exit with the status for unavailable task managers.
 *
 * Waiting for every rank in turn would take the 180 s bootstrap deadline for each attempt, so the
 * node manager has to exit long before that. With one rank nothing waits in a collective, and the
 * test then checks only the attempts and the exit status.
 */
TEST_P(FlightSqlLifecycleTest, BootstrapFailsFastWhenOneRankFails)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const trigger_file =
    std::filesystem::temp_directory_path() / std::format("gqe_fail_init_nvshmem_{}", getpid());
  std::filesystem::remove(trigger_file);
  scoped_file_removal remove_trigger_file(trigger_file);
  scoped_env fault_injection("GQE_TEST_FAIL_INIT_NVSHMEM_FILE", trigger_file.c_str());

  launch_node_manager();

  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE restart_probe (x BIGINT);"));
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("SELECT x FROM restart_probe"));

  auto const children = find_child_pids(nm_pid_);
  ASSERT_FALSE(children.empty()) << "No task manager children found";

  // Every start attempt fails from now on, on the last rank.
  std::ofstream(trigger_file) << num_gpus_ - 1 << '\n';
  auto const killed_at = std::chrono::steady_clock::now();
  kill(children[0], SIGKILL);

  int status          = 0;
  pid_t result        = 0;
  auto const deadline = killed_at + std::chrono::seconds(150);
  while (result == 0 && std::chrono::steady_clock::now() < deadline) {
    result = waitpid(nm_pid_, &status, WNOHANG);
    if (result == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
  }
  auto const exited_after = std::chrono::steady_clock::now() - killed_at;
  ASSERT_EQ(result, nm_pid_)
    << "Node manager did not exit within 150 seconds. A bootstrap that "
       "waits for every rank's deadline takes about 180 seconds per attempt";
  nm_pid_ = -1;  // Reaped, so TearDown must not signal it.

  // Three attempts take a few seconds each, plus the 1 s and 2 s waits between them. Waiting for
  // the bootstrap deadline would take minutes.
  EXPECT_LT(exited_after, std::chrono::seconds(100))
    << "The node manager did not act on the first failed rank of the bootstrap";

  ASSERT_TRUE(WIFEXITED(status)) << "Node manager did not exit normally, status " << status;
  EXPECT_EQ(WEXITSTATUS(status), gqe::node_manager::exit_code::task_managers_unavailable);
  EXPECT_FALSE(std::filesystem::exists("/dev/shm/gqe_shared_memory"))
    << "The node manager did not remove the shared-memory segment";
  for (auto child : children) {
    if (is_process_running_binary(child, tm_bin_)) {
      kill(child, SIGKILL);
      ADD_FAILURE() << "Task manager (pid " << child << ") survived the node teardown";
    }
  }
}

/**
 * An operation that registers its task-manager RPC after node teardown began must not hold up the
 * teardown. The handler is paused just before it registers the RPC, the segment's poisoned flag is
 * set so that the monitor begins teardown and cancels the registered RPCs (of which there are
 * none), and the handler then resumes. If it dispatched anyway, the stalled rank would keep the RPC
 * running and `Shutdown()` would wait for it. The RPC must instead be cancelled before it is
 * dispatched, so the node manager exits long before the stall ends.
 *
 * The test uses two hooks, both compiled only with the CMake option `GQE_ENABLE_FAULT_INJECTION`.
 * `GQE_TEST_PAUSE_BEFORE_RPC_REGISTRATION_FILE` pauses the node manager before it registers the
 * RPC of the listed rank, and `GQE_TEST_STALL_EXECUTE_PLAN_FILE` stalls the listed rank before it
 * runs a plan. The test skips without the option. See `QueryRejectedWhileRestartPending` for how to
 * enable it.
 */
TEST_P(FlightSqlLifecycleTest, TeardownDoesNotWaitForRpcRegisteredAfterCancellation)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const temp_dir   = std::filesystem::temp_directory_path();
  auto const pause_file = temp_dir / std::format("gqe_pause_before_rpc_registration_{}", getpid());
  auto const stall_file = temp_dir / std::format("gqe_stall_execute_plan_{}", getpid());
  std::filesystem::remove(pause_file);
  std::filesystem::remove(stall_file);
  scoped_file_removal remove_pause_file(pause_file);
  scoped_file_removal remove_stall_file(stall_file);
  scoped_env pause_hook("GQE_TEST_PAUSE_BEFORE_RPC_REGISTRATION_FILE", pause_file.c_str());
  scoped_env stall_hook("GQE_TEST_STALL_EXECUTE_PLAN_FILE", stall_file.c_str());

  launch_node_manager();

  auto const children = find_child_pids(nm_pid_);
  ASSERT_FALSE(children.empty()) << "No task manager children found";

  // A query that does not read a statistics table registers only the RPC that executes its plan.
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE registration_probe (x BIGINT);"));

  // The pause must outlast the monitor's detection of the poisoning, which polls once a second.
  // The stall must far outlast the bound below, so that a dispatched RPC is noticed.
  constexpr auto registration_pause = std::chrono::seconds(8);
  constexpr auto execute_stall      = std::chrono::seconds(300);
  constexpr auto exit_bound         = std::chrono::seconds(60);
  std::ofstream(pause_file) << "0 " << registration_pause.count() << '\n';
  std::ofstream(stall_file) << "0 " << execute_stall.count() << '\n';

  auto query = std::async(std::launch::async, [&] {
    return run_sql(client_bin_, port_, "SELECT x FROM registration_probe");
  });
  // An assertion below can end the test while the node manager is still waiting for the stalled
  // rank. The pending query joins when the test returns, which would take the stall time, far
  // longer than the bound. This runs before the query's destructor, so it stops the node manager
  // and its task managers first.
  scope_exit stop_stalled_node([&] {
    if (nm_pid_ > 0) {
      kill(nm_pid_, SIGKILL);
      waitpid(nm_pid_, nullptr, 0);
      nm_pid_ = -1;
    }
    for (auto child : children) {
      if (is_process_running_binary(child, tm_bin_)) { kill(child, SIGKILL); }
    }
  });

  // Let the handler reach the pause, then start the teardown while it is paused. The handler holds
  // the execution mutex, so no statement can run to observe a failed lock. The monitor notices the
  // flag instead.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  ASSERT_NO_FATAL_FAILURE(mark_segment_poisoned());
  auto const poisoned_at = std::chrono::steady_clock::now();

  int status          = 0;
  pid_t result        = 0;
  auto const deadline = poisoned_at + exit_bound;
  while (result == 0 && std::chrono::steady_clock::now() < deadline) {
    result = waitpid(nm_pid_, &status, WNOHANG);
    if (result == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
  }
  auto const exit_time = std::chrono::steady_clock::now() - poisoned_at;
  ASSERT_EQ(result, nm_pid_)
    << "Node manager did not exit within " << exit_bound.count()
    << " seconds of the poisoning. The RPC registered after the teardown began was probably "
       "dispatched and kept Shutdown() waiting for the stalled rank.";
  nm_pid_ = -1;  // Reaped, so TearDown must not signal it.

  ASSERT_TRUE(WIFEXITED(status)) << "Node manager did not exit normally, status " << status;
  EXPECT_EQ(WEXITSTATUS(status), gqe::node_manager::exit_code::shared_memory_unrecoverable);
  EXPECT_LT(exit_time, exit_bound);

  auto [exit_code, output] = query.get();
  EXPECT_NE(exit_code, 0) << "The query succeeded although the node was torn down. Output:\n"
                          << output;

  for (auto child : children) {
    if (is_process_running_binary(child, tm_bin_)) {
      kill(child, SIGKILL);
      ADD_FAILURE() << "Task manager (pid " << child << ") survived the node teardown";
    }
  }
}

/**
 * A write whose completion is unconfirmed terminates the task-manager group and leaves its table
 * unusable, although nothing was published when the node manager gave up on it.
 *
 * A rank is stalled before it publishes, the RPC deadline expires, and the node manager cannot tell
 * whether the rank will still publish. The stalled rank must be gone before the error reaches the
 * client, so that it cannot publish later. The error is `Unavailable`, which tells the client that
 * the task managers are being restarted. The table's storage revision is unchanged at that point,
 * so the flag must not depend on it. Reads and writes of the table are refused until it is
 * dropped, and the table can be dropped at once, because no rank can still append to it. A table
 * loaded afterwards, on the replacement group, holds only its own rows.
 *
 * The stall comes from the task manager's `GQE_TEST_STALL_EXECUTE_PLAN_FILE` hook, and the
 * deadline from a short `--query-timeout`. The hook is compiled only with the CMake option
 * `GQE_ENABLE_FAULT_INJECTION`, and the test skips without it, and without `TPCH_DATA_PATH`.
 */
TEST_P(FlightSqlLifecycleTest, UnconfirmedCompletionTerminatesGroupAndLeavesTableUnusable)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  if (!data_path_) { GTEST_SKIP() << "TPCH_DATA_PATH must be set"; }

  auto const stall_file =
    std::filesystem::temp_directory_path() / std::format("gqe_stall_execute_plan_{}", getpid());
  std::filesystem::remove(stall_file);
  scoped_file_removal remove_stall_file(stall_file);
  scoped_env stall_hook("GQE_TEST_STALL_EXECUTE_PLAN_FILE", stall_file.c_str());

  // The timeout leaves the healthy queries after the restart, which are the first on the new group
  // and pay its start-up costs, room to finish. The stall still outlasts it.
  constexpr auto query_timeout = std::chrono::seconds(5);
  constexpr auto stall         = std::chrono::seconds(10);
  launch_node_manager({"--query-timeout", std::to_string(query_timeout.count())});

  auto const create_region =
    "CREATE TABLE region (r_regionkey BIGINT NOT NULL, r_name VARCHAR NOT NULL, "
    "r_comment VARCHAR NOT NULL);";
  auto const copy_region =
    std::format("COPY region FROM '{}/region' (FORMAT parquet);", data_path_);
  auto const select_region = "SELECT r_regionkey FROM region";
  auto const refusal       = "incomplete after a failed write";

  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail(create_region));
  auto const original_task_managers = find_child_pids(nm_pid_);
  ASSERT_FALSE(original_task_managers.empty()) << "No task manager children found";

  // Rank 0 stalls for longer than the deadline, before it has published anything.
  std::ofstream(stall_file) << "0 " << stall.count() << '\n';
  auto const copy_start = std::chrono::steady_clock::now();
  {
    auto [exit_code, output] = run_sql(client_bin_, port_, copy_region);
    EXPECT_NE(exit_code, 0) << "The COPY succeeded although a rank stalled past the deadline. "
                               "Output:\n"
                            << output;
    EXPECT_NE(output.find("Unavailable"), std::string::npos)
      << "The failure of the task managers was not reported as Unavailable. Output:\n"
      << output;
    EXPECT_NE(output.find("being restarted"), std::string::npos)
      << "The error does not say that the task managers are being restarted. Output:\n"
      << output;
  }
  // The stalled rank was terminated before the error was reported, long before its stall ended.
  EXPECT_LT(std::chrono::steady_clock::now() - copy_start, stall)
    << "The node manager waited for the stalled rank instead of terminating the group";
  for (auto const pid : original_task_managers) {
    EXPECT_FALSE(is_process_running_binary(pid, tm_bin_))
      << "Task manager (pid " << pid << ") still runs after its group was terminated";
  }
  // Later operations must not stall, and must be refused before they reach a task manager.
  std::filesystem::remove(stall_file);

  {
    auto [exit_code, output] = run_sql(client_bin_, port_, select_region);
    EXPECT_NE(exit_code, 0) << "A read of the table succeeded after an unconfirmed write. Output:\n"
                            << output;
    EXPECT_NE(output.find(refusal), std::string::npos)
      << "The read was not refused as incomplete. Output:\n"
      << output;
  }
  {
    auto [exit_code, output] = run_sql(client_bin_, port_, copy_region);
    EXPECT_NE(exit_code, 0) << "A write to the table succeeded after an unconfirmed write. "
                               "Output:\n"
                            << output;
    EXPECT_NE(output.find(refusal), std::string::npos)
      << "The write was not refused as incomplete. Output:\n"
      << output;
  }

  // No rank can still append to the table, so it can be dropped at once.
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("DROP TABLE region;"));

  // A table loaded afterwards waits for the replacement group, is usable, and holds only its own
  // rows. The five rows of `region` are all the stalled COPY could have added, and it must not have
  // added them to this table.
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail(create_region));
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail(copy_region));
  auto const replacement_task_managers = find_child_pids(nm_pid_);
  EXPECT_FALSE(replacement_task_managers.empty());
  for (auto const pid : replacement_task_managers) {
    EXPECT_EQ(std::count(original_task_managers.begin(), original_task_managers.end(), pid), 0)
      << "The group was not replaced: task manager pid " << pid << " is an original one";
  }

  // The terminated rank cannot publish after its stall would have ended, so nothing changes.
  std::this_thread::sleep_until(copy_start + stall + std::chrono::seconds(3));
  auto [exit_code, output] = run_sql(client_bin_, port_, select_region);
  EXPECT_EQ(exit_code, 0) << "Reading the reloaded table failed. Output:\n" << output;
  EXPECT_NE(output.find("5 rows"), std::string::npos)
    << "The reloaded table does not hold exactly the five rows of `region`. Output:\n"
    << output;
  EXPECT_EQ(find_child_pids(nm_pid_), replacement_task_managers)
    << "The replacement task managers changed, so something failed them";
}

/**
 * A statistics refresh whose completion is unconfirmed kills the task-manager group, like an
 * unconfirmed plan execution.
 *
 * A query that reads a statistics table refreshes them on rank 0 first, if they are stale. Rank 0
 * stalls the refresh past the RPC deadline, so the refresh is unconfirmed, and rank 0
 * may still be writing the statistics tables. The group is killed, the query fails with
 * `Unavailable`, and the next query waits for the replacement group and succeeds.
 *
 * The stall comes from the task manager's `GQE_TEST_STALL_REFRESH_STATS_FILE` hook, and the
 * deadline from a short `--query-timeout`. The hook is compiled only with the CMake option
 * `GQE_ENABLE_FAULT_INJECTION`, and the test skips without it. Needs no TPC-H data.
 */
TEST_P(FlightSqlLifecycleTest, UnconfirmedStatisticsRefreshKillsGroup)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  auto const stall_file =
    std::filesystem::temp_directory_path() / std::format("gqe_stall_refresh_stats_{}", getpid());
  std::filesystem::remove(stall_file);
  scoped_file_removal remove_stall_file(stall_file);
  scoped_env stall_hook("GQE_TEST_STALL_REFRESH_STATS_FILE", stall_file.c_str());

  // The timeout leaves the healthy queries after the restart, which are the first on the new group
  // and pay its start-up costs, room to finish. The stall still outlasts it.
  constexpr auto query_timeout = std::chrono::seconds(5);
  constexpr auto stall         = std::chrono::seconds(30);
  launch_node_manager({"--query-timeout", std::to_string(query_timeout.count())});

  auto const original_task_managers = find_child_pids(nm_pid_);
  ASSERT_FALSE(original_task_managers.empty()) << "No task manager children found";

  // CREATE TABLE makes the statistics stale. Only a query that reads a statistics table refreshes
  // them first.
  auto const read_statistics = "SELECT table_name FROM _gqe_table_stats";
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE refresh_probe (x BIGINT);"));
  std::ofstream(stall_file) << "0 " << stall.count() << '\n';

  auto const start = std::chrono::steady_clock::now();
  {
    auto [exit_code, output] = run_sql(client_bin_, port_, read_statistics);
    EXPECT_NE(exit_code, 0) << "The query succeeded although the statistics refresh stalled past "
                               "the deadline. Output:\n"
                            << output;
    EXPECT_NE(output.find("Unavailable"), std::string::npos)
      << "The failure of the task managers was not reported as Unavailable. Output:\n"
      << output;
    EXPECT_NE(output.find("being restarted"), std::string::npos)
      << "The error does not say that the task managers are being restarted. Output:\n"
      << output;
  }
  // The group was killed at the deadline, long before the stall would have ended.
  EXPECT_LT(std::chrono::steady_clock::now() - start, stall)
    << "The node manager waited for the stalled rank instead of killing the group";
  for (auto const pid : original_task_managers) {
    EXPECT_FALSE(is_process_running_binary(pid, tm_bin_))
      << "Task manager (pid " << pid << ") still runs after its group was killed";
  }

  // The replacement group serves the next query, which refreshes the statistics without a stall.
  std::filesystem::remove(stall_file);
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail(read_statistics));
  auto const replacement_task_managers = find_child_pids(nm_pid_);
  EXPECT_FALSE(replacement_task_managers.empty());
  for (auto const pid : replacement_task_managers) {
    EXPECT_EQ(std::count(original_task_managers.begin(), original_task_managers.end(), pid), 0)
      << "The group was not replaced: task manager pid " << pid << " is an original one";
  }
}

/**
 * A DROP TABLE after the segment allocator lock was poisoned. The node manager's own deallocation
 * observes the unrecoverable lock, and the node tears down. Needs no TPC-H data.
 */
TEST_P(FlightSqlLifecycleTest, DropTearsDownNodeOnPoisonedSharedMemory)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  launch_node_manager();
  ASSERT_NO_FATAL_FAILURE(run_sql_or_fail("CREATE TABLE poison_probe (x BIGINT);"));

  expect_teardown_after_poisoning("DROP TABLE poison_probe;");
}

/**
 * A COPY after the segment allocator lock was poisoned. The task managers observe the unrecoverable
 * lock and report `node_fatal`, and the node tears down.
 */
TEST_P(FlightSqlLifecycleTest, CopyTearsDownNodeOnPoisonedSharedMemory)
{
#ifndef GQE_ENABLE_FAULT_INJECTION
  GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
  if (!data_path_) { GTEST_SKIP() << "TPCH_DATA_PATH must be set"; }

  launch_node_manager();
  ASSERT_NO_FATAL_FAILURE(
    run_sql_or_fail("CREATE TABLE region (r_regionkey BIGINT NOT NULL, r_name VARCHAR NOT NULL, "
                    "r_comment VARCHAR NOT NULL);"));

  expect_teardown_after_poisoning(
    std::format("COPY region FROM '{}/region' (FORMAT parquet);", data_path_));
}

}  // namespace gqe_test
