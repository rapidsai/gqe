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

#include <gqe/node_manager/task_manager_group.hpp>

#include <gtest/gtest.h>

#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace gqe::node_manager {
namespace {

/// Sleep duration for stand-in processes that the test expects to terminate before they exit.
constexpr int sleeper_seconds = 30;

/// Spawn a stand-in process that sleeps for @p seconds.
pid_t spawn_sleeper(int seconds)
{
  auto sec_str = std::to_string(seconds);
  char* argv[] = {const_cast<char*>("/bin/sleep"), sec_str.data(), nullptr};

  pid_t pid;
  if (posix_spawn(&pid, "/bin/sleep", nullptr, nullptr, argv, environ) != 0) {
    throw std::runtime_error("posix_spawn failed");
  }
  return pid;
}

/// Return whether @p pid is a child that has not been reaped yet.
bool is_unreaped_child(pid_t pid)
{
  int status = 0;
  return waitpid(pid, &status, WNOHANG) == 0;
}

/// Build the spawn result of a group of @p count stand-in task managers.
spawn_result stand_in_task_managers(std::vector<pid_t>& pids, int count, int seconds)
{
  spawn_result spawned;
  for (int rank = 0; rank < count; ++rank) {
    pids.push_back(spawn_sleeper(seconds));
    spawned.processes.add(pids.back());
  }
  return spawned;
}

}  // namespace

TEST(TaskManagerGroupTest, StartsStartingWithItsGenerationAndSize)
{
  std::vector<pid_t> pids;
  spawn_result spawned = stand_in_task_managers(pids, 2, sleeper_seconds);
  spawned.channels.resize(2);
  task_manager_group group(7, std::move(spawned));

  EXPECT_EQ(group.generation(), 7);
  EXPECT_EQ(group.size(), 2);
  EXPECT_EQ(group.current_state(), task_manager_group::state::starting);
  EXPECT_FALSE(group.is_ready());
  EXPECT_FALSE(group.is_failed());
}

TEST(TaskManagerGroupTest, BecomesReadyThenFailed)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 1, sleeper_seconds));

  group.mark_ready();
  EXPECT_TRUE(group.is_ready());

  group.mark_failed();
  EXPECT_TRUE(group.is_failed());
  EXPECT_FALSE(group.is_ready());
}

TEST(TaskManagerGroupTest, FailedGroupNeverBecomesReady)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 1, sleeper_seconds));

  group.mark_failed();
  group.mark_ready();
  EXPECT_TRUE(group.is_failed());
  EXPECT_FALSE(group.is_ready());
}

TEST(TaskManagerGroupTest, ReadyGroupStaysReadyWhenMarkedReadyAgain)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 1, sleeper_seconds));

  group.mark_ready();
  group.mark_ready();
  EXPECT_TRUE(group.is_ready());
}

TEST(TaskManagerGroupTest, TryWaitAnyReportsAnExitedRank)
{
  std::vector<pid_t> pids;
  spawn_result spawned = stand_in_task_managers(pids, 1, sleeper_seconds);
  spawned.processes.add(spawn_sleeper(0));  // Rank 1 exits at once.
  pids.push_back(spawned.processes.pid(1));
  task_manager_group group(1, std::move(spawned));

  std::optional<exit_status> status;
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!status && std::chrono::steady_clock::now() < deadline) {
    status = group.try_wait_any();
    if (!status) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
  }
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->rank, 1);
  EXPECT_EQ(status->kind, exit_kind::exited);
  // Observing an exit does not change the state. The node manager decides what to do about it.
  EXPECT_EQ(group.current_state(), task_manager_group::state::starting);
}

TEST(TaskManagerGroupTest, TryWaitAnyIsEmptyWhileEveryRankRuns)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 2, sleeper_seconds));
  EXPECT_FALSE(group.try_wait_any().has_value());
}

TEST(TaskManagerGroupTest, TerminateStopsTheProcessesAndMarksTheGroupFailed)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 2, sleeper_seconds));
  group.mark_ready();

  group.terminate();

  EXPECT_TRUE(group.is_failed());
  for (auto const pid : pids) {
    // terminate() reaped the children, so they no longer exist.
    EXPECT_EQ(kill(pid, 0), -1) << "pid " << pid << " is still running";
    EXPECT_EQ(errno, ESRCH);
  }
  EXPECT_FALSE(group.try_wait_any().has_value());
}

TEST(TaskManagerGroupTest, TerminateTwiceIsHarmless)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 1, sleeper_seconds));
  group.terminate();
  EXPECT_NO_THROW(group.terminate());
  EXPECT_TRUE(group.is_failed());
}

TEST(TaskManagerGroupTest, DestructorTerminatesTheProcesses)
{
  std::vector<pid_t> pids;
  {
    task_manager_group group(1, stand_in_task_managers(pids, 2, sleeper_seconds));
    for (auto const pid : pids) {
      ASSERT_TRUE(is_unreaped_child(pid));
    }
  }
  for (auto const pid : pids) {
    EXPECT_EQ(kill(pid, 0), -1) << "pid " << pid << " is still running";
    EXPECT_EQ(errno, ESRCH);
  }
}

TEST(TaskManagerGroupTest, TerminateAndTryWaitAnyMayRunConcurrently)
{
  std::vector<pid_t> pids;
  task_manager_group group(1, stand_in_task_managers(pids, 4, sleeper_seconds));

  auto monitor = std::async(std::launch::async, [&] {
    for (int i = 0; i < 200; ++i) {
      std::ignore = group.try_wait_any();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  group.terminate();
  monitor.get();

  EXPECT_TRUE(group.is_failed());
  for (auto const pid : pids) {
    EXPECT_EQ(kill(pid, 0), -1);
  }
}

}  // namespace gqe::node_manager
