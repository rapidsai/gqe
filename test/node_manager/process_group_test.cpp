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

#include <gqe/node_manager/spawn.hpp>

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>
#include <string>

extern char** environ;

namespace gqe::node_manager {
namespace {

/// Sleep duration for child processes that are expected to be killed before they exit.
constexpr int sleeper_seconds = 3;

/// Grace period for a process to exit or receive a signal before we check its status.
constexpr auto reap_grace_period = std::chrono::milliseconds(50);

/// Spawn a child process that sleeps for @p seconds.
pid_t spawn_sleeper(int seconds)
{
  auto sec_str = std::to_string(seconds);
  char* argv[] = {const_cast<char*>("/bin/sleep"), sec_str.data(), nullptr};

  pid_t pid;
  int err = posix_spawn(&pid, "/bin/sleep", nullptr, nullptr, argv, environ);
  if (err != 0) { throw std::runtime_error("posix_spawn failed"); }
  return pid;
}

/// A TCP port that was free a moment ago, so that no task manager answers on it.
int unused_port()
{
  int const fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { throw std::runtime_error("socket failed"); }
  sockaddr_in address{};
  address.sin_family      = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t length        = sizeof(address);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), length) != 0 ||
      getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    close(fd);
    throw std::runtime_error("Could not find an unused port");
  }
  close(fd);
  return ntohs(address.sin_port);
}

TEST(ProcessGroupTest, EmptyGroupIsValid)
{
  process_group group;
  EXPECT_EQ(group.size(), 0);
  EXPECT_FALSE(group.try_wait_any().has_value());
}

TEST(ProcessGroupTest, AddAndSize)
{
  process_group group;
  group.add(spawn_sleeper(sleeper_seconds));
  EXPECT_EQ(group.size(), 1);

  group.add(spawn_sleeper(sleeper_seconds));
  EXPECT_EQ(group.size(), 2);
  // Destructor cleans up.
}

TEST(ProcessGroupTest, AddInvalidPidThrows)
{
  process_group group;
  EXPECT_THROW(group.add(0), std::invalid_argument);
  EXPECT_THROW(group.add(-1), std::invalid_argument);
}

TEST(ProcessGroupTest, PidReturnsCorrectValue)
{
  process_group group;
  auto pid = spawn_sleeper(sleeper_seconds);
  group.add(pid);
  EXPECT_EQ(group.pid(0), pid);
}

TEST(ProcessGroupTest, TryWaitAnyDetectsExitedProcess)
{
  process_group group;
  // Spawn a process that exits immediately.
  group.add(spawn_sleeper(0));
  // Give it a moment to exit.
  std::this_thread::sleep_for(reap_grace_period);

  auto status = group.try_wait_any();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->rank, 0);
  EXPECT_EQ(status->kind, exit_kind::exited);
  EXPECT_EQ(status->code, 0);
}

TEST(ProcessGroupTest, TryWaitAnyReturnsNulloptForRunningProcesses)
{
  process_group group;
  group.add(spawn_sleeper(sleeper_seconds));

  auto status = group.try_wait_any();
  EXPECT_FALSE(status.has_value());
}

TEST(ProcessGroupTest, SignalAllSendsSignal)
{
  process_group group;
  group.add(spawn_sleeper(sleeper_seconds));
  group.add(spawn_sleeper(sleeper_seconds));

  group.signal_all(SIGTERM);

  // Give processes time to receive the signal.
  std::this_thread::sleep_for(reap_grace_period);

  // Both should have exited via signal.
  auto s1 = group.try_wait_any();
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(s1->kind, exit_kind::signaled);
  EXPECT_EQ(s1->code, SIGTERM);

  auto s2 = group.try_wait_any();
  ASSERT_TRUE(s2.has_value());
  EXPECT_EQ(s2->kind, exit_kind::signaled);
  EXPECT_EQ(s2->code, SIGTERM);
}

TEST(ProcessGroupTest, TryWaitAnyReportsExited)
{
  process_group group;
  group.add(spawn_sleeper(0));
  std::this_thread::sleep_for(reap_grace_period);

  auto status = group.try_wait_any();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->kind, exit_kind::exited);
  EXPECT_EQ(status->code, 0);
}

TEST(ProcessGroupTest, TryWaitAnyReportsSignaled)
{
  process_group group;
  auto pid = spawn_sleeper(sleeper_seconds);
  group.add(pid);

  kill(pid, SIGKILL);
  std::this_thread::sleep_for(reap_grace_period);

  auto status = group.try_wait_any();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->kind, exit_kind::signaled);
  EXPECT_EQ(status->code, SIGKILL);
}

TEST(ProcessGroupTest, WaitAllBlocksUntilExit)
{
  process_group group;
  group.add(spawn_sleeper(sleeper_seconds));

  group.signal_all(SIGTERM);
  group.wait_all();

  EXPECT_EQ(group.size(), 0);
}

TEST(ProcessGroupTest, TerminateAllKillsRunningProcesses)
{
  process_group group;
  auto pid1 = spawn_sleeper(sleeper_seconds);
  auto pid2 = spawn_sleeper(sleeper_seconds);
  group.add(pid1);
  group.add(pid2);

  group.terminate_all(std::chrono::seconds(2));
  EXPECT_EQ(group.size(), 0);

  // Both processes should have been reaped — waitpid must fail with ECHILD.
  int status;
  EXPECT_EQ(waitpid(pid1, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
  EXPECT_EQ(waitpid(pid2, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
}

TEST(ProcessGroupTest, DestructorCleansUpProcesses)
{
  pid_t pid;
  {
    process_group group;
    pid = spawn_sleeper(sleeper_seconds);
    group.add(pid);
    // group goes out of scope here.
  }
  // The process should have been reaped. waitpid should fail with ECHILD.
  int status;
  EXPECT_EQ(waitpid(pid, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
}

TEST(ProcessGroupTest, MoveTransfersOwnership)
{
  pid_t pid;
  {
    process_group group;
    pid = spawn_sleeper(sleeper_seconds);
    group.add(pid);

    process_group moved = std::move(group);
    EXPECT_EQ(moved.size(), 1);
    EXPECT_EQ(moved.pid(0), pid);
    // moved goes out of scope and cleans up.
  }
  int status;
  EXPECT_EQ(waitpid(pid, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
}

// A task manager that has exited never answers its health check. The start must fail when the exit
// is noticed, not after every poll of the health check has run out, which takes about 30 s per
// rank.
TEST(SpawnTaskManagersTest, FailsAtOnceWhenATaskManagerExitsDuringStartup)
{
  auto const start = std::chrono::steady_clock::now();
  try {
    [[maybe_unused]] auto const spawned =
      spawn_task_managers("/bin/false", unused_port(), 1, "127.0.0.1:1");
    FAIL() << "The start succeeded although the task manager exited";
  } catch (std::runtime_error const& error) {
    std::string const message = error.what();
    EXPECT_NE(message.find("rank 0"), std::string::npos) << message;
    EXPECT_NE(message.find("exited with status 1"), std::string::npos) << message;
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(10));
}

}  // namespace
}  // namespace gqe::node_manager
