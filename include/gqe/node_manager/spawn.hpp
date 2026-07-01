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

#pragma once

#include <grpcpp/channel.h>

#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe::node_manager {

/**
 * @brief How a child process terminated.
 */
enum class exit_kind {
  exited,   ///< Process called _exit() / exit().
  signaled  ///< Process was killed by a signal.
};

/**
 * @brief Decoded exit status of a child process.
 */
struct exit_status {
  std::size_t rank;  ///< Rank of the exited process.
  pid_t pid;         ///< Process ID.
  exit_kind kind;    ///< How the process terminated.
  int code;          ///< Exit code (if @c exited) or signal number (if @c signaled).
};

/**
 * @brief RAII owner for a group of child processes.
 *
 * On destruction, sends SIGTERM to every process, waits up to a timeout for
 * each to exit, and escalates to SIGKILL for stragglers. This ensures that
 * all processes drain concurrently during the SIGTERM window.
 *
 * Also provides exception safety for @ref spawn_task_managers: if a later
 * spawn or health-check fails, the destructor cleans up already-launched
 * processes.
 */
class process_group {
 public:
  process_group() = default;
  ~process_group();

  process_group(process_group&& other) noexcept : _pids(std::exchange(other._pids, {})) {}
  process_group& operator=(process_group&& other) noexcept
  {
    if (this != &other) {
      terminate_all();
      _pids = std::exchange(other._pids, {});
    }
    return *this;
  }
  process_group(process_group const&)            = delete;
  process_group& operator=(process_group const&) = delete;

  /**
   * @brief Reserve capacity for @p n processes.
   * @param n Number of processes to reserve space for.
   */
  void reserve(std::size_t n);

  /**
   * @brief Add a child process to the group.
   * @param pid Process ID. Must be > 0.
   * @throws std::invalid_argument if @p pid <= 0.
   */
  void add(pid_t pid);

  /** @brief Number of processes in the group. */
  [[nodiscard]] std::size_t size() const;

  /** @brief Process ID at a given rank. */
  [[nodiscard]] pid_t pid(std::size_t rank) const;

  /**
   * @brief Non-blocking check for a crashed/exited process.
   * @return Decoded exit status of the first exited process, or std::nullopt
   *         if all are still running.
   */
  [[nodiscard]] std::optional<exit_status> try_wait_any();

  /**
   * @brief Send a signal to every process in the group.
   * @param sig Signal number (e.g. SIGTERM).
   */
  void signal_all(int sig);

  /**
   * @brief Block until every process in the group has exited.
   *
   * Does not send any signals — the caller is responsible for signalling
   * first if needed. After this call the group is empty.
   */
  void wait_all();

  /**
   * @brief Send SIGTERM to all, wait up to @p timeout, then SIGKILL stragglers.
   *
   * After this call the group is empty.
   */
  void terminate_all(std::chrono::seconds timeout = std::chrono::seconds(5));

 private:
  std::vector<pid_t> _pids;
};

/**
 * @brief Result of spawning task manager subprocesses.
 */
struct spawn_result {
  process_group processes;                               ///< Spawned child processes.
  std::vector<std::shared_ptr<grpc::Channel>> channels;  ///< gRPC channels, indexed by rank.
};

/**
 * @brief Spawn task manager subprocesses and wait for them to become reachable.
 *
 * Forks one task manager process per GPU via `posix_spawn`, then polls each
 * process's gRPC HealthCheck endpoint until it responds.
 *
 * @param binary    Path to the gqe_task_manager executable.
 * @param base_port Base gRPC port; rank @a i listens on `base_port + i`.
 * @param num_ranks Number of task managers to spawn.
 * @return Process group and gRPC channels for all spawned processes.
 * @throws std::runtime_error if any spawn or health check fails.
 */
[[nodiscard]] spawn_result spawn_task_managers(std::string_view binary,
                                               int base_port,
                                               int num_ranks);

}  // namespace gqe::node_manager
