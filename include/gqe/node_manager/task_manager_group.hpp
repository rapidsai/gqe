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

#include <gqe/node_manager/spawn.hpp>

#include <grpcpp/channel.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace gqe::node_manager {

/**
 * @brief The error of an operation that a failing task-manager group cut short.
 *
 * The node manager replaces the group, so the client may retry the operation once the replacement
 * is ready. Flight SQL clients get it as `Unavailable`, as opposed to the `UnknownError` of an
 * operation that failed on its own account, such as a bad query. The node manager does not retry
 * the operation itself, because a retry of an operation that caused the failure would fail the
 * replacement too, and would hide the failure from the client.
 */
class task_manager_group_failed_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief One generation of task managers, which serves or fails as a unit.
 *
 * # Design
 *
 * The task managers of a node share one NVSHMEM world, and NVSHMEM cannot survive the loss of a
 * member. The node manager therefore treats them as one group: the group is either fully
 * bootstrapped and serving, or it is not used. A group is also the unit of restart. The node
 * manager never changes a group in place. It starts a new group, bootstraps it on its own, and
 * installs it only once every rank is ready. The group that is being replaced has failed and is not
 * used, and it is terminated before the new group starts. Until the new group is installed, the
 * node manager keeps the failed group in place and operations wait for the replacement.
 *
 * Dispatch takes a snapshot of the current group and keeps it for the whole operation. The group
 * outlives its replacement for as long as an operation holds it, so the channels of an operation
 * stay valid even after the group failed.
 *
 * # States
 *
 * - `starting`: the processes are spawned and the group is being bootstrapped. Dispatch does not
 *   use it.
 * - `ready`: every rank is bootstrapped. Only this state serves operations.
 * - `failed`: a rank exited, reported a fatal error, or did not confirm an operation, so the group
 *   must be replaced. It never changes state again. The processes may still be alive until
 *   @ref terminate runs.
 *
 * ## Thread Safety
 *
 * The state and the generation may be read concurrently. The processes are guarded internally, so
 * the monitor thread may call @ref try_wait_any while a Flight SQL handler calls @ref terminate.
 */
class task_manager_group {
 public:
  /// Lifecycle state, see the class documentation.
  enum class state { starting, ready, failed };

  /**
   * @brief Take ownership of freshly spawned task managers. The group starts in `starting`.
   *
   * @param generation Number of this group, which increases with every group the node manager
   * starts.
   * @param spawned The processes and their channels, indexed by rank.
   */
  task_manager_group(std::uint64_t generation, spawn_result spawned);

  /** @brief Terminate the processes that are still running. */
  ~task_manager_group();

  task_manager_group(task_manager_group const&)            = delete;
  task_manager_group& operator=(task_manager_group const&) = delete;
  task_manager_group(task_manager_group&&)                 = delete;
  task_manager_group& operator=(task_manager_group&&)      = delete;

  /** @brief Return the generation of this group. */
  [[nodiscard]] std::uint64_t generation() const noexcept { return _generation; }

  /** @brief Return the gRPC channels, indexed by rank. */
  [[nodiscard]] std::vector<std::shared_ptr<grpc::Channel>> const& channels() const noexcept
  {
    return _channels;
  }

  /** @brief Return the number of ranks. */
  [[nodiscard]] std::size_t size() const noexcept { return _channels.size(); }

  /** @brief Return the current state. */
  [[nodiscard]] state current_state() const noexcept
  {
    return _state.load(std::memory_order_acquire);
  }

  /** @brief Return whether the group is `ready`. */
  [[nodiscard]] bool is_ready() const noexcept { return current_state() == state::ready; }

  /** @brief Return whether the group is `failed`. */
  [[nodiscard]] bool is_failed() const noexcept { return current_state() == state::failed; }

  /**
   * @brief Move the group from `starting` to `ready`.
   *
   * Does nothing if the group has failed, because a failed group never serves again.
   */
  void mark_ready() noexcept;

  /**
   * @brief Move the group to `failed`, from any state.
   *
   * This does not stop the processes, see @ref terminate.
   */
  void mark_failed() noexcept;

  /**
   * @brief Check without blocking whether a rank has exited.
   *
   * @return The exit status of the first exited rank, or `std::nullopt` if every rank is running
   * or the group has been terminated.
   */
  [[nodiscard]] std::optional<exit_status> try_wait_any();

  /**
   * @brief Mark the group failed, then stop its processes.
   *
   * Sends SIGTERM to every rank, waits up to @p timeout for them to exit, and kills the ones that
   * remain. After this call the group has no processes. Calling it again does nothing.
   *
   * @param timeout How long the ranks get to exit after SIGTERM.
   */
  void terminate(std::chrono::seconds timeout = std::chrono::seconds(5));

 private:
  std::uint64_t _generation;
  std::vector<std::shared_ptr<grpc::Channel>> _channels;
  std::atomic<state> _state{state::starting};

  std::mutex _processes_mutex;
  process_group _processes;  ///< Guarded by @c _processes_mutex.
};

}  // namespace gqe::node_manager
