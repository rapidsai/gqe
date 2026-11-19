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

#include <utility>

namespace gqe::node_manager {

task_manager_group::task_manager_group(std::uint64_t generation, spawn_result spawned)
  : _generation(generation),
    _channels(std::move(spawned.channels)),
    _processes(std::move(spawned.processes))
{
}

task_manager_group::~task_manager_group() { terminate(); }

void task_manager_group::mark_ready() noexcept
{
  auto expected = state::starting;
  _state.compare_exchange_strong(
    expected, state::ready, std::memory_order_acq_rel, std::memory_order_acquire);
}

void task_manager_group::mark_failed() noexcept
{
  _state.store(state::failed, std::memory_order_release);
}

std::optional<exit_status> task_manager_group::try_wait_any()
{
  std::scoped_lock guard(_processes_mutex);
  return _processes.try_wait_any();
}

void task_manager_group::terminate(std::chrono::seconds timeout)
{
  mark_failed();
  std::scoped_lock guard(_processes_mutex);
  _processes.terminate_all(timeout);
}

}  // namespace gqe::node_manager
