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
#include "gqe/executor_next/utilities/thread_event.hpp"

namespace gqe::executor_next {

bool thread_event::wait() noexcept
{
  std::unique_lock<std::mutex> lock(_mutex);

  if (_state == event_state::notified) { return false; }

  _state = event_state::blocked;

  // Blocked.......................
  _block_var.wait(lock, [&]() { return _state == event_state::notified; });

  _state = event_state::unblocked;

  return true;
}

bool thread_event::notify() noexcept
{
  std::unique_lock<std::mutex> lock(_mutex);

  bool wasBlocked = _state == event_state::blocked;
  _state          = event_state::notified;

  // Unblock the waiting thread.
  _block_var.notify_one();

  return wasBlocked;
}

thread_event::event_state thread_event::state() const noexcept
{
  std::unique_lock<std::mutex> lock(_mutex);
  return _state;
}

}  // namespace gqe::executor_next
