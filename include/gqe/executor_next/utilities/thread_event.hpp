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
#pragma once

#include <condition_variable>
#include <mutex>

namespace gqe::executor_next {

/**
 * @brief
 *  A thread_event is a std::condition_variable that includes a notification state to
 *  prevent "lost wakeups".
 */
class thread_event {
 public:
  /**
   * @brief The state of the event.
   */
  enum class event_state : uint32_t {
    blocked,   //< The thread is blocked on the std::condition_variable.
    notified,  //< The thread is marked as notified.
    unblocked  //< The thread not blocked or notified.
  };

  /**
   * @brief Blocks the calling thread until notified only if it not already marked as notified.
   * @return True if the thead was not marked as notified, otherwise false.
   */
  bool wait() noexcept;

  /**
   * @brief Mark this thread as notified and notifies the internal condition_variable.
   * @return True if the thead was blocked, otherwise false.
   */
  bool notify() noexcept;

  /**
   * @return The current event state.
   */
  [[nodiscard]] event_state state() const noexcept;

 private:
  mutable std::mutex _mutex;           //< The mutex used to block the thread.
  std::condition_variable _block_var;  //< The condition variable used to block notify the thread.
  event_state _state = event_state::unblocked;  //< The current state.
};

}  // namespace gqe::executor_next
