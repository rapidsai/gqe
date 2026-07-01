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

#include <atomic>
#include <cassert>
#include <cstdint>

namespace gqe_test {

/**
 * @brief Spins while the predicate is true.
 */
template <typename FuncT>
void spin_while(FuncT predicate)
{
  while (predicate())
    ;
}

/**
 * @brief A spin barrier that blocks until all waiters are waiting and all signals
 *  have been received. It then releases the waiters and resets.
 */
class signal_barrier {
 public:
  /**
   * @brief Construct a new signal barrier object.
   * @param waitCount The number of waiters.
   * @param signalCount the number of signalers.
   */
  signal_barrier(uint32_t waitCount, uint32_t signalCount)
    : _wait_count(waitCount), _signal_count(signalCount)
  {
    assert(waitCount > 0 && signalCount > 0);
  }

  /**
   * @brief Adds to the signal count. After reset, this call blocks until
   *    all waiters have exited.
   */
  inline void signal()
  {
    // If fully signaled, wait for waiters to exit, then increment.
    while (true) {
      uint32_t expected = _num_signals.load(std::memory_order_acquire);
      if (expected < _signal_count &&
          _num_signals.compare_exchange_weak(
            expected, expected + 1, std::memory_order_release, std::memory_order_relaxed)) {
        break;
      }
    }
  }

  /**
   * @brief Blocks until all waiters are waiting and all signals
   *  have been received.
   */
  inline void wait()
  {
    uint32_t waiter_epoch = _epoch.load(std::memory_order_acquire);
    uint32_t waiter_id    = _num_waiters.fetch_add(1, std::memory_order_relaxed);

    if (waiter_id == _wait_count - 1)  // last waiter
    {
      //  Spin until all the signalers have signaled
      while (_num_signals.load(std::memory_order_acquire) < _signal_count)
        ;

      // reset the barrier
      _num_signals.exchange(0, std::memory_order_relaxed);
      _num_waiters.exchange(0, std::memory_order_release);

      // advance the epoch
      _epoch.fetch_add(1, std::memory_order_release);
    } else {
      while (waiter_epoch == _epoch.load(std::memory_order_acquire))
        ;
    }
  }

 private:
  uint32_t const _wait_count         = 0;
  uint32_t const _signal_count       = 0;
  std::atomic<uint32_t> _num_waiters = 0;
  std::atomic<uint32_t> _num_signals = 0;
  std::atomic<uint32_t> _epoch       = 0;
};

}  // namespace gqe_test
