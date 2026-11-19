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

#include "../test_utilities.hpp"

#include <gtest/gtest.h>

#include <barrier>
#include <thread>
#include <vector>

using namespace gqe_test;

TEST(signal_barrier, construct)
{
  signal_barrier sb(1, 1);
  EXPECT_DEBUG_DEATH({ signal_barrier sb(1, 0); }, "");
  EXPECT_DEBUG_DEATH({ signal_barrier sb(0, 1); }, "");
}

void signal_wait_test(size_t numWaiters, size_t numSignalers, size_t iters)
{
  signal_barrier sb(numWaiters, numSignalers);

  std::vector<std::jthread> waiter_threads;
  for (size_t i = 0; i < numWaiters; ++i) {
    waiter_threads.emplace_back([&]() {
      for (size_t iter = 0; iter < iters; ++iter) {
        sb.wait();
      }
    });
  }

  std::vector<std::jthread> signaler_threads;
  for (size_t i = 0; i < numSignalers; ++i) {
    signaler_threads.emplace_back([&]() {
      for (size_t iter = 0; iter < iters; ++iter) {
        sb.signal();
      }
    });
  }
}

TEST(signal_barrier, signal_wait) { signal_wait_test(1, 1, 2); }

TEST(signal_barrier, signal_wait_stress)
{
  size_t const nthreads = std::thread::hardware_concurrency() / 2;
  signal_wait_test(nthreads, nthreads, 1000);
}
