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

#include "../test_utilities.hpp"

#include <gtest/gtest.h>

#include <thread>

using namespace gqe::executor_next;
using namespace gqe_test;

TEST(thread_event, unblocked_notify)
{
  thread_event event;
  ASSERT_EQ(event.state(), thread_event::event_state::unblocked);

  // event is not blocked
  ASSERT_FALSE(event.notify());
  ASSERT_EQ(event.state(), thread_event::event_state::notified);
}

TEST(thread_event, wait_and_notify)
{
  thread_event event;
  signal_barrier sb(1, 1);

  // Do twice to verify reuse.
  int constexpr iters = 2;

  std::thread t([&]() {
    for (int i = 0; i < iters; ++i) {
      ASSERT_TRUE(event.wait());
      sb.wait();
    }
  });

  for (int i = 0; i < iters; ++i) {
    spin_while([&]() { return event.state() != thread_event::event_state::blocked; });
    ASSERT_TRUE(event.notify());
    spin_while([&]() { return event.state() != thread_event::event_state::unblocked; });
    sb.signal();
  }

  t.join();
}

TEST(thread_event, no_lost_wakeups)
{
  // TODO (breta): use relacy or tsan
  for (size_t iters = 0; iters < 100; ++iters) {
    thread_event event;
    std::thread t([&]() { event.wait(); });
    event.notify();

    // Join will never hang.
    t.join();
  }
}
