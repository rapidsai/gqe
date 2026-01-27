/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/helpers.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <semaphore>
#include <thread>
#include <type_traits>

namespace gqe {
namespace test {

namespace {
static constexpr auto TIMEOUT_DURATION = std::chrono::milliseconds(1);
}

/**
 * @brief Test that a second thread cannot acquire a counting_semaphore after the first thread
 * acquires and exits without releasing.
 *
 * The purpose of this test is to showcase that `counting_semaphore` needs an acquire guard, similar
 * to `std::lock_guard`.
 */
TEST(HelpersTest, CountingSemaphoreAcquireAfterThreadExit)
{
  std::binary_semaphore semaphore(1);

  // First thread acquires the semaphore and exits without releasing
  std::thread first_thread([&semaphore]() { semaphore.acquire(); });
  first_thread.join();

  // Second thread attempts to acquire the semaphore
  std::atomic_bool acquired_semaphore = false;
  std::thread second_thread([&semaphore, &acquired_semaphore]() {
    // Use try_acquire_for to avoid blocking indefinitely if the test fails
    acquired_semaphore = semaphore.try_acquire_for(TIMEOUT_DURATION);
  });
  second_thread.join();

  EXPECT_FALSE(acquired_semaphore);
}

/**
 * @brief Test that `semaphore_acquire_guard` properly releases the semaphore when the guard goes
 * out of scope.
 */
TEST(HelpersTest, SemaphoreAcquireGuardReleasesOnDestruction)
{
  std::binary_semaphore semaphore(1);
  {
    utility::semaphore_acquire_guard guard(semaphore);
    // The guard is released here.
  }
  EXPECT_TRUE(semaphore.try_acquire_for(TIMEOUT_DURATION));
}

/**
 * @brief Test that semaphore_acquire_guard properly releases the semaphore when the thread exits
 * unexpectedly.
 *
 * This test demonstrates that using semaphore_acquire_guard ensures the semaphore is released
 * even when the thread exits, allowing subsequent threads to acquire it.
 */
TEST(HelpersTest, SemaphoreAcquireGuardReleasesOnException)
{
  std::binary_semaphore semaphore(1);

  // First thread acquires the semaphore using the guard and throws an exception.
  std::thread first_thread([&semaphore]() {
    try {
      utility::semaphore_acquire_guard guard(semaphore);
      throw std::runtime_error("Test exception");
      // The guard should be released here because the exception is thrown.
    } catch (const std::exception&) {
      return;
    }
  });
  first_thread.join();

  // Second thread attempts to acquire the semaphore
  std::atomic_bool acquired_semaphore = false;
  std::thread second_thread([&semaphore, &acquired_semaphore]() {
    acquired_semaphore = semaphore.try_acquire_for(TIMEOUT_DURATION);
  });
  second_thread.join();

  EXPECT_TRUE(acquired_semaphore);
}

/**
 * @brief Test move constructor
 */
TEST(HelpersTest, SemaphoreAcquireGuardMoveConstructor)
{
  std::binary_semaphore semaphore(1);
  utility::semaphore_acquire_guard guard(semaphore);
  utility::semaphore_acquire_guard other_guard(std::move(guard));
  EXPECT_FALSE(guard.is_valid());
  EXPECT_TRUE(other_guard.is_valid());
}

/**
 * @brief Test move assignment operator
 */
TEST(HelpersTest, SemaphoreAcquireGuardMoveAssignmentOperator)
{
  std::binary_semaphore semaphore(1);
  utility::semaphore_acquire_guard guard(semaphore);
  utility::semaphore_acquire_guard other_guard = std::move(guard);
  EXPECT_FALSE(guard.is_valid());
  EXPECT_TRUE(other_guard.is_valid());
}

/**
 * @brief Test that copy constructor and copy assignment are deleted
 */
TEST(HelpersTest, SemaphoreAcquireGuardCopyConstructorAndAssignmentAreDeleted)
{
  static_assert(!std::is_copy_constructible_v<utility::semaphore_acquire_guard<>>,
                "semaphore_acquire_guard should not be copy constructible");
  static_assert(!std::is_copy_assignable_v<utility::semaphore_acquire_guard<>>,
                "semaphore_acquire_guard should not be copy assignable");
}

/**
 * @brief Test that `semaphore_acquire_guard` is movable between threads.
 */
TEST(HelpersTest, SemaphoreAcquireGuardIsMovable)
{
  std::binary_semaphore semaphore(1);
  utility::semaphore_acquire_guard guard(semaphore);

  std::thread thread([&guard]() {
    utility::semaphore_acquire_guard local_guard(std::move(guard));
    // The guard is released here.
  });
  thread.join();

  EXPECT_TRUE(semaphore.try_acquire_for(TIMEOUT_DURATION));
}

}  // namespace test
}  // namespace gqe
