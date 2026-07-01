/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/linux.hpp>

#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <sched.h>

#include <atomic>
#include <thread>

namespace gqe {
namespace {

/**
 * @brief Returns the first CPU available to the current process affinity mask.
 */
int first_allowed_cpu()
{
  cpu_set_t allowed;
  CPU_ZERO(&allowed);

  EXPECT_EQ(sched_getaffinity(0, sizeof(allowed), &allowed), 0);

  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) { return cpu; }
  }

  return -1;
}

/**
 * @brief Test that set_thread_affinity works.
 */
TEST(LinuxUtilityTest, SetThreadAffinity)
{
  // Choose one CPU from the allowed cpuset
  auto const cpu = first_allowed_cpu();
  ASSERT_GE(cpu, 0);
  cpu_set affinity(cpu);

  std::atomic<bool> success{false};
  std::thread worker([&]() {
    try {
      utility::set_thread_affinity(affinity);
      success = true;
    } catch (...) {
      success = false;
    }
  });
  worker.join();

  EXPECT_TRUE(success.load());
}

/**
 * @brief Test that get_thread_affinity works.
 */
TEST(LinuxUtilityTest, GetThreadAffinity)
{
  // Choose one CPU from the allowed cpuset
  auto const cpu = first_allowed_cpu();
  ASSERT_GE(cpu, 0);
  cpu_set affinity(cpu);

  std::atomic<bool> success{false};
  std::thread worker([&]() {
    try {
      cpu_set new_affinity;
      utility::set_thread_affinity(affinity);
      utility::get_thread_affinity(new_affinity);
      success = (affinity == new_affinity);
    } catch (...) {
      success = false;
    }
  });
  worker.join();

  EXPECT_TRUE(success.load());
}

/**
 * @brief Test that set_thread_affinity_fullmask works.
 */
TEST(LinuxUtilityTest, SetThreadAffinityFullmask)
{
  cpu_set fullmask;
  sched_getaffinity(0, cpu_set::byte_count, reinterpret_cast<cpu_set_t*>(fullmask.bits()));

  std::atomic<bool> success{false};
  std::thread worker([&]() {
    try {
      cpu_set new_affinity;
      utility::set_thread_affinity_fullmask();
      utility::get_thread_affinity(new_affinity);
      success = (new_affinity == fullmask);
    } catch (...) {
      success = false;
    }
  });
  worker.join();

  EXPECT_TRUE(success.load());
}

/**
 * @brief Test that cpu_set.zero() works.
 */
TEST(LinuxUtilityTest, CpuSetZero)
{
  cpu_set_t zero;
  CPU_ZERO(&zero);

  cpu_set mask;
  mask.zero();

  EXPECT_TRUE(CPU_EQUAL(&zero, reinterpret_cast<cpu_set_t*>(mask.bits())));
}

/**
 * @brief Test that scoped_cpu_affinity works as intended.
 */
TEST(LinuxUtilityTest, ScopedCpuAffinity)
{
  auto const cpu = first_allowed_cpu();
  ASSERT_GE(cpu, 0);
  cpu_set one_core(cpu);
  cpu_set fullmask;
  sched_getaffinity(0, cpu_set::byte_count, reinterpret_cast<cpu_set_t*>(fullmask.bits()));

  std::atomic<bool> success{false};
  std::thread worker([&]() {
    try {
      utility::set_thread_affinity(one_core);

      // see if mask matches what we set
      cpu_set local_affinity;
      utility::get_thread_affinity(local_affinity);
      success = (local_affinity == one_core);
      {
        scoped_cpu_affinity sca;
        // see if mask is the full mask
        utility::get_thread_affinity(local_affinity);
        success = success && (local_affinity == fullmask);
      }
      // see if mask is successfully restored when we exit scope
      utility::get_thread_affinity(local_affinity);
      success = success && (local_affinity == one_core);
    } catch (...) {
      success = false;
    }
  });
  worker.join();

  EXPECT_TRUE(success.load());
}

}  // namespace
}  // namespace gqe
