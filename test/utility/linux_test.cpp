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

#include <numa.h>
#include <numaif.h>
#include <sched.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <system_error>
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
 * @brief Test that a thread pinned after the startup capture can widen to the captured mask.
 *
 * The captured mask is cached once per process, and earlier tests in this binary already fill the
 * cache. The scenario therefore runs in a fresh process: the "threadsafe" death-test style
 * re-executes this test binary for the statement, so the child starts with an empty cache. Nothing
 * before `EXPECT_EXIT` may touch the cache, because the child re-runs the test body up to there.
 *
 * Without the capture, the first read would happen after pinning and cache the one-CPU mask, so
 * the child would exit with code 1.
 */
TEST(LinuxUtilityTest, CaptureProcessAffinityBeforePinning)
{
  auto const cpu = first_allowed_cpu();
  ASSERT_GE(cpu, 0);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
    {
      cpu_set fullmask;
      utility::get_thread_affinity(fullmask);
      utility::capture_process_affinity();

      utility::set_thread_affinity(cpu_set(cpu));
      utility::set_thread_affinity_fullmask();

      cpu_set new_affinity;
      utility::get_thread_affinity(new_affinity);
      std::_Exit(new_affinity == fullmask ? 0 : 1);
    },
    ::testing::ExitedWithCode(0),
    "");
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

TEST(LinuxUtilityTest, RelativeMembindIsUnsupported)
{
  if (numa_available() < 0) { GTEST_SKIP() << "NUMA policy is unavailable"; }
  // The worker's memory policy is discarded when the thread exits.
  std::jthread worker([] {
    int mode                = MPOL_DEFAULT;
    auto const query_result = ::get_mempolicy(&mode, nullptr, 0, nullptr, 0);
    if (query_result != 0 && (errno == EPERM || errno == ENOSYS)) {
      GTEST_SKIP() << "Reading NUMA memory policy is denied or unavailable";
    }
    ASSERT_EQ(query_result, 0) << std::strerror(errno);

    std::unique_ptr<bitmask, decltype(&numa_bitmask_free)> binding{numa_allocate_nodemask(),
                                                                   numa_bitmask_free};
    ASSERT_NE(binding, nullptr);
    numa_bitmask_setbit(binding.get(), 0);
    auto const result =
      ::set_mempolicy(MPOL_BIND | MPOL_F_RELATIVE_NODES, binding->maskp, binding->size);
    if (result != 0 && (errno == EPERM || errno == ENOSYS)) {
      GTEST_SKIP() << "Setting NUMA memory policy is denied or unavailable";
    }
    ASSERT_EQ(result, 0) << std::strerror(errno);

    try {
      static_cast<void>(gqe::utility::cpu_bearing_numa_nodes());
      FAIL() << "Relative membind must be rejected";
    } catch (std::system_error const& error) {
      FAIL() << "NUMA policy discovery failed: " << error.what();
    } catch (std::runtime_error const& error) {
      EXPECT_STREQ(error.what(),
                   "Cannot discover usable NUMA nodes under a relative membind policy");
    } catch (...) {
      FAIL() << "Unexpected exception from NUMA policy discovery";
    }
  });
}
