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

#include <gqe/utility/thread_entry.hpp>

#include <gqe/device_properties.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/linux.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <exception>
#include <thread>

/**
 * @brief Test that a new thread ends up on the given device and only on that device's CPUs.
 *
 * The last visible device is used so that the test also detects a thread left on the default
 * device 0 when more than one device is visible.
 */
TEST(ThreadEntryTest, SetsDeviceAndPinsToDeviceCpus)
{
  auto const device = rmm::cuda_device_id{gqe::utility::get_device_count() - 1};
  auto const device_cpus =
    gqe::device_properties::instance().get<gqe::device_properties::property::cpuAffinity>(device);

  // The kernel restricts the requested mask to the CPUs that the cgroup cpuset allows, not to the
  // mask that the thread inherits. The allowed CPUs are not known here, but they include every CPU
  // in the inherited mask, so the CPUs local to the device and in that mask are a lower bound.
  // This also holds when a launcher such as `taskset` has narrowed the inherited mask.
  gqe::cpu_set inherited_cpus;
  gqe::utility::get_thread_affinity(inherited_cpus);
  gqe::cpu_set required_cpus;
  for (int cpu = 0; cpu < gqe::cpu_set::max_count; ++cpu) {
    if (device_cpus.contains(cpu) && inherited_cpus.contains(cpu)) { required_cpus.add(cpu); }
  }
  if (required_cpus.count() == 0) {
    GTEST_SKIP() << "No CPU local to device " << device.value() << " is in the inherited mask";
  }

  int current_device = -1;
  gqe::cpu_set affinity;
  std::exception_ptr error;
  std::thread worker([&] {
    try {
      gqe::utility::enter_device_thread(device);
      GQE_CUDA_TRY(cudaGetDevice(&current_device));
      gqe::utility::get_thread_affinity(affinity);
    } catch (...) {
      error = std::current_exception();
    }
  });
  worker.join();
  if (error) { std::rethrow_exception(error); }

  EXPECT_EQ(current_device, device.value());
  for (int cpu = 0; cpu < gqe::cpu_set::max_count; ++cpu) {
    if (affinity.contains(cpu)) {
      EXPECT_TRUE(device_cpus.contains(cpu)) << "CPU " << cpu << " is not local to the device";
    }
    if (required_cpus.contains(cpu)) {
      EXPECT_TRUE(affinity.contains(cpu)) << "CPU " << cpu << " is missing from the affinity";
    }
  }
}

/**
 * @brief Test that an invalid device is reported instead of leaving the thread on another device.
 */
TEST(ThreadEntryTest, ThrowsOnInvalidDevice)
{
  auto const invalid_device = rmm::cuda_device_id{gqe::utility::get_device_count()};
  EXPECT_THROW(gqe::utility::enter_device_thread(invalid_device), gqe::cuda_error);
}
