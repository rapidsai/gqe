/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/device_properties.hpp>

#include <gtest/gtest.h>

#include <rmm/cuda_device.hpp>

namespace gqe {
namespace {

class DevicePropertiesTest : public ::testing::Test {
 protected:
  rmm::cuda_set_device_raii device_guard_{rmm::cuda_device_id{0}};
};

/**
 * @brief Test that cpuAffinity returns a non-empty cpu_set.
 */
TEST_F(DevicePropertiesTest, CpuAffinityReturnsNonEmptySet)
{
  auto const& cpu_affinity =
    device_properties::instance().get<device_properties::property::cpuAffinity>();

  // The CPU affinity should contain at least one CPU
  EXPECT_GT(cpu_affinity.count(), 0);
}

/**
 * @brief Test that memoryAffinity returns a non-empty cpu_set.
 */
TEST_F(DevicePropertiesTest, MemoryAffinityReturnsNonEmptySet)
{
  auto const& memory_affinity =
    device_properties::instance().get<device_properties::property::memoryAffinity>();

  // The memory affinity should contain at least one NUMA node
  EXPECT_GT(memory_affinity.count(), 0);
}

/**
 * @brief Test that managedMemory can be queried.
 */
TEST_F(DevicePropertiesTest, ManagedMemory)
{
  EXPECT_NO_THROW(device_properties::instance().get<device_properties::property::managedMemory>());
}

/**
 * @brief Test that memDecompressSupport can be queried.
 */
TEST_F(DevicePropertiesTest, MemDecompressSupport)
{
  EXPECT_NO_THROW(
    device_properties::instance().get<device_properties::property::memDecompressSupport>());
}

/**
 * @brief Test that multiProcessorCount returns a positive value.
 */
TEST_F(DevicePropertiesTest, MultiProcessorCount)
{
  auto mp_count =
    device_properties::instance().get<device_properties::property::multiProcessorCount>();

  EXPECT_GT(mp_count, 0);
}

/**
 * @brief Test that pageableMemoryAccess can be queried.
 */
TEST_F(DevicePropertiesTest, PageableMemoryAccess)
{
  EXPECT_NO_THROW(
    device_properties::instance().get<device_properties::property::pageableMemoryAccess>());
}

/**
 * @brief Test that unifiedAddressing can be queried.
 */
TEST_F(DevicePropertiesTest, UnifiedAddressing)
{
  EXPECT_NO_THROW(
    device_properties::instance().get<device_properties::property::unifiedAddressing>());
}

}  // namespace
}  // namespace gqe
