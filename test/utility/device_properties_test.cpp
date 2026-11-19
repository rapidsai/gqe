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

#include <gqe/device_properties.hpp>

#include <gqe/types.hpp>

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
 * @brief Test that the device name matches what the CUDA runtime reports.
 */
TEST_F(DevicePropertiesTest, NameMatchesRuntime)
{
  auto const device = rmm::get_current_cuda_device();
  cudaDeviceProp cuda_properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&cuda_properties, device.value()), cudaSuccess);
  auto const name = device_properties::instance().get<device_properties::property::name>(device);
  EXPECT_FALSE(name.empty());
  EXPECT_EQ(name, std::string(cuda_properties.name));
}

/**
 * @brief Test that the engine support flag is derived from the algorithm mask.
 */
TEST_F(DevicePropertiesTest, MemDecompressSupportFollowsAlgorithmMask)
{
  auto const device = rmm::get_current_cuda_device();
  auto const mask =
    device_properties::instance().get<device_properties::property::memDecompressAlgorithmMask>(
      device);
  EXPECT_EQ(
    device_properties::instance().get<device_properties::property::memDecompressSupport>(device),
    mask != 0);
}

/**
 * @brief Formats without an engine mask bit are never routed to the engine, gzip rides the Deflate
 * bit, and nothing is engine-decoded on a device that reports no engine.
 */
TEST_F(DevicePropertiesTest, DecompressionEngineDecodesFollowsTheDriverMask)
{
  auto const device = rmm::get_current_cuda_device();
  for (auto const format : {compression_format::none,
                            compression_format::ans,
                            compression_format::bitpacking,
                            compression_format::cascaded,
                            compression_format::zstd,
                            compression_format::bitcomp}) {
    EXPECT_FALSE(decompression_engine_decodes(device, format)) << to_string(format);
  }
  EXPECT_EQ(decompression_engine_decodes(device, compression_format::gzip),
            decompression_engine_decodes(device, compression_format::deflate));
  bool const has_engine =
    device_properties::instance().get<device_properties::property::memDecompressSupport>(device);
  for (auto const format :
       {compression_format::lz4, compression_format::snappy, compression_format::deflate}) {
    if (decompression_engine_decodes(device, format)) {
      EXPECT_TRUE(has_engine) << to_string(format);
    }
  }
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
