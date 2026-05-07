/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>

#include <rmm/cuda_device.hpp>

#include <type_traits>
#include <unordered_map>

namespace gqe {

/**
 * @brief Singleton wrapper class for caching and accessing device properties.
 *
 * Caches properties of all visible devices.
 *
 * @invariant The set of visible devices does not change during execution.
 *
 * # Design Rationale
 *
 * Calling `cudaGetDeviceProperties` and `cuDeviceGetAttribute` is expensive (several milliseconds).
 * Therefore, we cache properties of all visible devices to avoid repeated calls to these functions.
 *
 * Properties have different data types. Therefore, `get()` takes care to return the correct data
 * type for each property.
 *
 * `get()` returns a single property to hide the implementation details of the properties cache.
 *
 * As the header is used in many files, we avoid including C headers in this C++ header.
 */
class device_properties {
 public:
  /**
   * @brief Enum specifying accessible and enabled device properties
   *
   */
  enum property {
    cpuAffinity,
    hostNumaId,
    managedMemory,
    memDecompressSupport,
    memoryAffinity,
    memoryPoolsSupported,
    memoryPoolSupportedHandleTypes,
    multiProcessorCount,
    pageableMemoryAccess,
    unifiedAddressing
  };

  /**
   * @brief Get the singleton instance of device_properties.
   *
   * @return device_properties& Reference to the singleton instance
   */
  static device_properties const& instance();

  // Delete copy constructor and assignment operator
  device_properties(const device_properties&)            = delete;
  device_properties& operator=(const device_properties&) = delete;
  device_properties(device_properties&&)                 = delete;
  device_properties& operator=(device_properties&&)      = delete;

  /**
   * @brief Get some property p for device.
   *
   * @tparam p Property to get
   * @param device Device for which p is queried.
   * @return auto
   */
  template <property p>
  auto get(rmm::cuda_device_id device = utility::current_cuda_device_id()) const;

 private:
  /**
   * @brief Private constructor for singleton pattern
   */
  explicit device_properties();

  /**
   * @brief CUDA driver-derived properties cache for a device.
   */
  struct driver_properties {
    int host_numa_id;
    bool mem_decompress_support;
  };

  /**
   * @brief NVML-derived properties cache for a device.
   */
  struct nvml_properties {
    cpu_set cpu_affinity;    /**< CPU set for GPU CPU affinity */
    cpu_set memory_affinity; /**< NUMA node set for GPU memory affinity */
  };

  /**
   * @brief CUDA runtime-derived properties cache for a device.
   *
   * @note The properties are cached as bools and ints to avoid unnecessary conversions.
   * @note Don't use cudaDeviceProp directly to avoid including C headers in a C++ header.
   */
  struct runtime_properties {
    bool managed_memory;
    bool memory_pools_supported;
    int memory_pool_supported_handle_types;
    int multi_processor_count;
    bool pageable_memory_access;
    bool unified_addressing;
  };

  std::unordered_map<int, driver_properties> _driver_properties_cache;
  std::unordered_map<int, nvml_properties> _nvml_properties_cache;
  std::unordered_map<int, runtime_properties> _runtime_properties_cache;

  [[nodiscard]] driver_properties const& get_driver_properties(rmm::cuda_device_id device) const;
  [[nodiscard]] nvml_properties const& get_nvml_properties(rmm::cuda_device_id device) const;
  [[nodiscard]] runtime_properties const& get_runtime_properties(rmm::cuda_device_id device) const;

  template <property p>
  struct dependent_false : std::false_type {};
};

template <device_properties::property p>
auto device_properties::get(rmm::cuda_device_id device) const
{
  if constexpr (p == property::cpuAffinity) {
    return get_nvml_properties(device).cpu_affinity;
  } else if constexpr (p == property::hostNumaId) {
    return get_driver_properties(device).host_numa_id;
  } else if constexpr (p == property::managedMemory) {
    return get_runtime_properties(device).managed_memory;
  } else if constexpr (p == property::memDecompressSupport) {
    return get_driver_properties(device).mem_decompress_support;
  } else if constexpr (p == property::memoryAffinity) {
    return get_nvml_properties(device).memory_affinity;
  } else if constexpr (p == property::memoryPoolsSupported) {
    return get_runtime_properties(device).memory_pools_supported;
  } else if constexpr (p == property::memoryPoolSupportedHandleTypes) {
    return get_runtime_properties(device).memory_pool_supported_handle_types;
  } else if constexpr (p == property::multiProcessorCount) {
    return get_runtime_properties(device).multi_processor_count;
  } else if constexpr (p == property::pageableMemoryAccess) {
    return get_runtime_properties(device).pageable_memory_access;
  } else if constexpr (p == property::unifiedAddressing) {
    return get_runtime_properties(device).unified_addressing;
  } else {
    static_assert(dependent_false<p>::value, "The requested device property is not implemented.");
  }
}

}  // namespace gqe
