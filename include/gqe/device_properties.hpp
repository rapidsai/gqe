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

#include <rmm/cuda_device.hpp>

#include <string>
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
    computeCapabilityMajor,
    cpuAffinity,
    hostNumaId,
    managedMemory,
    memDecompressAlgorithmMask,
    memDecompressSupport,
    memoryAffinity,
    memoryPoolsSupported,
    memoryPoolSupportedHandleTypes,
    multiProcessorCount,
    name,
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
  auto get(rmm::cuda_device_id device = rmm::get_current_cuda_device()) const;

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
    /// The driver's per-format mask of what the hardware decompression engine decodes, as
    /// `CU_MEM_DECOMPRESS_ALGORITHM_*` bits. Zero on a device without an engine.
    unsigned mem_decompress_algorithm_mask;
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
    /// Device name as reported by `cudaDeviceProp::name`, e.g. "NVIDIA GB200".
    std::string name;
    int compute_capability_major;
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

/**
 * @brief Return property @p p of @p device from the cache.
 *
 * @tparam p Property to get
 * @param device Device for which p is queried.
 * @return The property value, with the type that belongs to @p p.
 */
template <device_properties::property p>
auto device_properties::get(rmm::cuda_device_id device) const
{
  if constexpr (p == property::computeCapabilityMajor) {
    return get_runtime_properties(device).compute_capability_major;
  } else if constexpr (p == property::cpuAffinity) {
    return get_nvml_properties(device).cpu_affinity;
  } else if constexpr (p == property::hostNumaId) {
    return get_driver_properties(device).host_numa_id;
  } else if constexpr (p == property::managedMemory) {
    return get_runtime_properties(device).managed_memory;
  } else if constexpr (p == property::memDecompressAlgorithmMask) {
    return get_driver_properties(device).mem_decompress_algorithm_mask;
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
  } else if constexpr (p == property::name) {
    return get_runtime_properties(device).name;
  } else if constexpr (p == property::pageableMemoryAccess) {
    return get_runtime_properties(device).pageable_memory_access;
  } else if constexpr (p == property::unifiedAddressing) {
    return get_runtime_properties(device).unified_addressing;
  } else {
    static_assert(dependent_false<p>::value, "The requested device property is not implemented.");
  }
}

/**
 * @brief Whether @p device's hardware decompression engine decodes @p format.
 *
 * @details Special cases:
 * - Gzip shares the Deflate bit, since a gzip stream is a Deflate payload with a header.
 * - A format the driver mask has no bit for, such as every SM-only format, is never decoded by the
 *   engine.
 * - The result is false on a device without an engine.
 *
 * @param device The device whose engine to ask.
 * @param format The compression format to look up.
 * @return True when the device's engine decodes @p format.
 */
[[nodiscard]] bool decompression_engine_decodes(rmm::cuda_device_id device,
                                                compression_format format);

}  // namespace gqe
