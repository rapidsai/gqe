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

#include <gqe/utility/cuda.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace gqe {

/**
 * @brief Singleton wrapper class for caching and accessing device properties.
 *
 */
class device_properties {
 public:
  /**
   * @brief Enum specifying accessible and enabled device properties
   *
   */
  enum property {
    multiProcessorCount,
    pageableMemoryAccess,
    unifiedAddressing,
    managedMemory,
    memDecompressSupport
  };

  /**
   * @brief Get the singleton instance of device_properties.  Properties of all visible devices are
   * cached.
   * @warning We assume that set of visible devices does not change during execution.
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

  struct driver_properties {
    int memDecompressSupport;
  };

  std::unordered_map<int, cudaDeviceProp> _device_properties_cache;
  std::unordered_map<int, driver_properties> _driver_properties_cache;

  template <property p>
  struct dependent_false : std::false_type {};
};

template <device_properties::property p>
auto device_properties::get(rmm::cuda_device_id device) const
{
  auto cuda_match = _device_properties_cache.find(device.value());
  if (cuda_match == _device_properties_cache.end()) {
    throw std::runtime_error("Device not found");
  }
  auto& cuda_properties = cuda_match->second;

  auto driver_match = _driver_properties_cache.find(device.value());
  if (driver_match == _driver_properties_cache.end()) {
    throw std::runtime_error("Driver properties not found");
  }
  auto& driver_properties = driver_match->second;

  if constexpr (p == property::multiProcessorCount) {
    return static_cast<int>(cuda_properties.multiProcessorCount);
  } else if constexpr (p == property::pageableMemoryAccess) {
    return static_cast<bool>(cuda_properties.pageableMemoryAccess);
  } else if constexpr (p == property::unifiedAddressing) {
    return static_cast<bool>(cuda_properties.unifiedAddressing);
  } else if constexpr (p == property::managedMemory) {
    return static_cast<bool>(cuda_properties.managedMemory);
  } else if constexpr (p == property::memDecompressSupport) {
    return static_cast<bool>(driver_properties.memDecompressSupport);
  } else {
    static_assert(dependent_false<p>::value, "The requested device property is not implemented.");
  }
}

}  // namespace gqe
