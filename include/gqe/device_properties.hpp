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

#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>

#include <gqe/utility/cuda.hpp>

#include <rmm/cuda_device.hpp>

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
  enum property { multiProcessorCount, pageableMemoryAccess, unifiedAddressing, managedMemory };

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

  std::unordered_map<int, cudaDeviceProp> _device_properties_cache;

  template <property p>
  struct dependent_false : std::false_type {};
};

template <device_properties::property p>
auto device_properties::get(rmm::cuda_device_id device) const
{
  auto match = _device_properties_cache.find(device.value());
  if (match == _device_properties_cache.end()) { throw std::runtime_error("Device not found"); }
  auto& properties = match->second;

  if constexpr (p == property::multiProcessorCount) {
    return static_cast<int>(properties.multiProcessorCount);
  } else if constexpr (p == property::pageableMemoryAccess) {
    return static_cast<bool>(properties.pageableMemoryAccess);
  } else if constexpr (p == property::unifiedAddressing) {
    return static_cast<bool>(properties.unifiedAddressing);
  } else if constexpr (p == property::managedMemory) {
    return static_cast<bool>(properties.managedMemory);
  } else {
    static_assert(dependent_false<p>::value, "The requested device property is not implemented.");
  }
}

}  // namespace gqe
