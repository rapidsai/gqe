/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
 * @brief Wrapper class for caching and accessing device properties. Intended to be accessed through
 * the task manager context.
 *
 */
struct device_properties {
  /**
   * @brief Enum specifying accessible and enabled device properties
   *
   */
  enum property { multiProcessorCount, pageableMemoryAccess, unifiedAddressing, managedMemory };

  /**
   * @brief Construct a new device properties object given a set of visible devices. By default,
   * properties of all visible devices are cached.
   * @warning We assume that set of visible_devices does not change during execution.
   *
   * @param visible_devices
   */
  explicit device_properties(const std::vector<rmm::cuda_device_id>&
                               visible_devices);  // query and cache all visible device properties
  device_properties();                            // query and cache all visible device properties
  device_properties(const device_properties&)            = default;
  device_properties(device_properties&&)                 = default;
  device_properties& operator=(const device_properties&) = default;
  device_properties& operator=(device_properties&&)      = default;

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
