/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/device_properties.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <numeric>
#include <rmm/cuda_device.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
std::vector<rmm::cuda_device_id> get_all_devices()
{
  int deviceCount;
  GQE_CUDA_TRY(cudaGetDeviceCount(&deviceCount));
  std::vector<rmm::cuda_device_id> devices;
  devices.reserve(deviceCount);
  for (auto i = 0; i < deviceCount; i++) {
    devices.emplace_back(rmm::cuda_device_id{i});
  }
  return devices;
}
}  // namespace

namespace gqe {

device_properties::device_properties(const std::vector<rmm::cuda_device_id>& visible_devices)
{
  for (auto const& device : visible_devices) {
    cudaDeviceProp deviceProp;
    GQE_CUDA_TRY(cudaGetDeviceProperties(&deviceProp, device.value()));
    _device_properties_cache.insert(std::make_pair(device.value(), deviceProp));
  }
}

device_properties::device_properties() : device_properties(get_all_devices()) {}

}  // namespace gqe