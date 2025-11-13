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

device_properties const& device_properties::instance()
{
  static device_properties instance;
  return instance;
}

device_properties::device_properties()
{
  for (auto const& device : get_all_devices()) {
    cudaDeviceProp deviceProp;
    GQE_CUDA_TRY(cudaGetDeviceProperties(&deviceProp, device.value()));
    _device_properties_cache.insert(std::make_pair(device.value(), deviceProp));
  }
}

}  // namespace gqe