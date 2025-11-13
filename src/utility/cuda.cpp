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

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gqe {

namespace utility {

namespace detail {
int detect_launch_grid_size(void const* const kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes)
{
  auto device_id = current_cuda_device_id();
  auto num_sms =
    device_properties::instance().get<device_properties::multiProcessorCount>(device_id);

  int max_active_blocks = 0;
  GQE_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, kernel, block_size, dynamic_shared_memory_bytes));

  return max_active_blocks * num_sms;
}
}  // namespace detail

rmm::cuda_device_id current_cuda_device_id()
{
  int id{};
  GQE_CUDA_TRY(cudaGetDevice(&id));
  return rmm::cuda_device_id{id};
}

}  // namespace utility

}  // namespace gqe
