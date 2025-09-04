/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gqe {

namespace utility {

namespace detail {
int detect_launch_grid_size(gqe::device_properties const& device_properties,
                            void const* const kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes)
{
  auto device_id = current_cuda_device_id();
  auto num_sms   = device_properties.get<device_properties::multiProcessorCount>(device_id);

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
