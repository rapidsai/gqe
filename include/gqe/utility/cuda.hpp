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

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstddef>

namespace gqe::utility {

namespace detail {
/**
 * @brief Implementation of `detect_launch_grid_size()`.
 *
 * The function is wrapped by a template function that takes a non-void
 * `kernel` argument type.
 */
int detect_launch_grid_size(void const* const kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes);
}  // namespace detail

/**
 * @brief Return the current CUDA device ID.
 */
rmm::cuda_device_id current_cuda_device_id();

/**
 * @brief Detect a "reasonable" grid size for the kernel.
 *
 * Calculates a "reasonable" grid size based on the theoretical occupancy using
 * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
 *
 * Reference:
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
 *
 * @param[in] kernel The kernel function.
 * @param[in] block_size The block size used for the kernel launch.
 * @param[in] dynamic_shared_memory_bytes The dynamic shared memory size in bytes used for the
 * kernel launch.
 *
 * @return The grid size.
 */
template <typename KernelType>
int detect_launch_grid_size(const KernelType kernel,
                            const int block_size,
                            const size_t dynamic_shared_memory_bytes = 0)
{
  // Cast the kernel function pointer to a void data pointer. g++ emits warning #167-D when passing
  // kernel directly to a `void*` argument.
  return detail::detect_launch_grid_size(
    reinterpret_cast<void const*>(kernel), block_size, dynamic_shared_memory_bytes);
}

/**
 * @brief Perform a batched memcpy operation.
 *
 * @param[out] dst_ptrs The destination pointers.
 * @param[in] src_ptrs The source pointers.
 * @param[out] sizes The sizes of each copied buffer.
 * @param[in] num_buffers The number of buffers to copy.
 * @param[in] stream The CUDA stream on which to execute the operation.
 */
void do_batched_memcpy(
  void** dst_ptrs, void** src_ptrs, size_t* sizes, size_t num_buffers, cudaStream_t stream);

/**
 * @brief NVTX domain which should be used across the GQE project
 */
struct gqe_nvtx_domain {
  static constexpr char const* name{"GQE"};
};

/**
 * @brief A RAII object for creating a NVTX range local to a thread within the GQE domain
 */
using nvtx_scoped_range = nvtx3::scoped_range_in<gqe_nvtx_domain>;

/**
 * @brief Create a NVTX marker within the GQE domain.
 */
template <typename... Args>
inline void nvtx_mark(Args&&... args)
{
  nvtx3::mark_in<gqe_nvtx_domain>(std::forward<Args>(args)...);
}

}  // namespace gqe::utility
