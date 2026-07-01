/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstddef>
#include <vector>

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
 * @brief Return the number of CUDA devices available on the host.
 */
int get_device_count();

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
 * @brief Collects device-to-device copy requests and executes them as one batched memcpy.
 *
 * Callers add one request per selected source range. The batch owns only the request metadata; the
 * referenced source and destination buffers must outlive @ref execute.
 */
class copy_batch {
 public:
  /**
   * @brief Add a copy request to the batch.
   *
   * Zero-byte requests are ignored.
   *
   * @param[out] dst_ptr Destination pointer.
   * @param[in] src_ptr Source pointer.
   * @param[in] size_in_bytes Number of bytes to copy.
   */
  void add(std::byte* dst_ptr, std::byte const* src_ptr, size_t size_in_bytes);

  /**
   * @brief Reserve storage for at least @p num_requests copy requests.
   */
  void reserve(size_t num_requests);

  /**
   * @brief Return true if the batch contains no copy requests.
   */
  [[nodiscard]] bool empty() const;

  /**
   * @brief Return the number of copy requests in the batch.
   */
  [[nodiscard]] size_t size() const;

  /**
   * @brief Execute all queued copies on a CUDA stream.
   *
   * @param[in] stream CUDA stream used for the batched memcpy.
   * @param[in] total_copy_multiplier Target total copy volume as a multiple of queued copy bytes.
   * Must be >= 1.0. Values greater than 1.0 add dummy memcpy work against the same buffers.
   */
  void execute(rmm::cuda_stream_view stream, double total_copy_multiplier = 1.0) const;

 private:
  /**
   * @brief Non-owning metadata for one copy operation.
   */
  struct copy_request {
    std::byte* dst_ptr;
    std::byte const* src_ptr;
    size_t size_in_bytes;
  };

  std::vector<copy_request> _requests;
};

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
