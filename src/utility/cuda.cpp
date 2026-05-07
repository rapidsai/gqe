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

#include <gqe/utility/cuda.hpp>

#include <gqe/device_properties.hpp>
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

void do_batched_memcpy(
  void** dst_ptrs, void** src_ptrs, size_t* sizes, size_t num_buffers, cudaStream_t stream)
{
  assert(num_buffers > 0 && "Must at least copy a single buffer");
  std::vector<cudaMemcpyAttributes> attrs(1);
  attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
  attrs[0].flags                = 0;
  std::vector<size_t> attrsIdxs = {0};
  size_t numAttrs               = attrs.size();
  size_t fail_idx;
#ifndef NDEBUG
  for (size_t i = 0; i < num_buffers; ++i) {
    GQE_LOG_DEBUG("i = {}, dst_ptrs[i] = {}, src_ptrs[i] = {}, sizes[i] = {}",
                  i,
                  (void*)dst_ptrs[i],
                  (void*)src_ptrs[i],
                  sizes[i]);
  }
#endif
  GQE_CUDA_TRY(cudaMemcpyBatchAsync((void**)dst_ptrs,
                                    (void**)src_ptrs,
                                    sizes,
                                    num_buffers,
                                    attrs.data(),
                                    attrsIdxs.data(),
                                    numAttrs,
                                    &fail_idx,
                                    stream));
}

}  // namespace utility

}  // namespace gqe
