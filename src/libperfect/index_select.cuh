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

#include "query_common.hpp"

namespace libperfect {

template <typename input_type, typename indices_type, typename output_type>
__global__ void index_select_kernel(input_type input,
                                    indices_type indices,
                                    output_type output,
                                    size_t input_size)
{
  auto const thread_in_block   = threadIdx.x;
  auto const block_in_grid     = blockIdx.x;
  auto const threads_per_block = blockDim.x;
  auto const blocks_per_grid   = gridDim.x;
  auto const thread_in_grid    = block_in_grid * threads_per_block + thread_in_block;
  auto const threads_per_grid  = threads_per_block * blocks_per_grid;
  for (auto i = thread_in_grid; i < input_size; i += threads_per_grid) {
    output[i] = input[indices[i]];
  }
}

template <typename input_type, typename indices_type>
CudaGpuArray<std::remove_const_t<input_type>> index_select(
  CudaGpuArray<input_type> const& input, CudaGpuArray<indices_type> const& indices)
{
  auto const threads_per_block = 1024;
  auto const block_count       = div_round_up(indices.numel(), size_t(threads_per_block));
  auto output                  = CudaGpuArray<std::remove_const_t<input_type>>(indices.numel());
  index_select_kernel<<<block_count, threads_per_block>>>(
    input.get(), indices.get(), output.get(), input.numel());
  return output;
}

}  // namespace libperfect
