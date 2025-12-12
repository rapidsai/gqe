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

#include <gqe/executor/join.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace {

__global__ void set_boolean_mask_kernel(bool* boolean_mask,
                                        cudf::size_type const* indices,
                                        cudf::size_type num_indices)
{
  for (cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_indices;
       idx += gridDim.x * blockDim.x) {
    boolean_mask[indices[idx]] = true;
  }
}

__global__ void increment_counts_kernel(int32_t* counts,
                                        cudf::size_type const* indices,
                                        cudf::size_type num_indices)
{
  for (cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_indices;
       idx += gridDim.x * blockDim.x) {
    counts[indices[idx]]++;
  }
}

}  // namespace

void gqe::detail::set_boolean_mask(cudf::mutable_column_view boolean_mask,
                                   cudf::column_view indices)
{
  GQE_EXPECTS(boolean_mask.type().id() == cudf::type_id::BOOL8,
              "The input column of set_boolean_mask is not a boolean column");
  if (indices.is_empty()) { return; }

  int constexpr block_size    = 128;
  constexpr int shared_memory = 0;
  auto const grid_size =
    gqe::utility::detect_launch_grid_size(set_boolean_mask_kernel, block_size, shared_memory);

  set_boolean_mask_kernel<<<grid_size, block_size>>>(
    boolean_mask.data<bool>(), indices.data<cudf::size_type>(), indices.size());
}

void gqe::detail::increment_counts(cudf::mutable_column_view counts, cudf::column_view indices)
{
  GQE_EXPECTS(counts.type().id() == cudf::type_id::INT32,
              "The input column of increment_counts does not have type int32");
  if (indices.is_empty()) { return; }

  int constexpr block_size    = 128;
  constexpr int shared_memory = 0;
  auto const grid_size =
    gqe::utility::detect_launch_grid_size(increment_counts_kernel, block_size, shared_memory);

  increment_counts_kernel<<<grid_size, block_size>>>(
    counts.data<int32_t>(), indices.data<cudf::size_type>(), indices.size());
}

template <typename T>
void gqe::detail::sequence(T* begin, T* end, T start, T step)
{
  thrust::sequence(thrust::device, begin, end, start, step);
}

template void gqe::detail::sequence<cudf::size_type>(cudf::size_type* begin,
                                                     cudf::size_type* end,
                                                     cudf::size_type start,
                                                     cudf::size_type step);
