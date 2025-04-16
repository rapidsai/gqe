/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/join.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

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

  int constexpr block_size = 128;
  auto const grid_size     = gqe::utility::divide_round_up(indices.size(), block_size);

  set_boolean_mask_kernel<<<grid_size, block_size>>>(
    boolean_mask.data<bool>(), indices.data<cudf::size_type>(), indices.size());
}

void gqe::detail::increment_counts(cudf::mutable_column_view counts, cudf::column_view indices)
{
  GQE_EXPECTS(counts.type().id() == cudf::type_id::INT32,
              "The input column of increment_counts does not have type int32");
  if (indices.is_empty()) { return; }

  int constexpr block_size = 128;
  auto const grid_size     = gqe::utility::divide_round_up(indices.size(), block_size);

  increment_counts_kernel<<<grid_size, block_size>>>(
    counts.data<int32_t>(), indices.data<cudf::size_type>(), indices.size());
}
