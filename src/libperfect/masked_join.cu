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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

#include "xor_hash_table.cuh"

#include "masked_join.hpp"

namespace libperfect {

// We'll do a bunch of lookups and then we need to process the result
// of each lookup as it happens.  The processing for us will involve
// putting the answer into one output list or another.
std::tuple<CudaGpuArray<cudf::size_type>,
           std::optional<CudaGpuArray<cudf::size_type>>,
           std::optional<CudaGpuBuffer>,
           std::optional<CudaGpuBuffer>>
masked_join(const std::vector<ConstCudaGpuBufferPointer>& left_keys,
            const size_t left_keys_numel,
            const std::vector<ConstCudaGpuBufferPointer>& right_keys,
            const size_t right_keys_numel,
            const std::optional<ConstCudaGpuBufferPointer>& left_mask,
            const std::optional<ConstCudaGpuBufferPointer>& right_mask,
            const bool& left_unique,
            const bool& right_unique,
            const bool& return_all)
{
  // Make a hash table backed by a tensor.  The xors are needed for hashing.
  PUSH_RANGE("perfect join", 0);
  PUSH_RANGE("build hash", 1);
  auto hash_table = xor_hash_table::make_hash_table(
    left_keys, left_keys_numel, std::make_pair(right_keys, right_keys_numel));
  POP_RANGE();
  // Do all the inserts.
  PUSH_RANGE("insert", 1);
  hash_table.template bulk_insert<xor_hash_table::CheckEquality::False,
                                  xor_hash_table::InsertOutput::False>(
    left_keys, left_keys_numel, left_mask);
  POP_RANGE();

  PUSH_RANGE("lookup", 1);
  auto ret = hash_table.bulk_lookup(right_keys,
                                    right_keys_numel,
                                    right_mask,
                                    left_keys,
                                    left_keys_numel,
                                    left_unique,
                                    right_unique,
                                    return_all);
  POP_RANGE();
  POP_RANGE();
  return ret;
}

std::vector<ConstCudaGpuBufferPointer> table_to_buffers(cudf::table_view const& t)
{
  std::vector<ConstCudaGpuBufferPointer> ret;
  for (int column_index = 0; column_index < t.num_columns(); column_index++) {
    const auto& current_column = t.column(column_index);
    ret.emplace_back(current_column.data<int>(), current_column.type().id());
  }
  return ret;
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys,
             const cudf::table_view& right_keys,
             const cudf::column_view& left_mask,
             const cudf::column_view& right_mask)
{
  if (right_keys.num_rows() == 0 || left_keys.num_rows() == 0) {
    return std::make_pair(
      std::make_unique<rmm::device_uvector<cudf::size_type>>(0, rmm::cuda_stream_default),
      std::make_unique<rmm::device_uvector<cudf::size_type>>(0, rmm::cuda_stream_default));
  }

  for (auto column_id = 0; column_id < left_keys.num_columns(); column_id++) {
    auto const& column = left_keys.column(column_id);
    if (column.has_nulls()) {
      throw std::logic_error("Perfect hashing requires that both sides have no nulls");
    }
  }
  for (auto column_id = 0; column_id < right_keys.num_columns(); column_id++) {
    auto const& column = right_keys.column(column_id);
    if (column.has_nulls()) {
      throw std::logic_error("Perfect hashing requires that both sides have no nulls");
    }
  }
  std::vector<ConstCudaGpuBufferPointer> left  = table_to_buffers(left_keys);
  std::vector<ConstCudaGpuBufferPointer> right = table_to_buffers(right_keys);
  if (!left_mask.is_empty()) {
    assert(left_mask.type().id() == cudf::type_id::BOOL8);
    assert(left_mask.size() == left_keys.num_rows());
  }
  if (!right_mask.is_empty()) {
    assert(right_mask.type().id() == cudf::type_id::BOOL8);
    assert(right_mask.size() == right_keys.num_rows());
  }
  std::optional<ConstCudaGpuBufferPointer> left_mask_ptr =
    left_mask.is_empty() ? std::nullopt
                         : std::make_optional(ConstCudaGpuBufferPointer(left_mask.data<bool>(),
                                                                        left_mask.type().id()));
  std::optional<ConstCudaGpuBufferPointer> right_mask_ptr =
    right_mask.is_empty() ? std::nullopt
                          : std::make_optional(ConstCudaGpuBufferPointer(right_mask.data<bool>(),
                                                                         right_mask.type().id()));
  auto ret          = masked_join(left,
                         left_keys.num_rows(),
                         right,
                         right_keys.num_rows(),
                         left_mask_ptr,
                         right_mask_ptr,
                         true,
                         false,
                         false);
  auto left_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
    std::move(std::get<0>(ret).get_buffer()));
  auto right_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
    std::move(std::get<1>(ret)->get_buffer()));
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace libperfect
