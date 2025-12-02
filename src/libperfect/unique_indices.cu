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

#include "xor_hash_table.cuh"

#include "unique_indices.hpp"

namespace libperfect {

static std::vector<ConstCudaGpuBufferPointer> columns_to_buffers(
  std::vector<cudf::column_view> const& columns)
{
  std::vector<ConstCudaGpuBufferPointer> ret;
  for (uint column_index = 0; column_index < columns.size(); column_index++) {
    const auto& current_column = columns[column_index];
    ret.emplace_back(current_column.data<int>(), current_column.type().id());
  }
  return ret;
}

std::tuple<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
unique_indices(std::vector<cudf::column_view> const& key_columns,
               cudf::column_view const& mask,
               rmm::cuda_stream_view stream)
{
  if (key_columns.empty()) { throw std::invalid_argument("key_columns is empty"); }
  if (key_columns[0].is_empty()) {
    return std::make_tuple(rmm::device_uvector<cudf::size_type>(0, stream),
                           rmm::device_uvector<cudf::size_type>(0, stream));
  }
  PUSH_RANGE("perfect unique indices", 0);
  PUSH_RANGE("make hash table", 1);
  auto key_buffers = columns_to_buffers(key_columns);
  auto keys_numel  = key_columns[0].size();
  std::optional<ConstCudaGpuBufferPointer> mask_buffer;
  if (!mask.is_empty()) { mask_buffer.emplace(mask.data<int>(), mask.type().id()); }

  auto hash_table = xor_hash_table::make_hash_table(key_buffers, keys_numel, std::nullopt, stream);
  POP_RANGE();
  auto ret = hash_table.template bulk_insert<xor_hash_table::CheckEquality::True,
                                             xor_hash_table::InsertOutput::True>(
    key_buffers, keys_numel, mask_buffer, stream);
  auto& unique_element_indices = std::get<0>(ret).get_buffer();
  auto& group_indices          = std::get<1>(ret).get_buffer();
  POP_RANGE();
  return std::make_tuple(std::move(unique_element_indices), std::move(group_indices));
}

}  // namespace libperfect
