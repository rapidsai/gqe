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

#include <gqe/executor/perfect_hash/masked_join.hpp>

#include <gqe/executor/perfect_hash/detail/xor_hash_table.cuh>
#include <gqe/utility/error.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <stdexcept>

namespace gqe::perfect_hash {

namespace {

/**
 * @brief Type-dispatch functor that gets a fixed-width column's device data pointer.
 *
 * `table_to_buffers` dispatches each column's logical cuDF type to its underlying device storage
 * type and invokes this functor. `column.data<StorageType>()` returns a pointer to the column's
 * first logical row, so the pointer accounts for any slice offset.
 *
 * The constrained overload returns the slice-aware pointer for representation-layout-compatible
 * storage types. The fallback rejects other types; this check makes pointer extraction safe but
 * does not define the set of key types supported by the perfect hash.
 */
struct fixed_width_column_data_pointer {
  template <typename StorageType>
    requires(cudf::is_rep_layout_compatible<StorageType>())
  [[nodiscard]] void const* operator()(cudf::column_view const& column) const
  {
    return column.data<StorageType>();
  }

  template <typename StorageType>
    requires(!cudf::is_rep_layout_compatible<StorageType>())
  [[nodiscard]] void const* operator()(cudf::column_view const&) const
  {
    throw std::invalid_argument(
      annotate_line("Perfect hashing supports only fixed-width key columns"));
  }
};

}  // namespace

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
            const bool& return_all,
            cuda::stream_ref stream,
            rmm::device_async_resource_ref output_mr)
{
  // Make a hash table backed by a tensor.  The xors are needed for hashing.
  PUSH_RANGE("perfect join", 0);
  PUSH_RANGE("build hash", 1);
  auto hash_table = xor_hash_table::make_hash_table(
    left_keys, left_keys_numel, std::make_pair(right_keys, right_keys_numel), stream);
  POP_RANGE();
  // Do all the inserts.
  PUSH_RANGE("insert", 1);
  hash_table.template bulk_insert<xor_hash_table::CheckEquality::False,
                                  xor_hash_table::InsertOutput::False>(
    left_keys, left_keys_numel, left_mask, stream);
  POP_RANGE();

  PUSH_RANGE("lookup", 1);
  auto ret = hash_table.bulk_lookup(right_keys,
                                    right_keys_numel,
                                    right_mask,
                                    left_keys,
                                    left_keys_numel,
                                    left_unique,
                                    right_unique,
                                    return_all,
                                    stream,
                                    output_mr);
  POP_RANGE();
  POP_RANGE();
  return ret;
}

std::vector<ConstCudaGpuBufferPointer> table_to_buffers(cudf::table_view const& t)
{
  std::vector<ConstCudaGpuBufferPointer> ret;
  for (int column_index = 0; column_index < t.num_columns(); column_index++) {
    const auto& current_column = t.column(column_index);
    auto const data            = cudf::type_dispatcher<cudf::dispatch_storage_type>(
      current_column.type(), fixed_width_column_data_pointer{}, current_column);
    ret.emplace_back(data, current_column.type().id());
  }
  return ret;
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys,
             const cudf::table_view& right_keys,
             const cudf::column_view& left_mask,
             const cudf::column_view& right_mask,
             cuda::stream_ref stream)
{
  return perfect_join(
    left_keys, right_keys, left_mask, right_mask, stream, cudf::get_current_device_resource_ref());
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys,
             const cudf::table_view& right_keys,
             const cudf::column_view& left_mask,
             const cudf::column_view& right_mask,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref output_mr)
{
  if (right_keys.num_rows() == 0 || left_keys.num_rows() == 0) {
    return std::make_pair(
      std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, output_mr),
      std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, output_mr));
  }

  for (auto column_id = 0; column_id < left_keys.num_columns(); column_id++) {
    GQE_EXPECTS(!left_keys.column(column_id).has_nulls(),
                "Perfect hashing requires that both sides have no nulls",
                std::invalid_argument);
  }
  for (auto column_id = 0; column_id < right_keys.num_columns(); column_id++) {
    GQE_EXPECTS(!right_keys.column(column_id).has_nulls(),
                "Perfect hashing requires that both sides have no nulls",
                std::invalid_argument);
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
                         false,
                         stream,
                         output_mr);
  auto left_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
    std::move(std::get<0>(ret).get_buffer()));
  auto right_indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
    std::move(std::get<1>(ret)->get_buffer()));
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace gqe::perfect_hash
