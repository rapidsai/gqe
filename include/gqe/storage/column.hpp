/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/storage/buffer.hpp>
#include <gqe/storage/sliced_columns.cuh>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>

#include <cuda/stream>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace gqe {

namespace detail {

/**
 * @brief Read the character offsets at which a strings column view begins and ends.
 *
 * The two values sit one row count apart in the offsets column, so they are read individually
 * rather than as a range. Both reads are issued on @p stream and awaited by one synchronization.
 *
 * @tparam offsets_type Type of the offsets, which cuDF defines as INT32 or INT64.
 *
 * @param offsets Offsets column of the strings column, which is not sliced by its parent.
 * @param first_index Index of the first row of the strings column view, within @p offsets.
 * @param last_index Index one past its last row, within @p offsets.
 * @param stream CUDA stream used for the reads.
 * @return The character offsets of the first and of the last row, widened to 64 bits.
 */
template <typename offsets_type>
std::pair<int64_t, int64_t> read_char_bounds(cudf::column_view offsets,
                                             cudf::size_type first_index,
                                             cudf::size_type last_index,
                                             cuda::stream_ref stream)
{
  std::array<offsets_type, 2> bounds{};
  auto const* offsets_begin = offsets.template head<offsets_type>() + offsets.offset();
  GQE_CUDA_TRY(cudaMemcpyAsync(bounds.data(),
                               offsets_begin + first_index,
                               sizeof(offsets_type),
                               cudaMemcpyDefault,
                               stream.get()));
  GQE_CUDA_TRY(cudaMemcpyAsync(bounds.data() + 1,
                               offsets_begin + last_index,
                               sizeof(offsets_type),
                               cudaMemcpyDefault,
                               stream.get()));
  stream.sync();
  return {static_cast<int64_t>(bounds[0]), static_cast<int64_t>(bounds[1])};
}

}  // namespace detail

/**
 * @brief Allocator-aware recursive column storage.
 *
 * @tparam Allocator Allocator used for persistent host-side child metadata.
 */
template <class Allocator = std::allocator<std::byte>>
class column {
 public:
  using allocator_type = gqe::storage_allocator_t<Allocator>;
  template <class T>
  using vector_type = gqe::storage_vector_t<Allocator, T>;

  column() : _children(gqe::storage_rebind_allocator_t<Allocator, column>(allocator_type{})) {}

  column(cudf::column_view source,
         cuda::stream_ref stream,
         rmm::device_async_resource_ref mr,
         allocator_type const& alloc = allocator_type{})
    : _type(source.type()),
      _size(source.size()),
      _null_count(source.null_count()),
      _children(gqe::storage_rebind_allocator_t<Allocator, column>(alloc))
  {
    // A stored column starts at row zero, because view() hands out an offset of zero. A sliced
    // source starts at source.offset() instead, and that offset applies to the data buffer and to
    // the null mask alike, so both are rebased on it below.
    //
    // A sliced column whose children carry offsets of their own is rejected, which today means
    // lists and structs. Their children are handed out unsliced, like the offsets child of a
    // strings column below, but unlike that child they are not a single rebasable range.
    auto const is_string = source.type().id() == cudf::type_id::STRING;
    GQE_EXPECTS(source.offset() == 0 || cudf::is_fixed_width(source.type()) || is_string,
                "Storing a sliced column is supported for fixed-width and string types only");

    // The characters of a strings column are addressed through its offsets, so the range to copy
    // is read from them rather than derived from the row count. Not
    // cudf::strings_column_view::chars_size(), which is documented to ignore a sliced parent, both
    // for a non-zero offset and for a reduced row count.
    // A strings column that carries no offsets at all is empty and has nothing to address.
    auto const has_string_offsets =
      is_string && source.num_children() > 0 && source.child(0).size() > 0;
    int64_t first_char_offset = 0;
    int64_t last_char_offset  = 0;
    if (has_string_offsets) {
      auto const offsets = cudf::strings_column_view(source).offsets();
      GQE_EXPECTS(
        offsets.type().id() == cudf::type_id::INT32 || offsets.type().id() == cudf::type_id::INT64,
        "String offsets must be of type INT32 or INT64");
      std::tie(first_char_offset, last_char_offset) =
        offsets.type().id() == cudf::type_id::INT32
          ? detail::read_char_bounds<int32_t>(
              offsets, source.offset(), source.offset() + source.size(), stream)
          : detail::read_char_bounds<int64_t>(
              offsets, source.offset(), source.offset() + source.size(), stream);
    }

    std::size_t data_buffer_size        = 0;
    std::byte const* source_data_buffer = nullptr;
    if (source.head<std::byte>() != nullptr && source.size() != 0) {
      if (is_string) {
        data_buffer_size   = static_cast<std::size_t>(last_char_offset - first_char_offset);
        source_data_buffer = source.head<std::byte>() + first_char_offset;
      } else if (cudf::is_fixed_width(source.type())) {
        auto const element_size = static_cast<std::size_t>(cudf::size_of(source.type()));
        data_buffer_size        = static_cast<std::size_t>(source.size()) * element_size;
        // Not source.data<std::byte>(), which advances the head pointer by one byte per row rather
        // than by one element, because it counts in units of its template type.
        source_data_buffer =
          source.head<std::byte>() + static_cast<std::size_t>(source.offset()) * element_size;
      }
    }
    if (data_buffer_size > 0) {
      _data.allocate(data_buffer_size, stream, mr);
      GQE_CUDA_TRY(cudaMemcpyAsync(
        _data.data(), source_data_buffer, data_buffer_size, cudaMemcpyDefault, stream.get()));
    }

    // The allocation follows the cuDF convention, which pads a null mask to a multiple of 64
    // bytes. cuDF APIs that copy a mask, such as cudf::copy_bitmask, copy that padded size, so a
    // mask allocated any smaller is read out of bounds once the view reaches such an API.
    //
    // Only the meaningful words are copied. The padding is left uninitialized, because bits at or
    // above the row count are undefined. Copying the padded size instead would read out of bounds
    // whenever the source is a view onto a mask that is not padded.
    auto const has_null_mask =
      source.null_count() > 0 && source.null_mask() != nullptr && source.size() > 0;
    if (has_null_mask) {
      auto const null_mask_copy_size =
        static_cast<std::size_t>(cudf::num_bitmask_words(source.size())) *
        sizeof(cudf::bitmask_type);
      _null_mask.allocate(cudf::bitmask_allocation_size_bytes(source.size()), stream, mr);

      if (source.offset() == 0) {
        GQE_CUDA_TRY(cudaMemcpyAsync(_null_mask.data(),
                                     source.null_mask(),
                                     null_mask_copy_size,
                                     cudaMemcpyDefault,
                                     stream.get()));
      } else {
        // The mask is rebased by shifting its bits down, which a pointer cannot express, because
        // source.offset() is a bit position and source.null_mask() is the base pointer regardless.
        // The shift writes the stored mask directly. cudf::copy_bitmask would shift into a
        // temporary that a second copy then moves here.
        gqe::storage::rebase_null_mask(static_cast<cudf::bitmask_type*>(_null_mask.data()),
                                       source.null_mask(),
                                       source.offset(),
                                       source.offset() + source.size(),
                                       stream);
      }
    }

    _children.reserve(source.num_children());
    if (has_string_offsets) {
      // The offsets child is handed out unsliced, so it is sliced here to the rows of this column
      // and then rebased onto the characters copied above. Storing it unsliced instead would keep
      // the offsets of the whole source, which index into characters that were not copied.
      auto const offsets = cudf::strings_column_view(source).offsets();
      const cudf::column_view sliced_offsets(offsets.type(),
                                             source.size() + 1,
                                             offsets.head<void>(),
                                             /* null_mask = */ nullptr,
                                             /* null_count = */ 0,
                                             offsets.offset() + source.offset());
      _children.emplace_back(sliced_offsets, stream, mr, alloc);

      auto* stored_offsets          = _children.back()._data.data();
      auto const num_stored_offsets = static_cast<size_t>(source.size()) + 1;
      if (offsets.type().id() == cudf::type_id::INT32) {
        gqe::storage::rebase_string_offsets(static_cast<int32_t*>(stored_offsets),
                                            num_stored_offsets,
                                            static_cast<int32_t>(first_char_offset),
                                            stream);
      } else {
        gqe::storage::rebase_string_offsets(
          static_cast<int64_t*>(stored_offsets), num_stored_offsets, first_char_offset, stream);
      }
    } else {
      for (auto idx = 0; idx < source.num_children(); ++idx) {
        _children.emplace_back(source.child(idx), stream, mr, alloc);
      }
    }
  }

  ~column() = default;

  column(column const&)                = delete;
  column& operator=(column const&)     = delete;
  column(column&&) noexcept            = default;
  column& operator=(column&&) noexcept = default;

  [[nodiscard]] cudf::column_view view() const
  {
    std::vector<cudf::column_view> child_views(_children.size());
    for (std::size_t idx = 0; idx < _children.size(); ++idx) {
      child_views[idx] = _children[idx].view();
    }

    auto const* null_mask =
      _null_count > 0 ? static_cast<cudf::bitmask_type const*>(_null_mask.data()) : nullptr;
    return cudf::column_view(
      _type, _size, static_cast<void const*>(_data.data()), null_mask, _null_count, 0, child_views);
  }

  [[nodiscard]] cudf::mutable_column_view mutable_view()
  {
    std::vector<cudf::mutable_column_view> child_views(_children.size());
    for (std::size_t idx = 0; idx < _children.size(); ++idx) {
      child_views[idx] = _children[idx].mutable_view();
    }

    auto* null_mask =
      _null_count > 0 ? static_cast<cudf::bitmask_type*>(_null_mask.data()) : nullptr;
    return cudf::mutable_column_view(
      _type, _size, static_cast<void*>(_data.data()), null_mask, _null_count, 0, child_views);
  }

  [[nodiscard]] cudf::data_type type() const noexcept { return _type; }
  [[nodiscard]] cudf::size_type size() const noexcept { return _size; }
  [[nodiscard]] cudf::size_type null_count() const noexcept { return _null_count; }
  [[nodiscard]] std::size_t data_size() const noexcept { return _data.size(); }
  [[nodiscard]] std::size_t null_mask_size() const noexcept { return _null_mask.size(); }

  [[nodiscard]] std::size_t total_data_size() const noexcept
  {
    std::size_t total = data_size() + null_mask_size();
    for (auto const& child : _children) {
      total += child.total_data_size();
    }
    return total;
  }

  [[nodiscard]] column const& child(cudf::size_type index) const
  {
    assert(index >= 0);
    return _children.at(static_cast<std::size_t>(index));
  }

  [[nodiscard]] column& child(cudf::size_type index)
  {
    assert(index >= 0);
    return _children.at(static_cast<std::size_t>(index));
  }

  [[nodiscard]] cudf::size_type num_children() const noexcept
  {
    assert(_children.size() <=
           static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
    return static_cast<cudf::size_type>(_children.size());
  }

  void deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    for (auto& child : _children) {
      child.deallocate(stream, mr);
    }
    _children.clear();
    _data.deallocate(stream, mr);
    _null_mask.deallocate(stream, mr);
    _type       = cudf::data_type(cudf::type_id::EMPTY);
    _size       = 0;
    _null_count = 0;
  }

 private:
  cudf::data_type _type{cudf::type_id::EMPTY};
  cudf::size_type _size{};
  cudf::size_type _null_count{};
  gqe::storage::buffer _data;
  gqe::storage::buffer _null_mask;
  vector_type<column> _children;
};

}  // namespace gqe
