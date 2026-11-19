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

/* IMPORTANT NOTE: Any changes to the in-class members  will break serialization and
   deserialization. Both need to be updated.*/

#include <gqe/executor/aggregate.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/serialization.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <google/protobuf/repeated_field.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace gqe {
namespace storage {

namespace {

/**
 * @brief Split a buffer's compressed byte count into the primary and secondary reporting buckets.
 *
 * Only the totals surfaced by `get_compression_stats` depend on this attribution. Decoding uses
 * `buffer.compression_format` directly, so a buffer whose codec no longer matches @p config is
 * still decoded correctly and merely lands in the primary bucket.
 *
 * @tparam BufferStorage The enclosing storage's nested `buffer_storage`, deduced.
 */
template <class BufferStorage>
void attribute_compressed_size(BufferStorage& buffer, gqe::compression::configuration const& config)
{
  bool const compressed = buffer.compression_format != gqe::compression_format::none;
  bool const secondary =
    compressed && buffer.compression_format == config.secondary_compression_format;
  buffer.primary_compressed_size   = (compressed && !secondary) ? buffer.compressed_size : 0;
  buffer.secondary_compressed_size = secondary ? buffer.compressed_size : 0;
}

/// Uncompressed byte size of a fixed-width data column.
[[nodiscard]] size_t uncompressed_data_bytes(cudf::data_type dtype, int64_t num_rows)
{
  if (num_rows <= 0) { return 0; }
  return static_cast<size_t>(num_rows) * cudf::size_of(dtype);
}

/// Uncompressed byte size of a partitioned null mask (per-partition 32-bit word rounding).
[[nodiscard]] size_t null_mask_uncompressed_bytes(size_t num_rows, size_t partition_size)
{
  if (num_rows == 0) { return 0; }
  size_t const npart      = gqe::utility::divide_round_up(num_rows, partition_size);
  size_t const tail_rows  = num_rows - (npart - 1) * partition_size;
  size_t const full_words = gqe::utility::divide_round_up(partition_size, 32);
  size_t const tail_words = gqe::utility::divide_round_up(tail_rows, 32);
  return ((npart - 1) * full_words + tail_words) * 4;
}

/// Split @p contiguous into partitions of the given byte sizes and compress them with @p config.
/// Returns the compressed partition buffers and reports the selected format via @p out_format.
[[nodiscard]] std::vector<rmm::device_buffer> compress_contiguous_partitions(
  gqe::compression::configuration const& config,
  cudf::data_type element_type,
  rmm::device_buffer const& contiguous,
  std::vector<size_t> const& partition_byte_sizes,
  memory_kind::type memory_kind,
  gqe::compression_format& out_format,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<rmm::device_buffer> partitions;
  partitions.reserve(partition_byte_sizes.size());
  auto const* base = static_cast<std::byte const*>(contiguous.data());
  size_t offset    = 0;
  for (size_t const bytes : partition_byte_sizes) {
    rmm::device_buffer partition(bytes, stream, mr);
    if (bytes > 0) {
      GQE_CUDA_TRY(cudaMemcpyAsync(
        partition.data(), base + offset, bytes, cudaMemcpyDeviceToDevice, stream.get()));
    }
    partitions.push_back(std::move(partition));
    offset += bytes;
  }
  std::vector<cudf::size_type> compressed_sizes;
  size_t compressed_size = 0, uncompressed_size = 0, primary_size = 0, secondary_size = 0;
  auto const data_type = gqe::compression::get_optimal_nvcomp_data_type(element_type.id());
  return gqe::compression::compress_batch(config,
                                          data_type,
                                          std::move(partitions),
                                          out_format,
                                          compressed_size,
                                          uncompressed_size,
                                          primary_size,
                                          secondary_size,
                                          compressed_sizes,
                                          memory_kind,
                                          /*try_secondary=*/false,
                                          stream,
                                          mr);
}

}  // namespace

bool compressed_sliced_buffer_view::is_compressed() const
{
  return _compression_format != gqe::compression_format::none;
}

gqe::compression_format compressed_sliced_buffer_view::compression_format() const
{
  return _compression_format;
}

cudf::data_type compressed_sliced_buffer_view::element_type() const { return _element_type; }

partition_span compressed_sliced_buffer_view::get_partition(size_t partition_idx) const
{
  if (partition_idx >= _buffers.size()) {
    throw std::out_of_range("compressed_sliced_buffer_view partition index out of range");
  }
  auto const* data = reinterpret_cast<uint8_t const*>(_buffers[partition_idx].data());
  return partition_span{data, _buffers[partition_idx].size()};
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  cudf::column&& cudf_column,
  int partition_size,
  memory_kind::type memory_kind,
  gqe::compression::configuration compression_config,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  allocator_type const& alloc)
  : compressed_sliced_column_storage(
      cudf_column, partition_size, std::move(compression_config), alloc)
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  const cudf::column& cudf_column,
  int partition_size,
  gqe::compression::configuration compression_config,
  allocator_type const& alloc)
  : _cudf_type(cudf_column.type()),
    _null_counts(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc)),
    _compression_config(std::move(compression_config)),
    _data_buffer(cudf_column.type(), alloc),
    _null_mask_buffer(cudf::data_type(cudf::type_id::INT32), alloc)
{
  if (partition_size <= 0) {
    throw std::invalid_argument("compressed_sliced_column requires a positive partition size.");
  }
  _partition_size = static_cast<size_t>(partition_size);
  _size           = cudf_column.size();
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::do_compress(
  rmm::device_buffer const* input,
  typename compressed_sliced_column_storage<Allocator>::buffer_storage& output,
  size_t num_rows,
  size_t num_partitions,
  bool is_null_mask,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  utility::nvtx_scoped_range compress_batch_range("compress_batch");

  const uint8_t* bytes_buffer = reinterpret_cast<const uint8_t*>(input->data());
  size_t start_ix             = 0;
  auto const data_type = gqe::compression::get_optimal_nvcomp_data_type(output.element_type.id());

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;
  device_uncompressed_data_buffers.reserve(num_partitions);
  input_ptrs.reserve(num_partitions);
  input_sizes.reserve(num_partitions);
  device_ptrs.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t row_start_ix = ix_partition * _partition_size;
    size_t row_count    = std::min(num_rows - row_start_ix, _partition_size);
    size_t size         = row_count * cudf::size_of(output.element_type);
    if (is_null_mask) { size = gqe::utility::divide_round_up(row_count, 32) * 4; }

    device_uncompressed_data_buffers.emplace_back(size, stream, mr);
    input_ptrs.push_back(reinterpret_cast<const void*>(bytes_buffer + start_ix));
    input_sizes.push_back(size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back().data()));
    start_ix += size;
  }

  gqe::utility::do_batched_memcpy(device_ptrs.data(),
                                  const_cast<void**>(input_ptrs.data()),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream.get());

  bool try_secondary_compression = is_null_mask ? false : true;
  std::vector<cudf::size_type> compressed_sizes;
  auto compressed_buffers =
    gqe::compression::compress_batch(_compression_config,
                                     data_type,
                                     std::move(device_uncompressed_data_buffers),
                                     output.compression_format,
                                     output.compressed_size,
                                     output.uncompressed_size,
                                     output.primary_compressed_size,
                                     output.secondary_compressed_size,
                                     compressed_sizes,
                                     memory_kind,
                                     try_secondary_compression,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  output.buffers.reserve(compressed_buffers.size());
  for (auto const& compressed_buffer : compressed_buffers) {
    output.buffers.emplace_back(compressed_buffer, stream, mr);
  }

  stream.sync();
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::compress(cudf::column&& cudf_column,
                                                           memory_kind::type memory_kind,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  const size_t num_rows        = cudf_column.size();
  const size_t num_partitions  = gqe::utility::divide_round_up(num_rows, _partition_size);
  const auto column_null_count = cudf_column.null_count();
  // Setting up offsets for the cudf slice API

  auto column_content = cudf_column.release();
  assert(column_content.children.empty());

  do_compress(column_content.data.get(),
              _data_buffer,
              num_rows,
              num_partitions,
              false /*is_null_mask*/,
              memory_kind,
              stream,
              mr);

  if (column_null_count > 0) {
    do_compress(column_content.null_mask.get(),
                _null_mask_buffer,
                num_rows,
                num_partitions,
                true /*is_null_mask*/,
                memory_kind,
                stream,
                mr);
    rmm::device_buffer null_mask_buffer(
      column_content.null_mask->data(), column_content.null_mask->size(), stream, mr);
    _null_counts.reserve(num_partitions);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      size_t start_ix = ix_partition * _partition_size;
      size_t stop_ix  = std::min(start_ix + _partition_size, num_rows);
      _null_counts.push_back(cudf::null_count(
        reinterpret_cast<const uint32_t*>(null_mask_buffer.data()), start_ix, stop_ix, stream));
    }
  }
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::deallocate(cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  // TODO: Fix system_memory_resource, pinned_memory_resource, numa_memory_resource, and
  // shared_memory_resource, which immediately free host/shared memory while ignoring
  // the stream. Remove this once they satisfy RMM stream-ordered semantics.
  stream.sync();
  _data_buffer.deallocate(stream, mr);
  _null_mask_buffer.deallocate(stream, mr);
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::size() const
{
  return _size;
}

template <class Allocator>
cudf::size_type compressed_sliced_column_storage<Allocator>::null_count() const
{
  return std::accumulate(_null_counts.begin(), _null_counts.end(), cudf::size_type{0});
}

template <class Allocator>
bool compressed_sliced_column_storage<Allocator>::is_compressed() const
{
  return _data_buffer.compression_format != gqe::compression_format::none;
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::get_compressed_size() const
{
  return static_cast<int64_t>(_data_buffer.compressed_size + _null_mask_buffer.compressed_size);
}

template <class Allocator>
int64_t compressed_sliced_column_storage<Allocator>::get_uncompressed_size() const
{
  return static_cast<int64_t>(_data_buffer.uncompressed_size + _null_mask_buffer.uncompressed_size);
}

template <class Allocator>
column_compression_statistics compressed_sliced_column_storage<Allocator>::get_compression_stats()
  const
{
  fixed_width_compression_statistics fixed_width_stats;
  fixed_width_stats.compressed_size   = get_compressed_size();
  fixed_width_stats.uncompressed_size = get_uncompressed_size();

  fixed_width_stats.primary_compressed_size   = _data_buffer.primary_compressed_size;
  fixed_width_stats.secondary_compressed_size = _data_buffer.secondary_compressed_size;
  auto const primary_format                   = _compression_config.primary_compression_format;
  auto const secondary_format                 = _compression_config.secondary_compression_format;
  auto const secondary_enabled                = secondary_format != gqe::compression_format::none;
  const bool data_is_compressed = _data_buffer.compression_format != gqe::compression_format::none;
  fixed_width_stats.num_primary_compressed_row_groups =
    (data_is_compressed && _data_buffer.compression_format == primary_format) ? 1 : 0;
  fixed_width_stats.num_secondary_compressed_row_groups =
    (data_is_compressed && secondary_enabled && _data_buffer.compression_format == secondary_format)
      ? 1
      : 0;

  fixed_width_stats.num_compressed_row_groups = data_is_compressed ? 1 : 0;
  return column_compression_statistics(fixed_width_stats);
}

template <bool large_string_mode, class Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>::
  string_compressed_sliced_column_storage(cudf::column&& cudf_column,
                                          int partition_size,
                                          memory_kind::type memory_kind,
                                          gqe::compression::configuration compression_config,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          allocator_type const& alloc)
  : compressed_sliced_column_storage<Allocator>(
      cudf_column, partition_size, std::move(compression_config), alloc),
    _offset_buffer(offset_element_type, alloc),
    _partition_char_array_sizes(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc))
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

template <bool large_string_mode, class Allocator>
bool string_compressed_sliced_column_storage<large_string_mode, Allocator>::is_large_string() const
{
  return large_string_mode;
}

/*
 * This is similar to the compress method for the regular compressed sliced column
 * but we need to compress the offsets separately and the char arrays for each slice are variably
 * sized based on the total string size for each slice
 *
 * We define each slice as a contiguous chunk of rows of partition_size
 * There's then an associated character array sized as offset((ix_partition+1) * partition_size) -
 * offset(ix_partition * partition_size) An offset array with one offset per row, where the offset
 * indicates the start position of the row in the string This is different from arrow, where there
 * is one more offset than row with the last offset holding the total string size
 *
 * When decompressing, we'll combine the strings from all the unpruned slices and adjust the offsets
 * to point correctly into the combined char array
 *
 * We'll also make this into valid arrow by adding an extra offset that indicates the total string
 * size
 */
template <bool large_string_mode, class Allocator>
void string_compressed_sliced_column_storage<large_string_mode, Allocator>::compress(
  cudf::column&& cudf_column,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  const size_t num_rows        = cudf_column.size();
  const size_t num_partitions  = gqe::utility::divide_round_up(num_rows, this->_partition_size);
  const auto column_null_count = cudf_column.null_count();

  // First we'll compress the character array
  // These are variable length depending on the row lengths in each partition
  //
  auto column_content = cudf_column.release();
  auto& child_column  = column_content.children.front();
  auto child_content  = child_column->release();

  uint8_t* char_data         = reinterpret_cast<uint8_t*>(column_content.data->data());
  offsets_type* offsets_data = reinterpret_cast<offsets_type*>(child_content.data->data());
  std::vector<offsets_type> offsets_data_host(num_rows + 1);
  GQE_CUDA_TRY(cudaMemcpy(
    offsets_data_host.data(), offsets_data, child_content.data->size(), cudaMemcpyDefault));

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;

  device_uncompressed_data_buffers.reserve(num_partitions);
  input_ptrs.reserve(num_partitions);
  input_sizes.reserve(num_partitions);
  device_ptrs.reserve(num_partitions);
  _partition_char_array_sizes.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t start_offset        = ix_partition * this->_partition_size;
    size_t end_offset          = std::min(start_offset + this->_partition_size, num_rows);
    size_t partition_char_size = offsets_data_host[end_offset] - offsets_data_host[start_offset];
    device_uncompressed_data_buffers.emplace_back(partition_char_size, stream, mr);
    input_ptrs.push_back(
      reinterpret_cast<const void*>(char_data + offsets_data_host[start_offset]));
    input_sizes.push_back(partition_char_size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back().data()));
    _partition_char_array_sizes.push_back(partition_char_size);
  }
  gqe::utility::do_batched_memcpy(device_ptrs.data(),
                                  const_cast<void**>(input_ptrs.data()),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream.get());

  std::vector<cudf::size_type> chars_compressed_sizes;
  auto const chars_data_type =
    gqe::compression::get_optimal_nvcomp_data_type(this->_data_buffer.element_type.id());
  auto chars_buffers =
    gqe::compression::compress_batch(this->_compression_config,
                                     chars_data_type,
                                     std::move(device_uncompressed_data_buffers),
                                     this->_data_buffer.compression_format,
                                     this->_data_buffer.compressed_size,
                                     this->_data_buffer.uncompressed_size,
                                     this->_data_buffer.primary_compressed_size,
                                     this->_data_buffer.secondary_compressed_size,
                                     chars_compressed_sizes,
                                     memory_kind,
                                     true /*try secondary compression*/,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  this->_data_buffer.buffers.reserve(chars_buffers.size());
  for (auto const& chars_buffer : chars_buffers) {
    this->_data_buffer.buffers.emplace_back(chars_buffer, stream, mr);
  }

  // Compress the null mask using the parent class method
  if (column_null_count > 0) {
    // Use do_compress here because the null mask is the same as in the normal compressed sliced
    // column Char buffers and offset arrays are set up differently for strings and can't use the
    // shared helper.
    this->do_compress(column_content.null_mask.get(),
                      this->_null_mask_buffer,
                      num_rows,
                      num_partitions,
                      true /*is_null_mask*/,
                      memory_kind,
                      stream,
                      mr);
    rmm::device_buffer null_mask_buffer(
      column_content.null_mask->data(), column_content.null_mask->size(), stream, mr);
    this->_null_counts.reserve(num_partitions);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      size_t start_ix = ix_partition * this->_partition_size;
      size_t stop_ix  = std::min(start_ix + this->_partition_size, num_rows);
      this->_null_counts.push_back(cudf::null_count(
        reinterpret_cast<const uint32_t*>(null_mask_buffer.data()), start_ix, stop_ix, stream));
    }
  }

  // Adjust the offset array and compress it. We can do the offset adjustment on the CPU because
  // the write task isn't timed.
  input_ptrs.clear();
  input_sizes.clear();
  device_ptrs.clear();
  std::vector<rmm::device_buffer> device_uncompressed_offsets_buffers;
  device_uncompressed_offsets_buffers.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ++ix_partition) {
    size_t ix_start       = ix_partition * this->_partition_size;
    size_t ix_end         = std::min(ix_start + this->_partition_size, num_rows);
    size_t partition_rows = ix_end - ix_start;
    size_t partition_base = offsets_data_host[ix_start];
    // We'll adjust each offset such that for the partition, the offsets start at zero
    for (size_t ix_offset = ix_start; ix_offset < ix_end; ++ix_offset) {
      offsets_data_host[ix_offset] -= partition_base;
    }
    input_ptrs.push_back(reinterpret_cast<const void*>(&offsets_data_host[ix_start]));
    input_sizes.push_back(partition_rows * sizeof(offsets_type));
    device_uncompressed_offsets_buffers.emplace_back(
      partition_rows * sizeof(offsets_type), stream, mr);
    device_ptrs.push_back(
      reinterpret_cast<void*>(device_uncompressed_offsets_buffers.back().data()));
  }
  gqe::utility::do_batched_memcpy(device_ptrs.data(),
                                  const_cast<void**>(input_ptrs.data()),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream.get());

  std::vector<cudf::size_type> offsets_compressed_sizes;
  auto const offsets_data_type =
    gqe::compression::get_optimal_nvcomp_data_type(_offset_buffer.element_type.id());
  auto offsets_buffers =
    gqe::compression::compress_batch(this->_compression_config,
                                     offsets_data_type,
                                     std::move(device_uncompressed_offsets_buffers),
                                     _offset_buffer.compression_format,
                                     _offset_buffer.compressed_size,
                                     _offset_buffer.uncompressed_size,
                                     _offset_buffer.primary_compressed_size,
                                     _offset_buffer.secondary_compressed_size,
                                     offsets_compressed_sizes,
                                     memory_kind,
                                     true /* try secondary compression */,
                                     stream,
                                     mr);
  // rmm::device_buffer does not expose ownership transfer for its allocation, so persist each
  // compressed partition with an explicit copy into gqe::storage::buffer storage.
  _offset_buffer.buffers.reserve(offsets_buffers.size());
  for (auto const& offsets_buffer : offsets_buffers) {
    _offset_buffer.buffers.emplace_back(offsets_buffer, stream, mr);
  }

  stream.sync();
}

template <bool large_string_mode, class Allocator>
size_t string_compressed_sliced_column_storage<large_string_mode, Allocator>::
  get_char_partition_output_size(size_t partition_idx) const
{
  return _partition_char_array_sizes[partition_idx];
}

template <bool large_string_mode, class Allocator>
int64_t string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_compressed_size()
  const
{
  return static_cast<int64_t>(this->_data_buffer.compressed_size + _offset_buffer.compressed_size +
                              this->_null_mask_buffer.compressed_size);
}

template <bool large_string_mode, class Allocator>
int64_t
string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_uncompressed_size() const
{
  return static_cast<int64_t>(this->_data_buffer.uncompressed_size +
                              _offset_buffer.uncompressed_size +
                              this->_null_mask_buffer.uncompressed_size);
}

template <bool large_string_mode, class Allocator>
bool string_compressed_sliced_column_storage<large_string_mode, Allocator>::is_compressed() const
{
  return this->_data_buffer.compression_format != gqe::compression_format::none ||
         _offset_buffer.compression_format != gqe::compression_format::none;
}

template <bool large_string_mode, class Allocator>
column_compression_statistics
string_compressed_sliced_column_storage<large_string_mode, Allocator>::get_compression_stats() const
{
  string_compression_statistics string_stats;
  auto const chars_compressed_size     = this->_data_buffer.compressed_size;
  auto const chars_uncompressed_size   = this->_data_buffer.uncompressed_size;
  auto const offsets_compressed_size   = _offset_buffer.compressed_size;
  auto const offsets_uncompressed_size = _offset_buffer.uncompressed_size;

  // Offsets buffer statistics
  string_stats.offsets_stats.compressed_size   = static_cast<int64_t>(offsets_compressed_size);
  string_stats.offsets_stats.uncompressed_size = static_cast<int64_t>(offsets_uncompressed_size);
  string_stats.offsets_stats.primary_compressed_size =
    static_cast<int64_t>(_offset_buffer.primary_compressed_size);
  string_stats.offsets_stats.secondary_compressed_size =
    static_cast<int64_t>(_offset_buffer.secondary_compressed_size);
  auto const primary_format    = this->_compression_config.primary_compression_format;
  auto const secondary_format  = this->_compression_config.secondary_compression_format;
  auto const secondary_enabled = secondary_format != gqe::compression_format::none;
  const bool offsets_are_compressed =
    _offset_buffer.compression_format != gqe::compression_format::none;
  string_stats.offsets_stats.num_compressed_row_groups = offsets_are_compressed ? 1ul : 0;
  string_stats.offsets_stats.num_primary_compressed_row_groups =
    (offsets_are_compressed && _offset_buffer.compression_format == primary_format) ? 1ul : 0;
  string_stats.offsets_stats.num_secondary_compressed_row_groups =
    (offsets_are_compressed && secondary_enabled &&
     _offset_buffer.compression_format == secondary_format)
      ? 1ul
      : 0;

  // Chars buffer statistics
  string_stats.chars_stats.compressed_size   = static_cast<int64_t>(chars_compressed_size);
  string_stats.chars_stats.uncompressed_size = static_cast<int64_t>(chars_uncompressed_size);
  string_stats.chars_stats.primary_compressed_size =
    static_cast<int64_t>(this->_data_buffer.primary_compressed_size);
  string_stats.chars_stats.secondary_compressed_size =
    static_cast<int64_t>(this->_data_buffer.secondary_compressed_size);
  const bool chars_are_compressed =
    this->_data_buffer.compression_format != gqe::compression_format::none;
  string_stats.chars_stats.num_compressed_row_groups = chars_are_compressed ? 1ul : 0;
  string_stats.chars_stats.num_primary_compressed_row_groups =
    (chars_are_compressed && this->_data_buffer.compression_format == primary_format) ? 1ul : 0;
  string_stats.chars_stats.num_secondary_compressed_row_groups =
    (chars_are_compressed && secondary_enabled &&
     this->_data_buffer.compression_format == secondary_format)
      ? 1ul
      : 0;

  return column_compression_statistics(string_stats);
}

template <bool large_string_mode, class Allocator>
void string_compressed_sliced_column_storage<large_string_mode, Allocator>::deallocate(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  // TODO: Fix system_memory_resource, pinned_memory_resource, numa_memory_resource, and
  // shared_memory_resource, which immediately free host/shared memory while ignoring
  // the stream. Remove this once they satisfy RMM stream-ordered semantics.
  stream.sync();
  this->_data_buffer.deallocate(stream, mr);
  this->_null_mask_buffer.deallocate(stream, mr);
  _offset_buffer.deallocate(stream, mr);
}

// ---------------------------------------------------------------------------
// Serialization (writes the partition buffers + metadata to a single .bin file).
//
// Binary layout (native byte order):
//   uint32  magic (utility::serialized_column_magic, "GQE")
//   uint64  metadata length
//   bytes   gqe.proto.ColumnMetadata (see proto/gqe_in_memory_tables.proto for every field)
//   bytes   concatenated data/chars, then null-mask, then (string) offset partition payloads
//
// The payloads are appended raw rather than embedded in the message because a column can exceed
// protobuf's 2 GB limit. Their per-partition byte sizes live in the message.
// ---------------------------------------------------------------------------
template <class Allocator>
void compressed_sliced_column_storage<Allocator>::serialize_impl(
  std::string const& column_name,
  std::filesystem::path const& file_path,
  cuda::stream_ref stream,
  buffer_storage const* offset_buffer,
  std::span<cudf::size_type const> partition_char_array_sizes,
  bool is_large_string) const
{
  bool const is_string        = _cudf_type.id() == cudf::type_id::STRING;
  size_t const num_partitions = _data_buffer.buffers.size();
  bool const has_null_mask    = !_null_mask_buffer.buffers.empty();

  // Gather per-partition byte sizes and null counts.
  std::vector<std::int32_t> data_sizes(num_partitions, 0);
  std::vector<std::int32_t> null_sizes(num_partitions, 0);
  std::vector<std::int32_t> null_counts(num_partitions, 0);
  std::vector<std::int32_t> offset_sizes(is_string ? num_partitions : 0, 0);
  std::vector<std::int32_t> char_array_sizes(is_string ? num_partitions : 0, 0);
  for (size_t i = 0; i < num_partitions; ++i) {
    data_sizes[i] = static_cast<std::int32_t>(_data_buffer.buffers[i].size());
    if (has_null_mask) {
      null_sizes[i] = static_cast<std::int32_t>(_null_mask_buffer.buffers[i].size());
    }
    if (i < _null_counts.size()) { null_counts[i] = static_cast<std::int32_t>(_null_counts[i]); }
    if (is_string) {
      if (offset_buffer != nullptr && i < offset_buffer->buffers.size()) {
        offset_sizes[i] = static_cast<std::int32_t>(offset_buffer->buffers[i].size());
      }
      if (i < partition_char_array_sizes.size()) {
        char_array_sizes[i] = static_cast<std::int32_t>(partition_char_array_sizes[i]);
      }
    }
  }

  // Each buffer records the codec it actually holds. String offset buffers are compressed
  // independently from the characters and may select a different codec, so they record their own.
  auto const offsets_format = (is_string && offset_buffer != nullptr)
                                ? offset_buffer->compression_format
                                : gqe::compression_format::none;

  std::filesystem::create_directories(file_path.parent_path());
  std::ofstream file(file_path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("serialize_to_disk: cannot open '" + file_path.string() +
                             "' for write");
  }

  namespace u = gqe::utility;
  gqe::proto::ColumnMetadata metadata;

  auto& layout           = *metadata.mutable_layout();
  *layout.mutable_type() = gqe::rpc::serialize_data_type(_cudf_type);
  layout.set_num_rows(static_cast<std::int64_t>(_size));
  layout.set_partition_size(static_cast<std::uint64_t>(_partition_size));

  auto& data = *metadata.mutable_data();
  data.set_compression_format(gqe::to_string(_data_buffer.compression_format));
  data.mutable_partition_sizes()->Assign(data_sizes.begin(), data_sizes.end());

  // The null-mask and string submessages are written only when they apply, so their absence is
  // what tells the reader the column has no mask or is not a string.
  if (has_null_mask) {
    auto& null_mask = *metadata.mutable_null_mask();
    null_mask.set_compression_format(gqe::to_string(_null_mask_buffer.compression_format));
    null_mask.mutable_partition_sizes()->Assign(null_sizes.begin(), null_sizes.end());
    null_mask.mutable_null_counts()->Assign(null_counts.begin(), null_counts.end());
  }

  if (is_string) {
    auto& strings = *metadata.mutable_strings();
    strings.set_large_offsets(is_large_string);
    strings.set_offsets_compression_format(gqe::to_string(offsets_format));
    strings.mutable_offset_partition_sizes()->Assign(offset_sizes.begin(), offset_sizes.end());
    strings.mutable_char_array_sizes()->Assign(char_array_sizes.begin(), char_array_sizes.end());
  }
  u::write_serialized_column_metadata(file, metadata);

  // Ensure any pending device work on this column's buffers has completed before copying to host.
  GQE_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto const write_partition_payload = [&](gqe::storage::buffer const& buffer,
                                           std::int32_t nbytes) {
    if (nbytes <= 0) { return; }
    std::vector<char> host(static_cast<size_t>(nbytes));
    // The buffer is backed by the table's memory kind, which may be host (numa, pinned, system) or
    // device memory. Use cudaMemcpyDefault and let the runtime infer the direction from the
    // pointer.
    GQE_CUDA_TRY(cudaMemcpyAsync(
      host.data(), buffer.data(), static_cast<size_t>(nbytes), cudaMemcpyDefault, stream.get()));
    GQE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    file.write(host.data(), static_cast<std::streamsize>(nbytes));
  };

  for (size_t i = 0; i < num_partitions; ++i) {
    write_partition_payload(_data_buffer.buffers[i], data_sizes[i]);
  }
  if (has_null_mask) {
    for (size_t i = 0; i < num_partitions; ++i) {
      write_partition_payload(_null_mask_buffer.buffers[i], null_sizes[i]);
    }
  }
  if (is_string && offset_buffer != nullptr) {
    for (size_t i = 0; i < num_partitions; ++i) {
      write_partition_payload(offset_buffer->buffers[i], offset_sizes[i]);
    }
  }

  // close() flushes buffered data. Check afterwards for deferred write errors such as disk full.
  file.close();
  if (file.fail()) {
    throw std::runtime_error("serialize_to_disk: write/close error for '" + file_path.string() +
                             "'");
  }
}

template <class Allocator>
void compressed_sliced_column_storage<Allocator>::serialize_to_disk(
  std::string const& column_name,
  std::filesystem::path const& file_path,
  cuda::stream_ref stream) const
{
  if (_cudf_type.id() == cudf::type_id::STRING) {
    throw std::logic_error(
      "serialize_to_disk: STRING column must be serialized through string storage (column '" +
      column_name + "')");
  }
  serialize_impl(column_name,
                 file_path,
                 stream,
                 /*offset_buffer=*/nullptr,
                 /*partition_char_array_sizes=*/{},
                 /*is_large_string=*/false);
}

template <bool large_string_mode, class Allocator>
void string_compressed_sliced_column_storage<large_string_mode, Allocator>::serialize_to_disk(
  std::string const& column_name,
  std::filesystem::path const& file_path,
  cuda::stream_ref stream) const
{
  this->serialize_impl(column_name,
                       file_path,
                       stream,
                       &_offset_buffer,
                       std::span<cudf::size_type const>{_partition_char_array_sizes.data(),
                                                        _partition_char_array_sizes.size()},
                       is_large_string());
}

// ---------------------------------------------------------------------------
// Deserialization support.
// ---------------------------------------------------------------------------

// Reconstruct storage from already-loaded compressed partition buffers. The raw device buffers are
// copied into the allocator-aware partition storage. Compression metadata is derived from the flags
// and config.
template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  size_t size,
  size_t partition_size,
  cudf::data_type cudf_type,
  gqe::compression::configuration compression_config,
  gqe::compression_format data_format,
  gqe::compression_format null_mask_format,
  std::vector<cudf::size_type>&& null_counts,
  std::vector<rmm::device_buffer>&& compressed_data_buffers,
  std::vector<rmm::device_buffer>&& compressed_null_masks,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  allocator_type const& alloc)
  : compressed_sliced_column_storage(
      size, partition_size, cudf_type, std::move(compression_config), std::move(null_counts), alloc)
{
  auto const sum_buffer_bytes = [](auto const& buffers) {
    return std::accumulate(
      buffers.begin(), buffers.end(), size_t{0}, [](size_t total, auto const& b) {
        return total + b.size();
      });
  };

  // Data buffer: copy the raw device buffers into allocator-aware partition storage. Each copy is
  // async on `stream`, so synchronize before releasing the staging buffer -- the pinned/NUMA
  // resource frees immediately (ignoring the stream). Freeing each partition as it is copied caps
  // the transient footprint at ~one partition instead of holding an entire extra column copy.
  _data_buffer.buffers.reserve(compressed_data_buffers.size());
  for (auto& dev : compressed_data_buffers) {
    _data_buffer.buffers.emplace_back(dev, stream, mr);
    stream.sync();
    dev = rmm::device_buffer{};
  }
  _data_buffer.compression_format = data_format;
  _data_buffer.compressed_size    = sum_buffer_bytes(_data_buffer.buffers);
  _data_buffer.uncompressed_size  = uncompressed_data_bytes(cudf_type, static_cast<int64_t>(size));
  attribute_compressed_size(_data_buffer, _compression_config);

  // Null-mask buffer (same eager-free staging strategy as the data buffer above).
  _null_mask_buffer.buffers.reserve(compressed_null_masks.size());
  for (auto& dev : compressed_null_masks) {
    _null_mask_buffer.buffers.emplace_back(dev, stream, mr);
    stream.sync();
    dev = rmm::device_buffer{};
  }
  _null_mask_buffer.compression_format = null_mask_format;
  _null_mask_buffer.compressed_size    = sum_buffer_bytes(_null_mask_buffer.buffers);
  _null_mask_buffer.uncompressed_size =
    _null_mask_buffer.buffers.empty() ? 0 : null_mask_uncompressed_bytes(size, partition_size);
  attribute_compressed_size(_null_mask_buffer, _compression_config);
}

template <class Allocator>
rmm::device_buffer compressed_sliced_column_storage<Allocator>::decompress_buffer_storage(
  typename compressed_sliced_column_storage<Allocator>::buffer_storage const& source,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  rmm::device_buffer out(source.uncompressed_size, stream, mr);
  if (source.uncompressed_size == 0 || source.buffers.empty()) { return out; }

  if (source.compression_format == gqe::compression_format::none) {
    // Stored uncompressed: concatenate partition bytes in order to rebuild the contiguous layout.
    size_t offset = 0;
    for (auto const& buffer : source.buffers) {
      if (buffer.empty()) { continue; }
      GQE_CUDA_TRY(cudaMemcpyAsync(static_cast<std::byte*>(out.data()) + offset,
                                   buffer.data(),
                                   buffer.size(),
                                   cudaMemcpyDeviceToDevice,
                                   stream.get()));
      offset += buffer.size();
    }
    return out;
  }

  // Stored compressed: decompress every partition into the contiguous output buffer.
  size_t const num_partitions = source.buffers.size();
  std::vector<uint8_t const*> compressed_ptrs(num_partitions);
  for (size_t ix = 0; ix < num_partitions; ++ix) {
    compressed_ptrs[ix] = reinterpret_cast<uint8_t const*>(source.buffers[ix].data());
  }
  auto const data_type = gqe::compression::get_optimal_nvcomp_data_type(source.element_type.id());
  gqe::compression::decompress_batch(_compression_config,
                                     data_type,
                                     reinterpret_cast<uint8_t*>(out.data()),
                                     compressed_ptrs.data(),
                                     compressed_ptrs.data(),
                                     num_partitions,
                                     source.compression_format,
                                     stream,
                                     mr);
  return out;
}

// Scalar-only reconstruct ctor (used by the string storage reconstruct path). Initializes the base
// members and leaves the partition buffers empty for the derived class to populate.
template <class Allocator>
compressed_sliced_column_storage<Allocator>::compressed_sliced_column_storage(
  size_t size,
  size_t partition_size,
  cudf::data_type cudf_type,
  gqe::compression::configuration compression_config,
  std::vector<cudf::size_type>&& null_counts,
  allocator_type const& alloc)
  : _cudf_type(cudf_type),
    _null_counts(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc)),
    _compression_config(std::move(compression_config)),
    _data_buffer(cudf_type, alloc),
    _null_mask_buffer(cudf::data_type(cudf::type_id::INT32), alloc)
{
  if (partition_size == 0) {
    throw std::invalid_argument("compressed_sliced_column deserialize: partition_size is zero");
  }
  _size           = size;
  _partition_size = partition_size;
  _null_counts.assign(null_counts.begin(), null_counts.end());
}

// Reconstruct string storage from already-loaded compressed partition buffers.
template <bool large_string_mode, class Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>::
  string_compressed_sliced_column_storage(size_t size,
                                          size_t partition_size,
                                          gqe::compression::configuration compression_config,
                                          gqe::compression_format chars_format,
                                          gqe::compression_format offsets_format,
                                          gqe::compression_format null_mask_format,
                                          std::vector<cudf::size_type>&& null_counts,
                                          std::vector<cudf::size_type>&& partition_char_array_sizes,
                                          std::vector<rmm::device_buffer>&& chars_buffers,
                                          std::vector<rmm::device_buffer>&& offset_buffers,
                                          std::vector<rmm::device_buffer>&& null_mask_buffers,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          allocator_type const& alloc)
  : base_type(size,
              partition_size,
              cudf::data_type{cudf::type_id::STRING},
              std::move(compression_config),
              std::move(null_counts),
              alloc),
    _offset_buffer(offset_element_type, alloc),
    _partition_char_array_sizes(gqe::storage_rebind_allocator_t<Allocator, cudf::size_type>(alloc))
{
  _partition_char_array_sizes.assign(partition_char_array_sizes.begin(),
                                     partition_char_array_sizes.end());

  // Copy raw device buffers into an allocator-aware buffer_storage and tag its compression
  // metadata.
  auto const populate = [&](buffer_storage& buf,
                            std::vector<rmm::device_buffer>& src,
                            gqe::compression_format format,
                            size_t uncompressed_size) {
    // Eager-free staging (see the fixed-width reconstruct ctor): free each partition as it is
    // copied to cap the transient footprint at ~one partition.
    buf.buffers.reserve(src.size());
    for (auto& dev : src) {
      buf.buffers.emplace_back(dev, stream, mr);
      stream.sync();
      dev = rmm::device_buffer{};
    }
    buf.compression_format = format;
    buf.compressed_size    = std::accumulate(
      buf.buffers.begin(), buf.buffers.end(), size_t{0}, [](size_t t, auto const& b) {
        return t + b.size();
      });
    buf.uncompressed_size = uncompressed_size;
    attribute_compressed_size(buf, this->_compression_config);
  };

  // Characters (data) buffer: uncompressed size is the sum of per-partition char-array sizes.
  size_t const chars_uncompressed =
    std::accumulate(_partition_char_array_sizes.begin(),
                    _partition_char_array_sizes.end(),
                    size_t{0},
                    [](size_t t, cudf::size_type v) { return t + static_cast<size_t>(v); });
  populate(this->_data_buffer, chars_buffers, chars_format, chars_uncompressed);

  // Offsets buffer: one offset per row.
  populate(_offset_buffer, offset_buffers, offsets_format, size * sizeof(offsets_type));

  size_t const null_mask_uncompressed =
    null_mask_buffers.empty() ? 0 : null_mask_uncompressed_bytes(size, partition_size);
  populate(this->_null_mask_buffer, null_mask_buffers, null_mask_format, null_mask_uncompressed);
}

template <class Allocator>
typename compressed_sliced_column_storage<Allocator>::recompressed_null_mask
compressed_sliced_column_storage<Allocator>::recompress_null_mask(
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  recompressed_null_mask out;
  if (_null_mask_buffer.buffers.empty()) { return out; }

  size_t const num_rows       = _size;
  size_t const num_partitions = gqe::utility::divide_round_up(num_rows, _partition_size);

  rmm::device_buffer null_raw = decompress_buffer_storage(_null_mask_buffer, stream, mr);
  std::vector<size_t> null_partition_sizes(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    size_t const row_start  = i * _partition_size;
    size_t const row_count  = std::min(num_rows - row_start, _partition_size);
    null_partition_sizes[i] = gqe::utility::divide_round_up(row_count, 32) * 4;
  }
  out.buffers = compress_contiguous_partitions(target_config,
                                               cudf::data_type(cudf::type_id::INT32),
                                               null_raw,
                                               null_partition_sizes,
                                               memory_kind,
                                               out.format,
                                               stream,
                                               mr);
  return out;
}

// Decompress this column's buffers and re-compress them with target_config, returning the resulting
// per-partition device buffers and selected formats. This is allocator-independent and shared by
// recompress() and recompress_into_segment().
template <class Allocator>
typename compressed_sliced_column_storage<Allocator>::recompressed_buffers
compressed_sliced_column_storage<Allocator>::recompute_buffers(
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  size_t const num_rows       = _size;
  size_t const num_partitions = gqe::utility::divide_round_up(num_rows, _partition_size);

  recompressed_buffers out;

  // Data buffer: decompress to a contiguous buffer, then re-partition and compress.
  rmm::device_buffer data_raw = decompress_buffer_storage(_data_buffer, stream, mr);
  std::vector<size_t> data_partition_sizes(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    size_t const row_start  = i * _partition_size;
    size_t const row_count  = std::min(num_rows - row_start, _partition_size);
    data_partition_sizes[i] = row_count * cudf::size_of(_cudf_type);
  }
  out.data_buffers = compress_contiguous_partitions(target_config,
                                                    _cudf_type,
                                                    data_raw,
                                                    data_partition_sizes,
                                                    memory_kind,
                                                    out.data_format,
                                                    stream,
                                                    mr);

  auto null_mask        = recompress_null_mask(target_config, memory_kind, stream, mr);
  out.null_mask_buffers = std::move(null_mask.buffers);
  out.null_format       = null_mask.format;

  out.null_counts.assign(_null_counts.begin(), _null_counts.end());
  stream.sync();
  return out;
}

template <class Allocator>
compressed_sliced_column_storage<Allocator> compressed_sliced_column_storage<Allocator>::recompress(
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto buffers = recompute_buffers(target_config, memory_kind, stream, mr);
  // Reuse this storage's allocator so the result matches its memory kind. Secondary compression is
  // never applied here, so the flag is false.
  allocator_type const alloc(_null_counts.get_allocator());
  return compressed_sliced_column_storage<Allocator>(_size,
                                                     _partition_size,
                                                     _cudf_type,
                                                     target_config,
                                                     buffers.data_format,
                                                     buffers.null_format,
                                                     std::move(buffers.null_counts),
                                                     std::move(buffers.data_buffers),
                                                     std::move(buffers.null_mask_buffers),
                                                     stream,
                                                     mr,
                                                     alloc);
}

template <class Allocator>
compressed_sliced_column_storage<Allocator>*
compressed_sliced_column_storage<Allocator>::recompress_into_segment(
  shared_segment& segment,
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto buffers = recompute_buffers(target_config, memory_kind, stream, mr);
  allocator_type const alloc(_null_counts.get_allocator());
  return segment.construct<compressed_sliced_column_storage<Allocator>>(
    boost::interprocess::anonymous_instance)(_size,
                                             _partition_size,
                                             _cudf_type,
                                             target_config,
                                             buffers.data_format,
                                             buffers.null_format,
                                             std::move(buffers.null_counts),
                                             std::move(buffers.data_buffers),
                                             std::move(buffers.null_mask_buffers),
                                             stream,
                                             mr,
                                             alloc);
}

// Decompress this string column's character, offset, and null-mask buffers and re-compress them
// with target_config, returning the resulting per-partition device buffers and formats. Shared by
// recompress() and recompress_into_segment().
template <bool large_string_mode, class Allocator>
typename string_compressed_sliced_column_storage<large_string_mode, Allocator>::recompressed_buffers
string_compressed_sliced_column_storage<large_string_mode, Allocator>::recompute_buffers(
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  size_t const num_rows       = this->_size;
  size_t const num_partitions = gqe::utility::divide_round_up(num_rows, this->_partition_size);

  recompressed_buffers out;

  // Characters: partition byte sizes are the per-partition uncompressed char-array sizes.
  rmm::device_buffer chars_raw = this->decompress_buffer_storage(this->_data_buffer, stream, mr);
  std::vector<size_t> chars_partition_sizes(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    chars_partition_sizes[i] = static_cast<size_t>(_partition_char_array_sizes[i]);
  }
  out.chars_buffers = compress_contiguous_partitions(target_config,
                                                     this->_data_buffer.element_type,
                                                     chars_raw,
                                                     chars_partition_sizes,
                                                     memory_kind,
                                                     out.chars_format,
                                                     stream,
                                                     mr);

  // Offsets: one offset per row in each partition.
  rmm::device_buffer offsets_raw = this->decompress_buffer_storage(_offset_buffer, stream, mr);
  std::vector<size_t> offset_partition_sizes(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    size_t const row_start    = i * this->_partition_size;
    size_t const row_count    = std::min(num_rows - row_start, this->_partition_size);
    offset_partition_sizes[i] = row_count * sizeof(offsets_type);
  }
  out.offset_buffers = compress_contiguous_partitions(target_config,
                                                      offset_element_type,
                                                      offsets_raw,
                                                      offset_partition_sizes,
                                                      memory_kind,
                                                      out.offsets_format,
                                                      stream,
                                                      mr);

  auto null_mask        = this->recompress_null_mask(target_config, memory_kind, stream, mr);
  out.null_mask_buffers = std::move(null_mask.buffers);
  out.null_format       = null_mask.format;

  out.null_counts.assign(this->_null_counts.begin(), this->_null_counts.end());
  out.char_array_sizes.assign(_partition_char_array_sizes.begin(),
                              _partition_char_array_sizes.end());
  stream.sync();
  return out;
}

template <bool large_string_mode, class Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>::recompress(
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto buffers = recompute_buffers(target_config, memory_kind, stream, mr);
  // Reuse this storage's allocator so the result matches its memory kind. Secondary compression is
  // never applied here, so all is_*_secondary flags are false.
  allocator_type const alloc(this->_null_counts.get_allocator());
  return string_compressed_sliced_column_storage<large_string_mode, Allocator>(
    this->_size,
    this->_partition_size,
    target_config,
    buffers.chars_format,
    buffers.offsets_format,
    buffers.null_format,
    std::move(buffers.null_counts),
    std::move(buffers.char_array_sizes),
    std::move(buffers.chars_buffers),
    std::move(buffers.offset_buffers),
    std::move(buffers.null_mask_buffers),
    stream,
    mr,
    alloc);
}

template <bool large_string_mode, class Allocator>
string_compressed_sliced_column_storage<large_string_mode, Allocator>*
string_compressed_sliced_column_storage<large_string_mode, Allocator>::recompress_into_segment(
  shared_segment& segment,
  gqe::compression::configuration const& target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto buffers = recompute_buffers(target_config, memory_kind, stream, mr);
  allocator_type const alloc(this->_null_counts.get_allocator());
  return segment.construct<string_compressed_sliced_column_storage<large_string_mode, Allocator>>(
    boost::interprocess::anonymous_instance)(this->_size,
                                             this->_partition_size,
                                             target_config,
                                             buffers.chars_format,
                                             buffers.offsets_format,
                                             buffers.null_format,
                                             std::move(buffers.null_counts),
                                             std::move(buffers.char_array_sizes),
                                             std::move(buffers.chars_buffers),
                                             std::move(buffers.offset_buffers),
                                             std::move(buffers.null_mask_buffers),
                                             stream,
                                             mr,
                                             alloc);
}

template class compressed_sliced_column_storage<std::allocator<std::byte>>;
template class compressed_sliced_column_storage<shared_memory_allocator>;
template class string_compressed_sliced_column_storage<false, std::allocator<std::byte>>;
template class string_compressed_sliced_column_storage<true, std::allocator<std::byte>>;
template class string_compressed_sliced_column_storage<false, shared_memory_allocator>;
template class string_compressed_sliced_column_storage<true, shared_memory_allocator>;

namespace {

// Reconstruct-constructor arguments read from a serialized column file, independent of where the
// resulting storage is built (process-local heap or the shared memory segment). The device
// buffers are allocated through the caller's `mr`, so passing a segment-backed resource lands the
// payloads in the segment.
struct parsed_serialized_column {
  cudf::data_type dtype;
  size_t size;
  size_t partition_size;
  bool is_string;
  bool large_string;
  // Codec each buffer actually holds, read from the header rather than inferred from the active
  // configuration.
  gqe::compression_format data_format;
  gqe::compression_format null_mask_format;
  gqe::compression_format offsets_format;             // string only
  std::vector<cudf::size_type> null_counts;           // empty when there is no null mask
  std::vector<cudf::size_type> partition_char_sizes;  // string only
  std::vector<rmm::device_buffer> data_buffers;       // character buffers for string columns
  std::vector<rmm::device_buffer> null_mask_buffers;
  std::vector<rmm::device_buffer> offset_buffers;  // string only
};

// Parse a serialized column file into reconstruct-constructor arguments. Validates the header and
// every length against the untrusted on-disk data, and stages each partition payload into a device
// buffer via `mr`. Shared with both the local and the in-segment reconstruction paths.
parsed_serialized_column parse_serialized_column(std::filesystem::path const& file_path,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  namespace u = gqe::utility;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("deserialize_from_disk: cannot open '" + file_path.string() + "'");
  }

  // The metadata helpers work on a stream or a message and name no file, so add the path here.
  auto const with_path = [&](auto&& read) {
    try {
      return read();
    } catch (std::invalid_argument const& error) {
      throw std::invalid_argument(std::string(error.what()) + " in '" + file_path.string() + "'");
    }
  };

  auto const metadata = with_path([&] { return u::read_serialized_column_metadata(file); });

  auto const& layout = metadata.layout();
  auto const& data   = metadata.data();
  // Reading an absent submessage yields the default instance, whose arrays are empty. That is safe
  // because every use below is guarded by the corresponding presence flag.
  auto const& null_mask = metadata.null_mask();
  auto const& strings   = metadata.strings();

  auto const dtype     = with_path([&] { return gqe::rpc::deserialize_data_type(layout.type()); });
  bool const is_string = dtype.id() == cudf::type_id::STRING;
  bool const has_null_mask           = metadata.has_null_mask();
  std::int64_t const size_in         = layout.num_rows();
  std::uint64_t const partition_size = layout.partition_size();

  // A string column must carry its offset buffer, so an absent submessage is an incomplete header
  // rather than a non-string column.
  if (is_string && !metadata.has_strings()) {
    throw std::invalid_argument(
      "deserialize_from_disk: STRING column is missing its string buffers in '" +
      file_path.string() + "'");
  }

  // An absent submessage means the buffer does not exist, not that it is uncompressed, so only a
  // present submessage carries a codec name to decode.
  auto const read_format = [&](std::string const& name) {
    return with_path([&] { return gqe::from_string<gqe::compression_format>(name); });
  };
  auto const data_format = read_format(data.compression_format());
  auto const null_mask_format =
    has_null_mask ? read_format(null_mask.compression_format()) : gqe::compression_format::none;
  auto const offsets_format =
    is_string ? read_format(strings.offsets_compression_format()) : gqe::compression_format::none;

  if (size_in < 0) {
    throw std::invalid_argument("deserialize_from_disk: negative column size " +
                                std::to_string(size_in) + " in '" + file_path.string() + "'");
  }
  if (partition_size == 0) {
    throw std::invalid_argument("deserialize_from_disk: zero partition_size in '" +
                                file_path.string() + "'");
  }

  auto const num_partitions = static_cast<std::size_t>(data.partition_sizes_size());
  if (auto const expected_num_partitions =
        gqe::utility::divide_round_up(static_cast<std::uint64_t>(size_in), partition_size);
      num_partitions != expected_num_partitions) {
    throw std::invalid_argument(
      "deserialize_from_disk: partition count " + std::to_string(num_partitions) +
      " does not match ceil(size / partition_size) = " + std::to_string(expected_num_partitions) +
      " in '" + file_path.string() + "'");
  }

  // Every per-partition array must carry exactly one entry per partition.
  auto const require_len = [&](google::protobuf::RepeatedField<std::int32_t> const& arr,
                               const char* name) {
    if (static_cast<std::size_t>(arr.size()) != num_partitions) {
      throw std::invalid_argument("deserialize_from_disk: '" + std::string(name) + "' has " +
                                  std::to_string(arr.size()) + " entries, expected " +
                                  std::to_string(num_partitions) + " in '" + file_path.string() +
                                  "'");
    }
  };
  if (has_null_mask) {
    require_len(null_mask.partition_sizes(), "null_mask.partition_sizes");
    require_len(null_mask.null_counts(), "null_mask.null_counts");
  }
  if (is_string) {
    require_len(strings.offset_partition_sizes(), "strings.offset_partition_sizes");
    require_len(strings.char_array_sizes(), "strings.char_array_sizes");
  }

  auto const& data_size_bytes   = data.partition_sizes();
  auto const& null_size_bytes   = null_mask.partition_sizes();
  auto const& null_count_values = null_mask.null_counts();
  auto const& offset_size_bytes = strings.offset_partition_sizes();
  auto const& char_array_sizes  = strings.char_array_sizes();

  // Read one partition payload into a device buffer. Check gcount() before constructing the
  // device buffer so truncated files never result in a DMA of uninitialized host memory.
  auto const read_partition_payload = [&](std::int32_t nbytes) -> rmm::device_buffer {
    if (nbytes <= 0) { return rmm::device_buffer(0, stream, mr); }
    std::vector<char> host(static_cast<size_t>(nbytes));
    file.read(host.data(), static_cast<std::streamsize>(nbytes));
    if (file.gcount() != static_cast<std::streamsize>(nbytes)) {
      throw std::invalid_argument("deserialize_from_disk: truncated partition payload (expected " +
                                  std::to_string(nbytes) + " bytes) in '" + file_path.string() +
                                  "'");
    }
    rmm::device_buffer device(host.data(), static_cast<size_t>(nbytes), stream, mr);
    // The H2D copy above reads from pageable `host` on `stream`. Synchronize before `host` is
    // destroyed so the copy source stays valid.
    GQE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return device;
  };

  std::vector<cudf::size_type> null_counts;
  std::vector<rmm::device_buffer> data_buffers;
  std::vector<rmm::device_buffer> null_mask_buffers;
  data_buffers.reserve(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) {
    data_buffers.push_back(read_partition_payload(data_size_bytes[i]));
  }
  if (has_null_mask) {
    null_counts.resize(num_partitions);
    null_mask_buffers.reserve(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      null_counts[i] = static_cast<cudf::size_type>(null_count_values[i]);
      null_mask_buffers.push_back(read_partition_payload(null_size_bytes[i]));
    }
  }

  // Offset partition payloads follow the chars and null-mask payloads for string columns.
  std::vector<rmm::device_buffer> offset_buffers;
  if (is_string) {
    offset_buffers.reserve(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      offset_buffers.push_back(read_partition_payload(offset_size_bytes[i]));
    }
  }

  if (file.fail()) {
    throw std::runtime_error("deserialize_from_disk: read error for '" + file_path.string() + "'");
  }

  // Ensure host->device copies complete before the buffers are handed to the column.
  GQE_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  std::vector<cudf::size_type> partition_char_sizes;
  if (is_string) {
    partition_char_sizes.resize(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      partition_char_sizes[i] = static_cast<cudf::size_type>(char_array_sizes[i]);
    }
  }

  return parsed_serialized_column{.dtype                = dtype,
                                  .size                 = static_cast<size_t>(size_in),
                                  .partition_size       = static_cast<size_t>(partition_size),
                                  .is_string            = is_string,
                                  .large_string         = strings.large_offsets(),
                                  .data_format          = data_format,
                                  .null_mask_format     = null_mask_format,
                                  .offsets_format       = offsets_format,
                                  .null_counts          = std::move(null_counts),
                                  .partition_char_sizes = std::move(partition_char_sizes),
                                  .data_buffers         = std::move(data_buffers),
                                  .null_mask_buffers    = std::move(null_mask_buffers),
                                  .offset_buffers       = std::move(offset_buffers)};
}

/**
 * @brief Build a process-local string column from a parsed snapshot, consuming @p parsed.
 *
 * @tparam large_string_mode Whether the column uses 64-bit character offsets.
 */
template <bool large_string_mode>
[[nodiscard]] std::unique_ptr<compressed_sliced_column_interface> make_local_string_column(
  parsed_serialized_column&& parsed,
  gqe::compression::configuration const& compression_config,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return std::make_unique<string_compressed_sliced_column<large_string_mode>>(
    parsed.size,
    parsed.partition_size,
    compression_config,
    parsed.data_format,
    parsed.offsets_format,
    parsed.null_mask_format,
    std::move(parsed.null_counts),
    std::move(parsed.partition_char_sizes),
    std::move(parsed.data_buffers),
    std::move(parsed.offset_buffers),
    std::move(parsed.null_mask_buffers),
    stream,
    mr);
}

}  // namespace

std::unique_ptr<compressed_sliced_column_interface> compressed_sliced_column::deserialize_from_disk(
  std::filesystem::path const& file_path,
  std::string const& column_name,
  gqe::compression::configuration const& compression_config,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  memory_kind::type recompression_memory_kind)
{
  parsed_serialized_column parsed = parse_serialized_column(file_path, stream, mr);

  std::unique_ptr<compressed_sliced_column_interface> result;
  if (parsed.is_string) {
    result = parsed.large_string
               ? make_local_string_column<true>(std::move(parsed), compression_config, stream, mr)
               : make_local_string_column<false>(std::move(parsed), compression_config, stream, mr);
  } else {
    result = std::make_unique<compressed_sliced_column>(parsed.size,
                                                        parsed.partition_size,
                                                        parsed.dtype,
                                                        compression_config,
                                                        parsed.data_format,
                                                        parsed.null_mask_format,
                                                        std::move(parsed.null_counts),
                                                        std::move(parsed.data_buffers),
                                                        std::move(parsed.null_mask_buffers),
                                                        stream,
                                                        mr);
  }

  // Secondary-compression on load: if the run enables a secondary codec, re-encode the loaded
  // (primary) column with it and keep whichever representation is smaller (subject to thresholds).
  if (compression_config.secondary_compression_format != gqe::compression_format::none) {
    gqe::compression::configuration secondary_config = compression_config;
    secondary_config.primary_compression_format   = compression_config.secondary_compression_format;
    secondary_config.secondary_compression_format = gqe::compression_format::none;

    auto recompressed =
      result->recompress_buffers(secondary_config, recompression_memory_kind, stream, mr);
    double const primary_ratio   = result->get_compression_ratio();
    double const secondary_ratio = recompressed->get_compression_ratio();
    bool const secondary_passes =
      secondary_ratio > compression_config.secondary_compression_ratio_threshold;
    bool const secondary_better =
      secondary_ratio >
      (primary_ratio * compression_config.secondary_compression_multiplier_threshold);
    bool const keep_secondary = secondary_passes && secondary_better;
    GQE_LOG_TRACE(
      "deserialize recompress: column='{}' primary_ratio={:.4f} secondary_ratio={:.4f} "
      "ratio_threshold={:.4f} multiplier_threshold={:.4f} passes={} better={} -> kept {}",
      column_name,
      primary_ratio,
      secondary_ratio,
      compression_config.secondary_compression_ratio_threshold,
      compression_config.secondary_compression_multiplier_threshold,
      secondary_passes,
      secondary_better,
      keep_secondary ? "secondary" : "primary");
    if (keep_secondary) {
      result->deallocate(stream, mr);
      return recompressed;
    }
    recompressed->deallocate(stream, mr);
  }
  return result;
}

deserialized_column_storage deserialize_column_into_segment(
  shared_segment& segment,
  shared_memory_allocator const& alloc,
  std::filesystem::path const& file_path,
  gqe::compression::configuration const& compression_config,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  parsed_serialized_column parsed = parse_serialized_column(file_path, stream, mr);
  auto const num_rows             = static_cast<int64_t>(parsed.size);

  // This reconstructs the column in its on-disk (primary) codec. Secondary-compression on load for
  // shared tables is applied by the caller (shared_in_memory_table::deserialize_table_from_disk),
  // which recompresses the built segment storage and keeps whichever representation wins.

  // `mr` must be segment-backed so the partition payloads live in the segment.
  if (parsed.is_string) {
    if (parsed.large_string) {
      using storage_t = shared_string_compressed_sliced_column_storage<true>;
      auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
        parsed.size,
        parsed.partition_size,
        compression_config,
        parsed.data_format,
        parsed.offsets_format,
        parsed.null_mask_format,
        std::move(parsed.null_counts),
        std::move(parsed.partition_char_sizes),
        std::move(parsed.data_buffers),
        std::move(parsed.offset_buffers),
        std::move(parsed.null_mask_buffers),
        stream,
        mr,
        alloc);
      return {column_storage_handle(column_layout::large_string_compressed_sliced,
                                    segment.get_handle_from_address(storage)),
              num_rows};
    }
    using storage_t = shared_string_compressed_sliced_column_storage<false>;
    auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
      parsed.size,
      parsed.partition_size,
      compression_config,
      parsed.data_format,
      parsed.offsets_format,
      parsed.null_mask_format,
      std::move(parsed.null_counts),
      std::move(parsed.partition_char_sizes),
      std::move(parsed.data_buffers),
      std::move(parsed.offset_buffers),
      std::move(parsed.null_mask_buffers),
      stream,
      mr,
      alloc);
    return {column_storage_handle(column_layout::string_compressed_sliced,
                                  segment.get_handle_from_address(storage)),
            num_rows};
  }

  using storage_t = shared_compressed_sliced_column_storage;
  auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
    parsed.size,
    parsed.partition_size,
    parsed.dtype,
    compression_config,
    parsed.data_format,
    parsed.null_mask_format,
    std::move(parsed.null_counts),
    std::move(parsed.data_buffers),
    std::move(parsed.null_mask_buffers),
    stream,
    mr,
    alloc);
  return {column_storage_handle(column_layout::compressed_sliced,
                                segment.get_handle_from_address(storage)),
          num_rows};
}

}  // namespace storage
}  // namespace gqe
