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

#include <gqe/executor/aggregate.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/cuda.hpp>

#include <cudf/utilities/pinned_memory.hpp>

namespace gqe {
namespace storage {

void do_decompress_buffers(
  std::byte* target_ptr,
  rmm::device_buffer* compression_buffer,
  const pruning_result_t& pruning_result,
  const compression_manager& nvcomp_manager,
  const std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
  const std::vector<cudf::size_type>& compressed_data_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  const bool is_secondary_compressed,
  cudf::data_type cudf_type)
{
  const std::vector<size_t> partition_idxs = pruning_result.partition_indexes();
  const size_t num_buffers                 = partition_idxs.size();
  std::byte* source_ptr                    = static_cast<std::byte*>(compression_buffer->data());

  auto cudf_pinned_resource     = cudf::get_pinned_memory_resource();
  std::byte** comp_buffers      = static_cast<std::byte**>(cudf_pinned_resource.allocate_async(
    2 * sizeof(std::byte*) * num_buffers, alignof(std::byte*), stream));
  std::byte** host_comp_buffers = comp_buffers + num_buffers;
  GQE_CUDA_TRY(cudaStreamSynchronize(stream));

  GQE_LOG_DEBUG(
    "Filling batched decompression arrays; num_buffers = {}, source_ptr = {}, target_ptr = {}",
    num_buffers,
    (void*)source_ptr,
    (void*)target_ptr);
  size_t copy_idx = 0;
  for (size_t partition_idx : partition_idxs) {
#ifndef NDEBUG
    GQE_LOG_DEBUG("Filling decompression arrays; partition_idx = {}, source_ptr = {}",
                  partition_idx,
                  (void*)source_ptr);
#endif
    host_comp_buffers[copy_idx] =
      static_cast<std::byte*>(compressed_data_buffers[partition_idx]->data());
    comp_buffers[copy_idx] = source_ptr;
    source_ptr += rmm::align_up(compressed_data_sizes[partition_idx], 8);
    ++copy_idx;
  }
  nvcomp_manager.decompress_batch(reinterpret_cast<uint8_t*>(target_ptr),
                                  reinterpret_cast<uint8_t**>(comp_buffers),
                                  reinterpret_cast<uint8_t**>(host_comp_buffers),
                                  num_buffers,
                                  is_secondary_compressed,
                                  cudf_type,
                                  stream,
                                  mr);
  // Release pointer arrays in pinned memory
  cudf_pinned_resource.deallocate_async(comp_buffers, sizeof(std::byte*) * num_buffers * 2, stream);
}

compressed_sliced_column::compressed_sliced_column(
  cudf::column&& cudf_column,
  int partition_size,
  memory_kind::type memory_kind,
  compression_format comp_format,
  compression_format secondary_compression_format,
  decompression_backend decompress_backend,
  int compression_chunk_size,
  double compression_ratio_threshold,
  double secondary_compression_ratio_threshold,
  double secondary_compression_multiplier_threshold,
  bool use_cpu_compression,
  int compression_level,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string column_name,
  cudf::data_type cudf_type)
  : compressed_sliced_column(cudf_column,
                             partition_size,
                             memory_kind,
                             comp_format,
                             secondary_compression_format,
                             decompress_backend,
                             compression_chunk_size,
                             compression_ratio_threshold,
                             secondary_compression_ratio_threshold,
                             secondary_compression_multiplier_threshold,
                             use_cpu_compression,
                             compression_level,
                             stream,
                             mr,
                             column_name,
                             cudf_type)
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

compressed_sliced_column::compressed_sliced_column(
  const cudf::column& cudf_column,
  int partition_size,
  memory_kind::type memory_kind,
  compression_format comp_format,
  compression_format secondary_comp_format,
  decompression_backend decompress_backend,
  int compression_chunk_size,
  double compression_ratio_threshold,
  double secondary_compression_ratio_threshold,
  double secondary_compression_multiplier_threshold,
  bool use_cpu_compression,
  int compression_level,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string column_name,
  cudf::data_type cudf_type)
  : column_base(),
    _compressed_size(0),
    _uncompressed_size(0),
    _null_mask_compressed_size(0),
    _null_mask_uncompressed_size(0),
    _partition_size(partition_size),
    _comp_format(comp_format),
    _is_compressed(false),
    _is_secondary_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(comp_format,
                    secondary_comp_format,
                    decompress_backend,
                    compression_chunk_size,
                    stream,
                    mr,
                    compression_ratio_threshold,
                    secondary_compression_ratio_threshold,
                    secondary_compression_multiplier_threshold,
                    use_cpu_compression,
                    compression_level,
                    column_name,
                    cudf_type),
    _nvcomp_null_manager(comp_format,
                         gqe::compression_format::none,
                         decompress_backend,
                         compression_chunk_size,
                         stream,
                         mr,
                         compression_ratio_threshold,
                         secondary_compression_ratio_threshold,
                         secondary_compression_multiplier_threshold,
                         use_cpu_compression,
                         compression_level,
                         column_name,
                         cudf_type),
    _secondary_compression_format(secondary_comp_format),
    _secondary_compression_ratio_threshold(secondary_compression_ratio_threshold),
    _secondary_compression_multiplier_threshold(secondary_compression_multiplier_threshold),
    _primary_compressed_size(0),
    _secondary_compressed_size(0),
    _null_mask_primary_compressed_size(0),
    _null_mask_secondary_compressed_size(0)
{
  _size       = cudf_column.size();
  _cudf_type  = cudf_column.type();
  _null_count = cudf_column.null_count();
}

size_t compressed_sliced_column::do_fill_memcpy_buffers(
  std::byte**& dst_ptrs,
  std::byte**& src_ptrs,
  size_t*& sizes,
  compression_buffer_map& compression_buffers,
  const pruning_result_t& pruning_result,
  std::byte* target_ptr,
  bool is_compressed,
  const std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data,
  const std::vector<cudf::size_type>& partition_sizes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto partition_idxs = pruning_result.partition_indexes();
  if (not is_compressed) {
    GQE_LOG_DEBUG("Copying into column buffers; target_ptr = {}", (void*)target_ptr);
  } else {
    size_t buffer_size = 0;
    for (auto partition_idx : partition_idxs) {
      buffer_size += rmm::align_up(partition_sizes[partition_idx], 8);
    }
    auto staging_buffer = std::make_unique<rmm::device_buffer>(buffer_size, stream, mr);
    target_ptr          = static_cast<std::byte*>(staging_buffer->data());
    compression_buffers.insert({this, std::move(staging_buffer)});
    GQE_LOG_DEBUG("Copying data to decompression staging buffer; buffer_size = {}, target_ptr = {}",
                  buffer_size,
                  (void*)target_ptr);
  }

  size_t copy_size = 0;
  for (const auto partition_idx : partition_idxs) {
    size_t size_in_bytes  = partition_sizes[partition_idx];
    std::byte* source_ptr = static_cast<std::byte*>(compressed_data[partition_idx]->data());
    *src_ptrs++           = source_ptr;
    *dst_ptrs++           = target_ptr;
    *sizes++              = size_in_bytes;
#ifndef NDEBUG
    GQE_LOG_DEBUG(
      "Filling memcpy arrays; partition_idx = {}, source_ptr = {}, target_ptr = {}, "
      "size_in_bytes = {}",
      partition_idx,
      (void*)source_ptr,
      (void*)target_ptr,
      size_in_bytes);
#endif
    if (is_compressed) { size_in_bytes = rmm::align_up(size_in_bytes, 8); }
    target_ptr += size_in_bytes;
    copy_size += size_in_bytes;
  }
  return copy_size;
}

void compressed_sliced_column::prepare_batched_memcpy(std::byte** dst_ptrs,
                                                      std::byte** src_ptrs,
                                                      size_t* sizes,
                                                      compression_buffer_map& compression_buffers,
                                                      const pruning_result_t& pruning_result,
                                                      std::byte* target_ptr,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) const
{
  do_fill_memcpy_buffers(dst_ptrs,
                         src_ptrs,
                         sizes,
                         compression_buffers,
                         pruning_result,
                         target_ptr,
                         _is_compressed,
                         _compressed_data_buffers,
                         _compressed_data_sizes,
                         stream,
                         mr);
}

bool compressed_sliced_column::decompress(std::byte* target_ptr,
                                          rmm::device_buffer* compression_buffer,
                                          const pruning_result_t& pruning_result,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr) const
{
  if (not _is_compressed) {
    GQE_LOG_DEBUG("Column was not compressed");
    return false;
  }
  if (not compression_buffer) {
    GQE_LOG_DEBUG(
      "_is_compressed = {}, compression_buffer = {}", _is_compressed, (void*)compression_buffer);
    throw std::logic_error("Table was compressed but compression buffer is not valid");
  }
  do_decompress_buffers(target_ptr,
                        compression_buffer,
                        pruning_result,
                        _nvcomp_manager,
                        _compressed_data_buffers,
                        _compressed_data_sizes,
                        stream,
                        mr,
                        _is_secondary_compressed,
                        _cudf_type);
  return true;
}

void compressed_sliced_column::do_compress(
  rmm::device_buffer const* input,
  std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
  std::vector<cudf::size_type>& compressed_sizes,
  size_t num_rows,
  size_t num_partitions,
  bool& is_compressed,
  size_t& compressed_size,
  size_t& uncompressed_size,
  size_t& primary_compressed_size,
  size_t& secondary_compressed_size,
  bool is_null_mask,
  memory_kind::type memory_kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  utility::nvtx_scoped_range compress_batch_range("compress_batch");

  const uint8_t* bytes_buffer = reinterpret_cast<const uint8_t*>(input->data());
  size_t start_ix             = 0;

  std::vector<std::unique_ptr<rmm::device_buffer>> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t row_start_ix = ix_partition * _partition_size;
    size_t row_count    = std::min(num_rows - row_start_ix, _partition_size);
    size_t size         = row_count * cudf::size_of(_cudf_type);
    if (is_null_mask) { size = gqe::utility::divide_round_up(row_count, 32) * 4; }

    device_uncompressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(size, stream, mr));
    input_ptrs.push_back(reinterpret_cast<const void*>(bytes_buffer + start_ix));
    input_sizes.push_back(size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back()->data()));
    start_ix += size;
  }

  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  compression_manager& manager   = is_null_mask ? _nvcomp_null_manager : _nvcomp_manager;
  bool try_secondary_compression = is_null_mask ? false : true;
  compressed_data_buffers = manager.compress_batch(std::move(device_uncompressed_data_buffers),
                                                   is_compressed,
                                                   compressed_size,
                                                   uncompressed_size,
                                                   primary_compressed_size,
                                                   secondary_compressed_size,
                                                   compressed_sizes,
                                                   _cudf_type,
                                                   memory_kind,
                                                   _is_secondary_compressed,
                                                   try_secondary_compression,
                                                   stream,
                                                   mr);
}

void compressed_sliced_column::compress(cudf::column&& cudf_column,
                                        memory_kind::type memory_kind,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  const size_t num_rows       = cudf_column.size();
  const size_t num_partitions = gqe::utility::divide_round_up(num_rows, _partition_size);
  // Setting up offsets for the cudf slice API

  auto column_content = cudf_column.release();
  assert(column_content.children.empty());

  do_compress(column_content.data.get(),
              _compressed_data_buffers,
              _compressed_data_sizes,
              num_rows,
              num_partitions,
              _is_compressed,
              _compressed_size,
              _uncompressed_size,
              _primary_compressed_size,
              _secondary_compressed_size,
              false /*is_null_mask*/,
              memory_kind,
              stream,
              mr);

  if (_null_count > 0) {
    do_compress(column_content.null_mask.get(),
                _compressed_null_masks,
                _compressed_null_mask_sizes,
                num_rows,
                num_partitions,
                _is_null_mask_compressed,
                _null_mask_compressed_size,
                _null_mask_uncompressed_size,
                _null_mask_primary_compressed_size,
                _null_mask_secondary_compressed_size,
                true /*is_null_mask*/,
                memory_kind,
                stream,
                mr);
    rmm::device_buffer null_mask_buffer(
      column_content.null_mask->data(), column_content.null_mask->size(), stream, mr);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      size_t start_ix = ix_partition * _partition_size;
      size_t stop_ix  = std::min(start_ix + _partition_size, num_rows);
      _null_counts.push_back(cudf::null_count(
        reinterpret_cast<const uint32_t*>(null_mask_buffer.data()), start_ix, stop_ix, stream));
    }
  }
}

int64_t compressed_sliced_column::size() const { return _size; }

cudf::size_type compressed_sliced_column::null_count() const { return _null_count; }

bool compressed_sliced_column::is_compressed() const { return _is_compressed; }

int64_t compressed_sliced_column::get_compressed_size() const
{
  return _compressed_size + _null_mask_compressed_size;
}

int64_t compressed_sliced_column::get_uncompressed_size() const
{
  return _uncompressed_size + _null_mask_uncompressed_size;
}

column_compression_statistics compressed_sliced_column::get_compression_stats() const
{
  fixed_width_compression_statistics fixed_width_stats;
  fixed_width_stats.compressed_size   = get_compressed_size();
  fixed_width_stats.uncompressed_size = get_uncompressed_size();

  fixed_width_stats.primary_compressed_size   = _primary_compressed_size;
  fixed_width_stats.secondary_compressed_size = _secondary_compressed_size;
  fixed_width_stats.num_primary_compressed_row_groups =
    _is_compressed && !_is_secondary_compressed ? 1 : 0;
  fixed_width_stats.num_secondary_compressed_row_groups = _is_secondary_compressed ? 1 : 0;

  fixed_width_stats.num_compressed_row_groups = _is_compressed ? 1 : 0;
  return column_compression_statistics(fixed_width_stats);
}

template <bool large_string_mode>
string_compressed_sliced_column<large_string_mode>::string_compressed_sliced_column(
  cudf::column&& cudf_column,
  int partition_size,
  memory_kind::type memory_kind,
  compression_format comp_format,
  compression_format secondary_compression_format,
  decompression_backend decompress_backend,
  int compression_chunk_size,
  double compression_ratio_threshold,
  double secondary_compression_ratio_threshold,
  double secondary_compression_multiplier_threshold,
  bool use_cpu_compression,
  int compression_level,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string column_name)
  : string_compressed_sliced_column_base(cudf_column,
                                         partition_size,
                                         memory_kind,
                                         comp_format,
                                         secondary_compression_format,
                                         decompress_backend,
                                         compression_chunk_size,
                                         compression_ratio_threshold,
                                         secondary_compression_ratio_threshold,
                                         secondary_compression_multiplier_threshold,
                                         use_cpu_compression,
                                         compression_level,
                                         stream,
                                         mr,
                                         column_name,
                                         cudf::data_type(cudf::type_id::STRING)),
    _offsets_are_compressed(false),
    _offsets_compressed_size(0),
    _offsets_uncompressed_size(0)
{
  compress(std::move(cudf_column), memory_kind, stream, mr);
}

template <bool large_string_mode>
bool string_compressed_sliced_column<large_string_mode>::is_large_string() const
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
template <bool large_string_mode>
void string_compressed_sliced_column<large_string_mode>::compress(cudf::column&& cudf_column,
                                                                  memory_kind::type memory_kind,
                                                                  rmm::cuda_stream_view stream,
                                                                  rmm::device_async_resource_ref mr)
{
  const size_t num_rows       = cudf_column.size();
  const size_t num_partitions = gqe::utility::divide_round_up(num_rows, _partition_size);

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

  std::vector<std::unique_ptr<rmm::device_buffer>> device_uncompressed_data_buffers;
  std::vector<const void*> input_ptrs;
  std::vector<size_t> input_sizes;
  std::vector<void*> device_ptrs;

  device_uncompressed_data_buffers.reserve(num_partitions);
  input_ptrs.reserve(num_partitions);
  input_sizes.reserve(num_partitions);
  device_ptrs.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t start_offset        = ix_partition * _partition_size;
    size_t end_offset          = std::min(start_offset + _partition_size, num_rows);
    size_t partition_char_size = offsets_data_host[end_offset] - offsets_data_host[start_offset];
    device_uncompressed_data_buffers.push_back(
      std::make_unique<rmm::device_buffer>(partition_char_size, stream, mr));
    input_ptrs.push_back(
      reinterpret_cast<const void*>(char_data + offsets_data_host[start_offset]));
    input_sizes.push_back(partition_char_size);
    device_ptrs.push_back(reinterpret_cast<void*>(device_uncompressed_data_buffers.back()->data()));
    _partition_char_array_sizes.push_back(partition_char_size);
    _partition_row_counts.push_back(end_offset - start_offset);
  }
  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  _compressed_data_buffers =
    _nvcomp_manager.compress_batch(std::move(device_uncompressed_data_buffers),
                                   _is_compressed,
                                   _compressed_size,
                                   _uncompressed_size,
                                   _primary_compressed_size,
                                   _secondary_compressed_size,
                                   _compressed_data_sizes,
                                   cudf::data_type(cudf::type_id::STRING),
                                   memory_kind,
                                   _is_secondary_compressed,
                                   true /*try secondary compression*/,
                                   stream,
                                   mr);

  // Compress the null mask using the parent class method
  if (_null_count > 0) {
    // Use do_compress here because the null mask is the same as in the normal compressed sliced
    // column Char buffers and offset arrays are set up differently for strings and can't use the
    // shared helper.
    do_compress(column_content.null_mask.get(),
                _compressed_null_masks,
                _compressed_null_mask_sizes,
                num_rows,
                num_partitions,
                _is_null_mask_compressed,
                _null_mask_compressed_size,
                _null_mask_uncompressed_size,
                _null_mask_primary_compressed_size,
                _null_mask_secondary_compressed_size,
                true /*is_null_mask*/,
                memory_kind,
                stream,
                mr);
    for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
      // Get the null count for this partition
      _null_counts.push_back(
        cudf::null_count(reinterpret_cast<const uint32_t*>(column_content.null_mask.get()),
                         ix_partition * _partition_size,
                         (ix_partition + 1) * _partition_size));
    }
  }

  // Adjust the offset array and compress it. We can do the offset adjustment on the CPU because
  // the write task isn't timed.
  input_ptrs.clear();
  input_sizes.clear();
  device_ptrs.clear();
  std::vector<std::unique_ptr<rmm::device_buffer>> device_uncompressed_offsets_buffers;
  device_uncompressed_offsets_buffers.reserve(num_partitions);
  for (size_t ix_partition = 0; ix_partition < num_partitions; ++ix_partition) {
    size_t ix_start       = ix_partition * _partition_size;
    size_t ix_end         = std::min(ix_start + _partition_size, num_rows);
    size_t partition_rows = ix_end - ix_start;
    size_t partition_base = offsets_data_host[ix_start];
    // We'll adjust each offset such that for the partition, the offsets start at zero
    for (size_t ix_offset = ix_start; ix_offset < ix_end; ++ix_offset) {
      offsets_data_host[ix_offset] -= partition_base;
    }
    input_ptrs.push_back(reinterpret_cast<const void*>(&offsets_data_host[ix_start]));
    input_sizes.push_back(partition_rows * sizeof(offsets_type));
    device_uncompressed_offsets_buffers.push_back(
      std::make_unique<rmm::device_buffer>(partition_rows * sizeof(offsets_type), stream, mr));
    device_ptrs.push_back(
      reinterpret_cast<void*>(device_uncompressed_offsets_buffers.back()->data()));
  }
  gqe::utility::do_batched_memcpy((void**)device_ptrs.data(),
                                  (void**)input_ptrs.data(),
                                  input_sizes.data(),
                                  num_partitions,
                                  stream);

  _compressed_offset_partitions =
    _nvcomp_manager.compress_batch(std::move(device_uncompressed_offsets_buffers),
                                   _offsets_are_compressed,
                                   _offsets_compressed_size,
                                   _offsets_uncompressed_size,
                                   _offsets_primary_compressed_size,
                                   _offsets_secondary_compressed_size,
                                   _compressed_offset_sizes,
                                   offset_element_type,
                                   memory_kind,
                                   _offsets_are_secondary_compressed,
                                   true /* try secondary compression */,
                                   stream,
                                   mr);
}

template <bool large_string_mode>
std::pair<size_t, size_t> string_compressed_sliced_column<large_string_mode>::buffer_sizes(
  const pruning_result_t& pruning_result) const
{
  size_t char_buffer_size = 0;
  for (const auto partition_idx : pruning_result.partition_indexes()) {
    char_buffer_size += _partition_char_array_sizes[partition_idx];
  }
  const size_t offset_buffer_size = pruning_result.num_rows() * sizeof(offsets_type);
  return {char_buffer_size, offset_buffer_size};
}

template <bool large_string_mode>
std::tuple<size_t, size_t, size_t>
string_compressed_sliced_column<large_string_mode>::prepare_batched_memcpy(
  std::byte** dst_ptrs,
  std::byte** src_ptrs,
  size_t* sizes,
  compression_buffer_map& char_compression_buffers,
  compression_buffer_map& offset_compression_buffers,
  const pruning_result_t& pruning_result,
  std::byte* char_target_ptr,
  std::byte* offset_target_ptr,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  GQE_LOG_DEBUG("Filling memcpy arrays for string column character buffer");
  size_t char_copy_size = do_fill_memcpy_buffers(dst_ptrs,
                                                 src_ptrs,
                                                 sizes,
                                                 char_compression_buffers,
                                                 pruning_result,
                                                 char_target_ptr,
                                                 _is_compressed,
                                                 _compressed_data_buffers,
                                                 _compressed_data_sizes,
                                                 stream,
                                                 mr);
  GQE_LOG_DEBUG("Filling memcpy arrays for string column offset buffer");
  size_t offset_copy_size = do_fill_memcpy_buffers(dst_ptrs,
                                                   src_ptrs,
                                                   sizes,
                                                   offset_compression_buffers,
                                                   pruning_result,
                                                   offset_target_ptr,
                                                   _offsets_are_compressed,
                                                   _compressed_offset_partitions,
                                                   _compressed_offset_sizes,
                                                   stream,
                                                   mr);
  // Indicate that we need to copy the character buffer and offset buffer for each partition
  size_t num_buffers = 2 * pruning_result.partition_indexes().size();
  return {num_buffers, char_copy_size, offset_copy_size};
}

template <bool large_string_mode>
std::pair<bool, size_t> string_compressed_sliced_column<large_string_mode>::decompress(
  std::byte* char_target_ptr,
  std::byte* offset_target_ptr,
  rmm::device_buffer* char_compression_buffer,
  rmm::device_buffer* offset_compression_buffer,
  const pruning_result_t& pruning_result,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  bool was_compressed           = false;
  size_t char_decompressed_size = 0;

  if (not _is_compressed) {
    GQE_LOG_DEBUG("Char column was not compressed");
  } else if (not char_compression_buffer) {
    GQE_LOG_DEBUG("_is_compressed = {}, char_compression_buffer = {}",
                  _is_compressed,
                  (void*)char_compression_buffer);
    throw std::logic_error("Char column was compressed but compression buffer is not valid");
  } else {
    was_compressed = true;
    do_decompress_buffers(char_target_ptr,
                          char_compression_buffer,
                          pruning_result,
                          _nvcomp_manager,
                          _compressed_data_buffers,
                          _compressed_data_sizes,
                          stream,
                          mr,
                          _is_secondary_compressed,
                          _cudf_type);
    for (auto partition_idx : pruning_result.partition_indexes()) {
      char_decompressed_size += _partition_char_array_sizes[partition_idx];
    }
  }

  if (not _offsets_are_compressed) {
    GQE_LOG_DEBUG("Offset column was not compressed");
  } else if (not offset_compression_buffer) {
    GQE_LOG_DEBUG("_offsets_are_compressed = {}, offset_compression_buffer = {}",
                  _offsets_are_compressed,
                  (void*)offset_compression_buffer);
    throw std::logic_error("Char column was compressed but compression buffer is not valid");
  } else {
    was_compressed = true;
    do_decompress_buffers(offset_target_ptr,
                          offset_compression_buffer,
                          pruning_result,
                          _nvcomp_manager,
                          _compressed_offset_partitions,
                          _compressed_offset_sizes,
                          stream,
                          mr,
                          _offsets_are_secondary_compressed,
                          offset_element_type);
  }

  return {was_compressed, char_decompressed_size};
}

template <bool large_string_mode>
void string_compressed_sliced_column<large_string_mode>::fill_partition_offsets(
  offsets_type* partition_char_offsets,
  cudf::size_type* partition_row_offsets,
  offsets_type& char_offset,
  cudf::size_type& row_offset,
  const pruning_result_t& pruning_result,
  size_t partition_offset_idx) const
{
  auto partition_idxs             = pruning_result.partition_indexes();
  auto chars_partition_offset_idx = partition_offset_idx;
  for (auto partition_idx : partition_idxs) {
#ifndef NDEBUG
    GQE_LOG_DEBUG("partition_idx = {}, char_offset = {}", partition_idx, char_offset);
#endif
    partition_char_offsets[chars_partition_offset_idx] = char_offset;
    char_offset += _partition_char_array_sizes[partition_idx];
    ++chars_partition_offset_idx;
  }

  auto const partition_size     = pruning_result.partition_size();
  auto row_partition_offset_idx = partition_offset_idx;
  for (auto const& consolidated_partition : pruning_result.consolidated_partitions()) {
    auto partition_start = consolidated_partition.start;
    while (partition_start < consolidated_partition.end) {
      auto const partition_end =
        std::min<cudf::size_type>(partition_start + partition_size, consolidated_partition.end);
      auto const row_count = partition_end - partition_start;
#ifndef NDEBUG
      auto const partition_idx = static_cast<size_t>(partition_start / partition_size);
      GQE_LOG_DEBUG("partition_idx = {}, row_offset = {}, char_offset = {}",
                    partition_idx,
                    row_offset,
                    char_offset);
#endif
      partition_row_offsets[row_partition_offset_idx] = row_offset;
      row_offset += row_count;
      ++row_partition_offset_idx;
      partition_start = partition_end;
    }
  }
}

template <bool large_string_mode>
int64_t string_compressed_sliced_column<large_string_mode>::get_compressed_size() const
{
  return _compressed_size + _offsets_compressed_size + _null_mask_compressed_size;
}

template <bool large_string_mode>
int64_t string_compressed_sliced_column<large_string_mode>::get_uncompressed_size() const
{
  return _uncompressed_size + _offsets_uncompressed_size + _null_mask_uncompressed_size;
}

template <bool large_string_mode>
bool string_compressed_sliced_column<large_string_mode>::is_compressed() const
{
  return _is_compressed || _offsets_are_compressed;
}

template <bool large_string_mode>
column_compression_statistics
string_compressed_sliced_column<large_string_mode>::get_compression_stats() const
{
  string_compression_statistics string_stats;

  // Offsets buffer statistics
  string_stats.offsets_stats.compressed_size   = static_cast<int64_t>(_offsets_compressed_size);
  string_stats.offsets_stats.uncompressed_size = static_cast<int64_t>(_offsets_uncompressed_size);
  string_stats.offsets_stats.primary_compressed_size =
    static_cast<int64_t>(_offsets_primary_compressed_size);
  string_stats.offsets_stats.secondary_compressed_size =
    static_cast<int64_t>(_offsets_secondary_compressed_size);
  string_stats.offsets_stats.num_compressed_row_groups = _offsets_are_compressed ? 1ul : 0;
  string_stats.offsets_stats.num_primary_compressed_row_groups =
    _offsets_are_compressed && !_offsets_are_secondary_compressed ? 1ul : 0;
  string_stats.offsets_stats.num_secondary_compressed_row_groups =
    _offsets_are_secondary_compressed ? 1ul : 0;

  // Chars buffer statistics
  string_stats.chars_stats.compressed_size         = static_cast<int64_t>(_compressed_size);
  string_stats.chars_stats.uncompressed_size       = static_cast<int64_t>(_uncompressed_size);
  string_stats.chars_stats.primary_compressed_size = static_cast<int64_t>(_primary_compressed_size);
  string_stats.chars_stats.secondary_compressed_size =
    static_cast<int64_t>(_secondary_compressed_size);
  string_stats.chars_stats.num_compressed_row_groups = _is_compressed ? 1ul : 0;
  string_stats.chars_stats.num_primary_compressed_row_groups =
    _is_compressed && !_is_secondary_compressed ? 1ul : 0;
  string_stats.chars_stats.num_secondary_compressed_row_groups = _is_secondary_compressed ? 1ul : 0;

  return column_compression_statistics(string_stats);
}

template class string_compressed_sliced_column<false>;
template class string_compressed_sliced_column<true>;

}  // namespace storage
}  // namespace gqe
