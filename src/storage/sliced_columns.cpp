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

#include <cudf/utilities/pinned_memory.hpp>

#include <gqe/storage/in_memory.hpp>

namespace gqe {
namespace storage {

compressed_sliced_column::compressed_sliced_column(cudf::column&& cudf_column,
                                                   compression_format comp_format,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   nvcompType_t nvcomp_data_format,
                                                   int chunk_size,
                                                   int partition_size,
                                                   double compression_ratio_threshold,
                                                   std::string column_name,
                                                   cudf::data_type cudf_type)
  : column_base(),
    _partition_size(partition_size),
    _comp_format(comp_format),
    _compression_ratio(0.0),
    _null_mask_compression_ratio(0.0),
    _is_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(comp_format,
                    nvcomp_data_format,
                    chunk_size,
                    stream,
                    mr,
                    compression_ratio_threshold,
                    column_name,
                    cudf_type),
    _nvcomp_null_manager(
      comp_format, NVCOMP_TYPE_CHAR, chunk_size, stream, mr, compression_ratio_threshold)
{
  _size       = cudf_column.size();
  _dtype      = cudf_column.type();
  _null_count = cudf_column.null_count();

  compress(std::move(cudf_column), stream, mr);
}

compressed_sliced_column::compressed_sliced_column(const cudf::column& cudf_column,
                                                   compression_format comp_format,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   nvcompType_t nvcomp_data_format,
                                                   int chunk_size,
                                                   int partition_size,
                                                   double compression_ratio_threshold,
                                                   std::string column_name,
                                                   cudf::data_type cudf_type)
  : column_base(),
    _partition_size(partition_size),
    _comp_format(comp_format),
    _compression_ratio(0.0),
    _null_mask_compression_ratio(0.0),
    _is_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(comp_format,
                    nvcomp_data_format,
                    chunk_size,
                    stream,
                    mr,
                    compression_ratio_threshold,
                    column_name,
                    cudf_type),
    _nvcomp_null_manager(
      comp_format, NVCOMP_TYPE_CHAR, chunk_size, stream, mr, compression_ratio_threshold)

{
  _size       = cudf_column.size();
  _dtype      = cudf_column.type();
  _null_count = cudf_column.null_count();
}

void do_batched_memcpy(
  void** src_ptrs, void** dst_ptrs, size_t* sizes, size_t num_buffers, rmm::cuda_stream_view stream)
{
  assert(num_buffers > 0);
  std::vector<cudaMemcpyAttributes> attrs(1);
  attrs[0].srcAccessOrder       = cudaMemcpySrcAccessOrderStream;
  attrs[0].flags                = 0;
  std::vector<size_t> attrsIdxs = {0};
  size_t numAttrs               = attrs.size();
  // attrs.dstLocHint.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
  // attrs.srcLocHint.type = CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST;
  // attrs.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
  // attrs.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;

  // For this API, the dest and source arrays are both CUDevicePtr. Need to fill these arrays
  // Attrs is a host array
  size_t fail_idx;
  GQE_CUDA_TRY(cudaMemcpyBatchAsync(dst_ptrs,
                                    src_ptrs,
                                    sizes,
                                    num_buffers,
                                    attrs.data(),
                                    attrsIdxs.data(),
                                    numAttrs,
                                    &fail_idx,
                                    stream));
}

void compressed_sliced_column::fill_copy_ptrs(
  uint8_t** host_compressed_ptrs,
  uint8_t** device_compressed_ptrs,
  const std::vector<cudf::size_type>& compressed_sizes,
  const std::vector<size_t>& ix_partition_slices,
  std::vector<nvcomp::DecompressionConfig>& decompression_configs,
  const std::vector<nvcomp::CompressionConfig>& compression_configs,
  std::vector<size_t>& reduced_compressed_sizes,
  std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
  uint8_t* dst_ptr,
  const bool is_compressed,
  rmm::cuda_stream_view stream)
{
  size_t compressed_offset = 0;
  for (size_t ix = 0; ix < ix_partition_slices.size(); ix++) {
    size_t ix_partition          = ix_partition_slices[ix];
    reduced_compressed_sizes[ix] = compressed_sizes[ix_partition];

    host_compressed_ptrs[ix] =
      reinterpret_cast<uint8_t*>(compressed_data_buffers[ix_partition]->data());
    device_compressed_ptrs[ix] = dst_ptr + compressed_offset;

    if (is_compressed) {
      decompression_configs.push_back(
        _nvcomp_manager.configure_decompression(compression_configs[ix_partition]));
    }

    if (is_compressed) {
      // Guarantee 8 byte alignment for the compressed buffers. This is required for the nvcomp
      // batched decompress API.
      compressed_offset += ((compressed_sizes[ix_partition] + 7) / 8) * 8;
    } else {
      // For uncompressed data, no alignment needed - just use the actual size
      compressed_offset += compressed_sizes[ix_partition];
    }
  }
}

// This will decompress either the data or the null mask
rmm::device_buffer compressed_sliced_column::do_decompress(
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  const size_t total_uncompressed_size,
  const size_t total_compressed_size,
  const std::vector<size_t>& ix_partition_slices,
  std::vector<cudf::size_type>& full_compressed_sizes,
  std::vector<std::unique_ptr<rmm::device_buffer>>& full_compressed_data_buffers,
  std::vector<nvcomp::CompressionConfig>& full_compression_configs,
  const bool is_compressed)
{
  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();

  size_t remaining_partitions = ix_partition_slices.size();

  uint8_t** host_compressed_ptrs = reinterpret_cast<uint8_t**>(cudf_pinned_resource.allocate_async(
    sizeof(uint8_t*) * remaining_partitions, alignof(uint8_t*), stream));

  uint8_t** device_compressed_ptrs =
    reinterpret_cast<uint8_t**>(cudf_pinned_resource.allocate_async(
      sizeof(uint8_t*) * remaining_partitions, alignof(uint8_t*), stream));
  GQE_CUDA_TRY(cudaStreamSynchronize(stream));

  std::vector<nvcomp::DecompressionConfig> buffer_decompression_configs;

  std::vector<size_t> compressed_sizes(remaining_partitions);
  rmm::device_buffer decompressed_data(total_uncompressed_size, stream, mr);
  rmm::device_buffer compressed_data_buffer;
  if (is_compressed) {
    /* Each compressed partition buffer needs to be 8-byte aligned.
       Thus the maximum adjustment for each partition is 7 byte.
       Add this maximum adjustment for every partition but the first one (the first is already
       aligned)*/
    compressed_data_buffer =
      rmm::device_buffer(total_compressed_size + 7 * (remaining_partitions - 1), stream, mr);
  }

  uint8_t* memcpy_target_ptr = reinterpret_cast<uint8_t*>(
    is_compressed ? compressed_data_buffer.data() : decompressed_data.data());
  fill_copy_ptrs(host_compressed_ptrs,
                 device_compressed_ptrs,
                 full_compressed_sizes,
                 ix_partition_slices,
                 buffer_decompression_configs,
                 full_compression_configs,
                 compressed_sizes,
                 full_compressed_data_buffers,
                 memcpy_target_ptr,
                 is_compressed,
                 stream);

  do_batched_memcpy(reinterpret_cast<void**>(host_compressed_ptrs),
                    reinterpret_cast<void**>(device_compressed_ptrs),
                    compressed_sizes.data(),
                    remaining_partitions,
                    stream);

  if (is_compressed) {
    uint8_t** device_decompressed_ptrs =
      reinterpret_cast<uint8_t**>(cudf_pinned_resource.allocate_async(
        sizeof(uint8_t*) * remaining_partitions, alignof(uint8_t*), stream));
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
    size_t decompressed_offset = 0;
    for (size_t ix = 0; ix < ix_partition_slices.size(); ix++) {
      size_t ix_partition = ix_partition_slices[ix];
      device_decompressed_ptrs[ix] =
        reinterpret_cast<uint8_t*>(decompressed_data.data()) + decompressed_offset;
      decompressed_offset += full_compression_configs[ix_partition].uncompressed_buffer_size;
    }
    _nvcomp_manager.decompress_batch(device_decompressed_ptrs,
                                     device_compressed_ptrs,
                                     buffer_decompression_configs,
                                     host_compressed_ptrs,
                                     stream,
                                     mr);
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));
    cudf_pinned_resource.deallocate_async(
      device_decompressed_ptrs, sizeof(uint8_t*) * remaining_partitions, stream);
  }

  cudf_pinned_resource.deallocate_async(
    device_compressed_ptrs, sizeof(uint8_t*) * remaining_partitions, stream);
  cudf_pinned_resource.deallocate_async(
    host_compressed_ptrs, sizeof(uint8_t*) * remaining_partitions, stream);

  return decompressed_data;
}

/*
 * @brief Get the indices of the partitions to decompress
 *
 * This method takes consolidated partitions and returns the slices of the partitions to decompress
 *
 * TODO: don't consolidate in the first place, since here we have to undo that work
 *
 * Add the consolidation to the columns that need it.
 */
std::vector<size_t> compressed_sliced_column::get_compressed_slice_indices(
  const std::vector<zone_map::partition>& partitions)
{
  std::vector<size_t> decomp_slice_indices;
  for (size_t ix_partition = 0; ix_partition < partitions.size(); ix_partition++) {
    if (partitions[ix_partition].pruned) { continue; }
    size_t ix_start_partition = partitions[ix_partition].start / _partition_size;

    // Round up, this isn't inclusive.
    size_t ix_end_partition = gqe::utility::divide_round_up(partitions[ix_partition].end,
                                                            _partition_size);  // Non-inclusive
    for (size_t ix_decomp = ix_start_partition; ix_decomp < ix_end_partition; ix_decomp++) {
      decomp_slice_indices.push_back(ix_decomp);
    }
  }
  return decomp_slice_indices;
}

std::unique_ptr<cudf::column> compressed_sliced_column::decompress(
  rmm::cuda_stream_view stream,
  const std::vector<zone_map::partition>& partitions,
  rmm::device_async_resource_ref mr)
{
  // We'll start with just decompressing one buffer at a time
  // Batched mode is easy to migrate to later

  // Want a device decompressed buffer that will fit all row groups
  // We know the partition size up front, so based on the data type
  // we can pre allocate
  // We need to get the decompress indices to use for the merge
  size_t total_rows         = 0;
  size_t reduced_null_count = 0;
  size_t total_null_size    = 0;

  size_t total_compressed_size            = 0;
  size_t total_compressed_null_mask_size  = 0;
  std::vector<size_t> ix_partition_slices = get_compressed_slice_indices(partitions);
  for (const auto& ix_partition : ix_partition_slices) {
    const size_t partition_row_count =
      std::min(_partition_size, _size - ix_partition * _partition_size);
    total_rows += partition_row_count;

    total_compressed_size += _compressed_data_sizes[ix_partition];
  }

  size_t total_uncompressed_size       = total_rows * cudf::size_of(_dtype);
  rmm::device_buffer decompressed_data = do_decompress(stream,
                                                       mr,
                                                       total_uncompressed_size,
                                                       total_compressed_size,
                                                       ix_partition_slices,
                                                       _compressed_data_sizes,
                                                       _compressed_data_buffers,
                                                       _compression_configs,
                                                       _is_compressed);

  rmm::device_buffer decompressed_null_mask;
  if (_null_count > 0) {
    for (const auto& ix_partition : ix_partition_slices) {
      // cudf::bitmask_type is uint32_t, so here we assumed that the row count was a multiple of 32
      // This is erroneous when the row group itself is not a multiple of 32, and you need to
      // concatenate multiple row groups
      const size_t partition_row_count =
        std::min(_partition_size, _size - ix_partition * _partition_size);
      total_null_size += gqe::utility::divide_round_up(partition_row_count, 32) * 4;
      assert(partition_row_count % 32 == 0 or ix_partition == ix_partition_slices.size() - 1);
      reduced_null_count += _null_counts[ix_partition];
      total_compressed_null_mask_size += _compressed_null_mask_sizes[ix_partition];

      total_compressed_size += _compressed_data_sizes[ix_partition];
    }

    decompressed_null_mask = do_decompress(stream,
                                           mr,
                                           total_null_size,
                                           total_compressed_null_mask_size,
                                           ix_partition_slices,
                                           _compressed_null_mask_sizes,
                                           _compressed_null_masks,
                                           _null_compression_configs,
                                           _is_null_mask_compressed);
  }

  return std::make_unique<cudf::column>(_dtype,
                                        total_rows,
                                        std::move(decompressed_data),
                                        std::move(decompressed_null_mask),
                                        reduced_null_count);
}

void compressed_sliced_column::do_compress(
  rmm::device_buffer const* input,
  std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
  std::vector<nvcomp::CompressionConfig>& compression_configs,
  std::vector<cudf::size_type>& compressed_sizes,
  size_t num_rows,
  size_t num_partitions,
  bool& is_compressed,
  size_t& compressed_size,
  bool is_null_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  const uint8_t* bytes_buffer = reinterpret_cast<const uint8_t*>(input->data());
  size_t start_ix             = 0;

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t row_start_ix = ix_partition * _partition_size;
    size_t row_count    = std::min(num_rows - row_start_ix, _partition_size);
    size_t size         = row_count * cudf::size_of(_dtype);
    if (is_null_mask) { size = gqe::utility::divide_round_up(row_count, 32) * 4; }
    device_uncompressed_data_buffers.emplace_back(bytes_buffer + start_ix, size, stream, mr);
    start_ix += size;
  }

  compression_manager& manager = is_null_mask ? _nvcomp_null_manager : _nvcomp_manager;

  std::tie(compressed_data_buffers, compression_configs) =
    manager.compress_batch(device_uncompressed_data_buffers,
                           _compression_ratio,
                           is_compressed,
                           compressed_size,
                           compressed_sizes,
                           stream,
                           mr);
}

void compressed_sliced_column::compress(cudf::column&& cudf_column,
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
              _compression_configs,
              _compressed_data_sizes,
              num_rows,
              num_partitions,
              _is_compressed,
              _compressed_size,
              false /*is_null_mask*/,
              stream,
              mr);

  if (_null_count > 0) {
    do_compress(column_content.null_mask.get(),
                _compressed_null_masks,
                _null_compression_configs,
                _compressed_null_mask_sizes,
                num_rows,
                num_partitions,
                _is_null_mask_compressed,
                _null_mask_compressed_size,
                true /*is_null_mask*/,
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

template <bool large_string_mode>
string_compressed_sliced_column<large_string_mode>::string_compressed_sliced_column(
  cudf::column&& cudf_column,
  compression_format comp_format,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int chunk_size,
  int partition_size,
  double compression_ratio_threshold,
  std::string column_name)
  : compressed_sliced_column(cudf_column,
                             comp_format,
                             stream,
                             mr,
                             NVCOMP_TYPE_CHAR,
                             chunk_size,
                             partition_size,
                             compression_ratio_threshold,
                             column_name,
                             cudf::data_type(cudf::type_id::STRING)),
    _nvcomp_offset_manager(
      comp_format, offset_nvcomp_data_type, chunk_size, stream, mr, compression_ratio_threshold)
{
  compress(std::move(cudf_column), stream, mr);
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

  std::vector<rmm::device_buffer> device_uncompressed_data_buffers;
  for (size_t ix_partition = 0; ix_partition < num_partitions; ix_partition++) {
    size_t start_offset        = ix_partition * _partition_size;
    size_t end_offset          = std::min(start_offset + _partition_size, num_rows);
    size_t partition_char_size = offsets_data_host[end_offset] - offsets_data_host[start_offset];
    device_uncompressed_data_buffers.emplace_back(
      char_data + offsets_data_host[start_offset], partition_char_size, stream, mr);
    _partition_char_array_sizes.push_back(partition_char_size);
    _partition_row_counts.push_back(end_offset - start_offset);
  }
  float compression_ratio;
  std::tie(_compressed_data_buffers, _compression_configs) =
    _nvcomp_manager.compress_batch(device_uncompressed_data_buffers,
                                   compression_ratio,
                                   _is_compressed,
                                   _compressed_size,
                                   _compressed_data_sizes,
                                   stream,
                                   mr);

  // Compress the null mask using the parent class method
  if (_null_count > 0) {
    do_compress(column_content.null_mask.get(),
                _compressed_null_masks,
                _null_compression_configs,
                _compressed_null_mask_sizes,
                num_rows,
                num_partitions,
                _is_null_mask_compressed,
                _null_mask_compressed_size,
                true /*is_null_mask*/,
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
  size_t child_compressed_size;
  std::vector<rmm::device_buffer> device_uncompressed_offsets_buffers;
  for (size_t ix_partition = 0; ix_partition < num_partitions; ++ix_partition) {
    size_t ix_start       = ix_partition * _partition_size;
    size_t ix_end         = std::min(ix_start + _partition_size, num_rows);
    size_t partition_rows = ix_end - ix_start;
    size_t partition_base = offsets_data_host[ix_start];
    // We'll adjust each offset such that for the partition, the offsets start at zero
    for (size_t ix_offset = ix_start; ix_offset < ix_end; ++ix_offset) {
      offsets_data_host[ix_offset] -= partition_base;
    }
    device_uncompressed_offsets_buffers.emplace_back(
      &offsets_data_host[ix_start], partition_rows * sizeof(offsets_type), stream, mr);
  }

  std::tie(_compressed_offset_partitions, _compressed_offset_configs) =
    _nvcomp_offset_manager.compress_batch(device_uncompressed_offsets_buffers,
                                          compression_ratio,
                                          _offsets_are_compressed,
                                          child_compressed_size,
                                          _compressed_offset_sizes,
                                          stream,
                                          mr);
}

// Add the decompress method -- after this we can modify the existing slice methods to use this type
template <bool large_string_mode>
std::unique_ptr<cudf::column> string_compressed_sliced_column<large_string_mode>::decompress(
  rmm::cuda_stream_view stream,
  const std::vector<zone_map::partition>& partitions,
  rmm::device_async_resource_ref mr)
{
  // We'll start with just decompressing one buffer at a time
  // Batched mode is easy to migrate to later

  // We can decompress the nulls just as with the regular compressed sliced column
  auto ix_partition_slices = get_compressed_slice_indices(partitions);
  size_t num_partitions    = ix_partition_slices.size();

  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  void* partition_offsets   = cudf_pinned_resource.allocate_async(
    num_partitions * sizeof(offsets_type), alignof(offsets_type), stream);
  offsets_type* partition_offsets_ptr = reinterpret_cast<offsets_type*>(partition_offsets);
  // We'll decompress the columns directly
  size_t total_rows                      = 0;
  size_t total_null_size                 = 0;
  size_t reduced_null_count              = 0;
  size_t total_char_array_size           = 0;
  size_t total_compressed_size           = 0;
  size_t total_compressed_null_mask_size = 0;
  size_t total_compressed_offset_size    = 0;

  for (size_t ix = 0; ix < num_partitions; ix++) {
    size_t ix_partition = ix_partition_slices[ix];
    total_rows += _partition_row_counts[ix_partition];
    if (_null_count > 0) {
      total_null_size += _partition_row_counts[ix_partition] / 8;
      reduced_null_count += _null_counts[ix_partition];
      total_compressed_null_mask_size += _compressed_null_mask_sizes[ix_partition];
    }

    partition_offsets_ptr[ix] = total_char_array_size;
    total_char_array_size += _partition_char_array_sizes[ix_partition];
    total_compressed_size += _compressed_data_sizes[ix_partition];
    total_compressed_offset_size += _compressed_offset_sizes[ix_partition];
  }

  size_t total_offsets = total_rows + 1;

  rmm::device_buffer decompressed_null_mask;
  if (_null_count > 0) {
    decompressed_null_mask = do_decompress(stream,
                                           mr,
                                           total_null_size,
                                           total_compressed_null_mask_size,
                                           ix_partition_slices,
                                           _compressed_null_mask_sizes,
                                           _compressed_null_masks,
                                           _null_compression_configs,
                                           _is_null_mask_compressed);
  }

  rmm::device_buffer decompressed_char_array = do_decompress(stream,
                                                             mr,
                                                             total_char_array_size,
                                                             total_compressed_size,
                                                             ix_partition_slices,
                                                             _compressed_data_sizes,
                                                             _compressed_data_buffers,
                                                             _compression_configs,
                                                             _is_compressed);

  size_t total_offset_uncompressed_size   = total_offsets * sizeof(offsets_type);
  rmm::device_buffer decompressed_offsets = do_decompress(stream,
                                                          mr,
                                                          total_offset_uncompressed_size,
                                                          total_compressed_offset_size,
                                                          ix_partition_slices,
                                                          _compressed_offset_sizes,
                                                          _compressed_offset_partitions,
                                                          _compressed_offset_configs,
                                                          _offsets_are_compressed);

  GQE_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  adjust_offsets_api(reinterpret_cast<offsets_type*>(decompressed_offsets.data()),
                     total_rows,
                     _partition_size,
                     partition_offsets_ptr,
                     total_char_array_size,
                     stream);
  GQE_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  // Finally we will construct the column to return
  // First build the offsets column
  rmm::device_buffer offsets_null_mask;
  cudf::data_type offsets_dtype = large_string_mode ? cudf::data_type(cudf::type_id::INT64)
                                                    : cudf::data_type(cudf::type_id::INT32);
  auto offsets_column           = std::make_unique<cudf::column>(offsets_dtype,
                                                       total_offsets,
                                                       std::move(decompressed_offsets),
                                                       std::move(offsets_null_mask),
                                                       0 /*null count*/);

  std::vector<std::unique_ptr<cudf::column>> child_columns;
  child_columns.push_back(std::move(offsets_column));
  auto final_column = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::STRING),
                                                     total_rows,
                                                     std::move(decompressed_char_array),
                                                     std::move(decompressed_null_mask),
                                                     reduced_null_count,
                                                     std::move(child_columns));
  cudf_pinned_resource.deallocate_async(
    partition_offsets, num_partitions * sizeof(offsets_type), stream);
  return final_column;
}

template class string_compressed_sliced_column<false>;
template class string_compressed_sliced_column<true>;

}  // namespace storage
}  // namespace gqe
