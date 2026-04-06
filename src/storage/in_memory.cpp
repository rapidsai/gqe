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

#include <gqe/storage/in_memory.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/boost.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/mpi_helpers.hpp>

#include <cudf_test/default_stream.hpp>

#include <cuda/__barrier/barrier_arrive_tx.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <fmt/format.h>
#include <nvcomp.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock
#include <ranges>
#include <shared_mutex>  // std::shared_lock
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {

namespace storage {

// Get the size of a string column in bytes, can only be called on string columns. Returns a pair of
// (chars_size, offsets_size).
std::pair<int64_t, int64_t> get_string_column_size(cudf::column const& cudf_column)
{
  if (cudf_column.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("get_string_column_size can only be called on string columns");
  }
  cudf::strings_column_view string_col(cudf_column);
  int64_t chars_size = static_cast<int64_t>(string_col.chars_size(cudf::get_default_stream()));
  cudf::column_view offsets = string_col.offsets();
  int64_t offsets_size      = offsets.size() * cudf::size_of(offsets.type());
  return std::make_pair(chars_size, offsets_size);
}

// Get the size of a column in bytes, string column includes both chars and offsets size.
int64_t get_column_size(cudf::column const& cudf_column)
{
  auto const num_rows = cudf_column.size();
  auto const type     = cudf_column.type();
  if (type.id() == cudf::type_id::STRING) {
    auto [chars_size, offsets_size] = get_string_column_size(cudf_column);
    return chars_size + offsets_size;
  }
  return static_cast<int64_t>(num_rows * cudf::size_of(type));
}

// Support logging of in_memory_column_type
constexpr auto format_as(const in_memory_column_type type)
{
  switch (type) {
    case in_memory_column_type::CONTIGUOUS: return "CONTIGUOUS";
    case in_memory_column_type::COMPRESSED: return "COMPRESSED";
    case in_memory_column_type::COMPRESSED_SLICED: return "COMPRESSED_SLICED";
    case in_memory_column_type::SHARED_CONTIGUOUS: return "SHARED_CONTIGUOUS";
    case in_memory_column_type::SHARED_COMPRESSED: return "SHARED_COMPRESSED";
  }
  // Throw an exception so this will get fixed quickly when we add a new type.
  throw std::runtime_error("in_memory_column_type not supported for log formatting");
}

// Cast a column from a row group to a desired type
template <typename T>
const T& get_column(const row_group* row_group, size_t column_idx)
{
  const auto& column_base = row_group->get_column(column_idx);
  const T& column         = static_cast<const T&>(column_base);
  return column;
}

pruning_result_t::pruning_result_t(const std::vector<zone_map::partition>& partitions,
                                   cudf::size_type partition_size)
  : _partitions(partitions), _partition_size(partition_size)
{
  // Compute candidate partitions
  std::copy_if(partitions.begin(),
               partitions.end(),
               std::back_inserter(_candidate_partitions),
               [](const auto& partition) { return !partition.pruned; });

  // Compute consolidated partitions
  _consolidated_partitions = zone_map::consolidate_partitions(partitions);
  auto it                  = std::remove_if(_consolidated_partitions.begin(),
                           _consolidated_partitions.end(),
                           [](const zone_map::partition& partition) { return partition.pruned; });
  _consolidated_partitions.erase(it, _consolidated_partitions.end());

  // Compute vector of partition indexes
  for (const auto& partition : _consolidated_partitions) {
    size_t start_idx = partition.start / _partition_size;
    size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
    for (size_t partition_idx = start_idx; partition_idx < end_idx; ++partition_idx) {
      _partition_indexes.emplace_back(partition_idx);
    }
  }

  // Compute total number of rows
  _num_rows = 0;
  for (const auto& partition : _consolidated_partitions) {
    _num_rows += partition.end - partition.start;
  }
}

cudf::size_type pruning_result_t::partition_size() const { return _partition_size; }

const std::vector<zone_map::partition>& pruning_result_t::candidate_partitions() const
{
  return _candidate_partitions;
}

const std::vector<zone_map::partition>& pruning_result_t::consolidated_partitions() const
{
  return _consolidated_partitions;
}

const std::vector<size_t>& pruning_result_t::partition_indexes() const
{
  return _partition_indexes;
}

cudf::size_type pruning_result_t::num_rows() const { return _num_rows; }

shared_column::shared_column(cudf::column_view col,
                             boost::interprocess::managed_shared_memory& segment,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : _type(col.type()),
    _size(col.size()),
    _null_count(col.null_count()),
    _children(SharedColumnAllocator(segment.get_segment_manager())),
    _mr(mr)
{
  if (col.data<std::byte>() == nullptr || col.size() == 0) {
    throw std::runtime_error("Shared column data pointer is null or size is 0");
  }

  if (!cudf::is_fixed_width(col.type()) && col.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error(
      "Shared column only supports fixed width types and strings, does not support cudf type id " +
      std::to_string(static_cast<int>(col.type().id())));
  }

  // Calculate size of data
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view string_device_view(col);
    _data_size                = sizeof(char) * string_device_view.chars_size(stream);
    cudf::column_view offsets = string_device_view.offsets();
    _offsets_size =
      (offsets.size() - 1) *
      (offsets.type().id() == cudf::type_id::INT64 ? sizeof(int64_t) : sizeof(int32_t));
  } else {
    _data_size = cudf::size_of(col.type()) * col.size();
  }

  // Allocate data and copy from device to host
  _data = static_cast<std::byte*>(mr.allocate(_data_size));
  GQE_CUDA_TRY(cudaMemcpy(_data.get(), col.data<std::byte>(), _data_size, cudaMemcpyDeviceToHost));

  // Allocate and copy null mask from device to host
  if (_null_count > 0) {
    _null_mask_size = col.size() * sizeof(cudf::bitmask_type);
    _null_mask      = static_cast<cudf::bitmask_type*>(mr.allocate(_null_mask_size));
    GQE_CUDA_TRY(
      cudaMemcpy(_null_mask.get(), col.null_mask(), _null_mask_size, cudaMemcpyDeviceToHost));
  } else {
    _null_mask_size = 0;
  }

  // Handle children recursively
  auto children_size = col.num_children();
  _children.reserve(children_size);
  for (decltype(children_size) idx = 0; idx < children_size; ++idx) {
    _children.emplace_back(col.child(idx), segment, stream, mr);
  }
}

int64_t shared_column::get_data_size() const { return _data_size; }

int64_t shared_column::get_offsets_size() const { return _offsets_size; }

cudf::data_type shared_column::get_type() const { return _type; }

cudf::column_view shared_column::view() const
{
  // Safety check for dangling pointers
  if (_data.get() == nullptr) {
    throw std::runtime_error(
      "shared_column: Invalid data pointer - original column may have been destroyed");
  }

  // Create child views
  std::vector<cudf::column_view> child_views(_children.size());
  for (decltype(_children.size()) idx = 0; idx < _children.size(); ++idx) {
    child_views[idx] = _children[idx].view();
  }

  auto null_mask =
    _null_count > 0 ? static_cast<const cudf::bitmask_type*>(_null_mask.get()) : nullptr;

  // Create column view
  return cudf::column_view(_type,
                           _size,
                           static_cast<const void*>(_data.get()),
                           null_mask,
                           _null_count,
                           0,  // offset
                           child_views);
}

shared_column::~shared_column()
{
  // Deallocate data and null mask if allocated
  if (_data && _data_size > 0) {
    _mr.deallocate(_data.get(), _data_size);
    _data      = nullptr;
    _data_size = 0;
  }
  if (_null_mask && _null_mask_size > 0) {
    _mr.deallocate(_null_mask.get(), _null_mask_size);
    _null_mask      = nullptr;
    _null_mask_size = 0;
  }
  // Children will be destroyed recursively
}

cudf::size_type shared_contiguous_column::null_count() const { return view().null_count(); }

bool shared_contiguous_column::is_compressed() const { return false; }

int64_t shared_contiguous_column::get_compressed_size() const { return get_uncompressed_size(); }

int64_t shared_contiguous_column::get_uncompressed_size() const
{
  auto found = gqe::utility::find_object<gqe::storage::shared_column>(&_segment, _column_name);
  return found->get_data_size() + found->get_offsets_size();
}

column_compression_statistics shared_contiguous_column::get_compression_stats() const
{
  auto found = gqe::utility::find_object<gqe::storage::shared_column>(&_segment, _column_name);

  if (found->get_type().id() == cudf::type_id::STRING) {
    string_compression_statistics string_stats;
    string_stats.offsets_stats.compressed_size   = found->get_offsets_size();
    string_stats.offsets_stats.uncompressed_size = found->get_offsets_size();
    string_stats.chars_stats.compressed_size     = found->get_data_size();
    string_stats.chars_stats.uncompressed_size   = found->get_data_size();

    return column_compression_statistics(string_stats);
  } else {
    fixed_width_compression_statistics fixed_width_stats;
    fixed_width_stats.compressed_size           = get_compressed_size();
    fixed_width_stats.uncompressed_size         = get_uncompressed_size();
    fixed_width_stats.primary_compressed_size   = fixed_width_stats.compressed_size;
    fixed_width_stats.secondary_compressed_size = fixed_width_stats.uncompressed_size;

    return column_compression_statistics(fixed_width_stats);
  }
}

cudf::column_view shared_contiguous_column::view() const
{
  auto found = gqe::utility::find_object<gqe::storage::shared_column>(&_segment, _column_name);
  return found->view();
}

shared_contiguous_column::~shared_contiguous_column()
{
  if (gqe::utility::multi_process::mpi_rank_zero()) {
    _segment.destroy<gqe::storage::shared_column>(_column_name.c_str());
  }
}

shared_table::shared_table(cudf::table_view table,
                           boost::interprocess::managed_shared_memory& segment,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
  : _columns(SharedColumnAllocator(segment.get_segment_manager())), _size(table.num_rows())
{
  auto num_columns = table.num_columns();

  _columns.reserve(num_columns);
  for (int i = 0; i < num_columns; i++) {
    _columns.emplace_back(table.column(i), segment, stream, mr);
  }
}

cudf::table_view shared_table::view() const
{
  std::vector<cudf::column_view> column_views(_columns.size());
  for (size_t i = 0; i < _columns.size(); i++) {
    column_views[i] = _columns[i].view();
  }
  return cudf::table_view(column_views);
}

std::unique_ptr<cudf::table> shared_table::copy_to_device(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto table_view = view();

  // Creates a new table on memory resource specified in parameters
  auto table_ptr = std::make_unique<cudf::table>(table_view, stream, mr);
  return table_ptr;
}

contiguous_column::contiguous_column(cudf::column&& cudf_column)
  : column_base(), _data(std::move(cudf_column))
{
}

int64_t contiguous_column::size() const
{
  assert(static_cast<std::size_t>(_data.size()) <= std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(_data.size());
}

cudf::size_type contiguous_column::null_count() const { return view().null_count(); }

bool contiguous_column::is_compressed() const { return false; }

int64_t contiguous_column::get_compressed_size() const { return get_uncompressed_size(); }

int64_t contiguous_column::get_uncompressed_size() const { return get_column_size(_data); }

column_compression_statistics contiguous_column::get_compression_stats() const
{
  if (_data.type().id() == cudf::type_id::STRING) {
    string_compression_statistics string_stats;
    auto [chars_size, offsets_size]              = get_string_column_size(_data);
    string_stats.offsets_stats.compressed_size   = offsets_size;
    string_stats.offsets_stats.uncompressed_size = offsets_size;
    string_stats.chars_stats.compressed_size     = chars_size;
    string_stats.chars_stats.uncompressed_size   = chars_size;

    return column_compression_statistics(string_stats);
  } else {
    fixed_width_compression_statistics fixed_width_stats;
    fixed_width_stats.compressed_size           = get_compressed_size();
    fixed_width_stats.uncompressed_size         = get_uncompressed_size();
    fixed_width_stats.primary_compressed_size   = fixed_width_stats.compressed_size;
    fixed_width_stats.secondary_compressed_size = fixed_width_stats.uncompressed_size;

    return column_compression_statistics(fixed_width_stats);
  }
}

cudf::column_view contiguous_column::view() const { return _data.view(); }

cudf::mutable_column_view contiguous_column::mutable_view() { return _data.mutable_view(); }

std::unique_ptr<rmm::device_buffer> compressed_column::compress(rmm::device_buffer const* input,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr,
                                                                bool is_null_mask)
{
  std::unique_ptr<rmm::device_buffer> output;
  if (is_null_mask) {
    output = _nvcomp_manager.do_compress(input,
                                         _is_null_mask_compressed,
                                         _null_mask_compressed_size,
                                         _null_mask_uncompressed_size,
                                         stream,
                                         mr);
  } else {
    output = _nvcomp_manager.do_compress(
      input, _is_compressed, _compressed_size, _uncompressed_size, stream, mr);
  }
  return output;
}

compressed_column::compressed_column(cudf::column&& cudf_column,
                                     compression_format comp_format,
                                     decompression_backend decompress_backend,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     int compression_chunk_size,
                                     double compression_ratio_threshold,
                                     bool use_cpu_compression,
                                     int compression_level,
                                     std::string column_name,
                                     cudf::data_type cudf_type)
  : column_base(),
    _comp_format(comp_format),
    _is_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(comp_format,
                    compression_format::none,
                    decompress_backend,
                    compression_chunk_size,
                    stream,
                    mr,
                    compression_ratio_threshold,
                    0.0,  // secondary compression ratio threshold
                    0.0,  // secondary compression multiplier threshold
                    use_cpu_compression,
                    compression_level,
                    column_name,
                    cudf_type)
{
  _size       = cudf_column.size();
  _cudf_type  = cudf_column.type();
  _null_count = cudf_column.null_count();

  auto column_content        = cudf_column.release();
  _uncompressed_size         = column_content.data->size();
  _compressed_data           = compress(column_content.data.get(), stream, mr, false);
  _compressed_size           = _compressed_data->size();
  _null_mask_compressed_size = 0;
  if (_null_count > 0) {
    _compressed_null_mask      = compress(column_content.null_mask.get(), stream, mr, true);
    _null_mask_compressed_size = _compressed_null_mask->size();
  }

  _compressed_children.reserve(column_content.children.size());

  for (auto& child : column_content.children) {
    auto dtype = child->type().id();

    if ((comp_format == gqe::compression_format::best_compression_ratio) ||
        (comp_format == gqe::compression_format::best_decompression_speed)) {
      best_compression_config(
        dtype,
        comp_format,
        (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
    }

    _compressed_children.push_back(std::make_unique<compressed_column>(std::move(*child),
                                                                       comp_format,
                                                                       decompress_backend,
                                                                       stream,
                                                                       mr,
                                                                       compression_chunk_size,
                                                                       compression_ratio_threshold,
                                                                       use_cpu_compression,
                                                                       compression_level,
                                                                       column_name + "_child",
                                                                       cudf_type));
  }
}

bool compressed_column::is_compressed() const { return _is_compressed; }

int64_t compressed_column::get_compressed_size() const
{
  return _compressed_size + _null_mask_compressed_size;
}

int64_t compressed_column::get_uncompressed_size() const
{
  return _uncompressed_size + _null_mask_uncompressed_size;
}

column_compression_statistics compressed_column::get_compression_stats() const
{
  // `compressed_column` is about to be deprecated, thus we don't add further support for this.
  column_compression_statistics stats;
  return stats;
}

shared_compressed_column_base::shared_compressed_column_base(
  gqe::storage::compressed_column&& compressed_column,
  boost::interprocess::managed_shared_memory& segment,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : _size(compressed_column._size),
    _compressed_size(compressed_column._compressed_data->size()),
    _null_mask_compressed_size(
      compressed_column._null_count > 0 ? compressed_column._compressed_null_mask->size() : 0),
    _cudf_type(compressed_column._cudf_type),
    _null_count(compressed_column._null_count),
    _comp_format(compressed_column._comp_format),
    _is_compressed(compressed_column._is_compressed),
    _is_null_mask_compressed(compressed_column._is_null_mask_compressed),
    _nvcomp_manager(compressed_column._nvcomp_manager.get_comp_format(),
                    compression_format::none,
                    compressed_column._nvcomp_manager.get_decompress_backend(),
                    compressed_column._nvcomp_manager.get_compression_chunk_size(),
                    stream,
                    mr,
                    compressed_column._nvcomp_manager.get_compression_ratio_threshold(),
                    0.0,  // secondary compression ratio threshold
                    0.0,  // secondary compression multiplier threshold
                    compressed_column._nvcomp_manager.get_use_cpu_compression(),
                    compressed_column._nvcomp_manager.get_compression_level(),
                    compressed_column._nvcomp_manager.get_column_name(),
                    compressed_column._nvcomp_manager.get_cudf_type()),
    _compressed_children(SharedColumnAllocator(segment.get_segment_manager())),
    _mr(mr)
{
  // Allocate and copy main data
  _compressed_data = mr.allocate(compressed_column._compressed_data->size());
  GQE_CUDA_TRY(cudaMemcpy(_compressed_data.get(),
                          compressed_column._compressed_data->data(),
                          compressed_column._compressed_data->size(),
                          cudaMemcpyDeviceToHost));

  // Allocate and copy null mask
  if (compressed_column._null_count > 0) {
    _compressed_null_mask = mr.allocate(compressed_column._compressed_null_mask->size());
    GQE_CUDA_TRY(cudaMemcpy(_compressed_null_mask.get(),
                            compressed_column._compressed_null_mask->data(),
                            compressed_column._compressed_null_mask->size(),
                            cudaMemcpyDeviceToHost));
  }

  // Handle children
  _compressed_children.reserve(compressed_column._compressed_children.size());
  for (size_t i = 0; i < compressed_column._compressed_children.size(); ++i) {
    auto& compressed_child = compressed_column._compressed_children[i];
    _compressed_children.emplace_back(std::move(*compressed_child), segment, stream, mr);
  }
}

std::unique_ptr<cudf::column> shared_compressed_column_base::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> decompressed_children;
  decompressed_children.reserve(_compressed_children.size());

  for (auto& compressed_child : _compressed_children) {
    decompressed_children.push_back(compressed_child.decompress(stream, mr));
  }

  std::unique_ptr<rmm::device_buffer> decompressed_data;

  if (_is_compressed) {
    decompressed_data =
      _nvcomp_manager.do_decompress(_compressed_data.get(), _compressed_size, stream, mr);
  } else {
    decompressed_data = std::make_unique<rmm::device_buffer>(
      static_cast<const void*>(_compressed_data.get()), _compressed_size, stream, mr);
  }

  std::unique_ptr<rmm::device_buffer> decompressed_null_mask;
  if (_null_count > 0) {
    if (_is_null_mask_compressed) {
      decompressed_null_mask = _nvcomp_manager.do_decompress(
        _compressed_null_mask.get(), _null_mask_compressed_size, stream, mr);
    } else {
      decompressed_null_mask =
        std::make_unique<rmm::device_buffer>(static_cast<const void*>(_compressed_null_mask.get()),
                                             _null_mask_compressed_size,
                                             stream,
                                             mr);
    }
  } else {
    decompressed_null_mask = std::make_unique<rmm::device_buffer>();
  }

  return std::make_unique<cudf::column>(_cudf_type,
                                        _size,
                                        std::move(*decompressed_data),
                                        std::move(*decompressed_null_mask),
                                        _null_count,
                                        std::move(decompressed_children));
}

shared_compressed_column_base::~shared_compressed_column_base()
{
  if (_compressed_data) {
    _mr.deallocate(_compressed_data.get(), _compressed_size);
    _compressed_data = nullptr;
    _compressed_size = 0;
  }
  if (_compressed_null_mask) {
    _mr.deallocate(_compressed_null_mask.get(), _null_mask_compressed_size);
    _compressed_null_mask      = nullptr;
    _null_mask_compressed_size = 0;
  }
  // Children will be destroyed recursively
}

std::unique_ptr<cudf::column> shared_compressed_column::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);
  return shared_compressed_column_base->decompress(stream, mr);
}

shared_compressed_column::shared_compressed_column(
  std::string name, boost::interprocess::managed_shared_memory& segment)
  : _name(std::move(name)), _segment(segment)
{
}

shared_compressed_column::~shared_compressed_column()
{
  if (gqe::utility::multi_process::mpi_rank_zero()) {
    _segment.destroy<gqe::storage::shared_compressed_column_base>(_name.c_str());
  }
}

int64_t shared_compressed_column::size() const
{
  // TODO: optimize and only use find once, and store pointer
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);

  return shared_compressed_column_base->_size;
}

bool shared_compressed_column::is_compressed() const
{
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);

  return shared_compressed_column_base->_is_compressed;
}

int64_t shared_compressed_column::get_compressed_size() const
{
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);

  return shared_compressed_column_base->_compressed_size +
         shared_compressed_column_base->_null_mask_compressed_size;
}

int64_t shared_compressed_column::get_uncompressed_size() const
{
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);

  return shared_compressed_column_base->_uncompressed_size +
         shared_compressed_column_base->_null_mask_uncompressed_size;
}

column_compression_statistics shared_compressed_column::get_compression_stats() const
{
  // FIXME: Given that we need 1) support for shared compressed sliced column and 2) refactor for
  // IPC communication without using boost shared memory, we leave the implementation of this
  // function until those are implemented, as the current implementation would likely be
  // significantly changed by those features.
  column_compression_statistics stats;
  return stats;
}

cudf::size_type shared_compressed_column::null_count() const
{
  // TODO: DRY with shared_compressed_column::size()
  auto shared_compressed_column_base =
    gqe::utility::find_object<gqe::storage::shared_compressed_column_base>(&_segment, _name);

  return shared_compressed_column_base->_null_count;
}

int64_t compressed_column::size() const { return _size; }

cudf::size_type compressed_column::null_count() const { return _null_count; }

// TODO: DRY with shared_compressed_column::decompress
// Current difference is that this does not store data as rmm::device_buffers, as they are not safe
// in shared memory And, children are not stored as unique_ptrs, as directly creating structures is
// significantly less complicated with memory management
std::unique_ptr<cudf::column> compressed_column::decompress(rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> decompressed_children;
  decompressed_children.reserve(_compressed_children.size());

  for (auto const& compressed_child : _compressed_children) {
    decompressed_children.push_back(compressed_child->decompress(stream, mr));
  }

  std::unique_ptr<rmm::device_buffer> decompressed_data;
  // We'll save off the compression config to get the size, then we can allocate here like we do for
  // the sliced columns

  if (_is_compressed) {
    decompressed_data = _nvcomp_manager.do_decompress(
      cudf::device_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_compressed_data->data()),
                                       _compressed_size),

      stream,
      mr);
  } else {
    decompressed_data = std::make_unique<rmm::device_buffer>(*_compressed_data, stream, mr);
  }

  std::unique_ptr<rmm::device_buffer> decompressed_null_mask;
  if (_null_count > 0) {
    if (_is_null_mask_compressed) {
      decompressed_null_mask = _nvcomp_manager.do_decompress(
        cudf::device_span<uint8_t const>(
          reinterpret_cast<uint8_t const*>(_compressed_null_mask->data()),
          _compressed_null_mask->size()),
        stream,
        mr);
    } else {
      decompressed_null_mask =
        std::make_unique<rmm::device_buffer>(*_compressed_null_mask, stream, mr);
    }
  } else {
    decompressed_null_mask = std::make_unique<rmm::device_buffer>();
  }

  return std::make_unique<cudf::column>(_cudf_type,
                                        _size,
                                        std::move(*decompressed_data),
                                        std::move(*decompressed_null_mask),
                                        _null_count,
                                        std::move(decompressed_children));
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map)
  : _columns(std::move(columns)), _zone_map(std::move(zone_map))
{
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns)
  : row_group(std::move(columns), nullptr)
{
}

int64_t row_group::size() const
{
  if (_columns.empty()) {
    return 0;
  } else {
    return _columns.front()->size();
  }
}

int64_t row_group::num_columns() const { return _columns.size(); }

column_base& row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

gqe::zone_map* row_group::zone_map() const { return _zone_map.get(); }

in_memory_table::in_memory_table(memory_kind::type memory_kind,
                                 std::vector<std::string> const& column_names,
                                 std::vector<cudf::data_type> const& column_types,
                                 task_manager_context* ctx)
  : table(),
    _task_manager_context(ctx),
    _memory_kind(memory_kind),
    _column_types(std::move(column_types)),
    _row_groups(),
    _row_group_latch()
{
  if (!_task_manager_context) {
    throw std::invalid_argument("New in-memory table requires a non-null task_manager_context.");
  }

  // Populate column name-to-index map
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }
}

bool in_memory_table::is_readable() const { return true; }

bool in_memory_table::is_writeable() const { return true; }

int32_t in_memory_table::max_concurrent_readers() const
{
  auto const nrow_groups = _row_groups.size();
  assert(nrow_groups <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));

  return static_cast<int32_t>(nrow_groups);
}

int32_t in_memory_table::max_concurrent_writers() const
{
  // Each writer appends a new row group to the table. Thus, the number of
  // writers is theoretically unlimited.
  return std::numeric_limits<int32_t>::max();
}

std::unique_ptr<readable_view> in_memory_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new in_memory_readable_view(this));
}

std::unique_ptr<writeable_view> in_memory_table::writeable_view()
{
  return std::unique_ptr<storage::writeable_view>(new in_memory_writeable_view(this));
}

cudf::size_type in_memory_table::get_column_index(std::string const& column_name) const
{
  return _column_name_to_index.at(column_name);
}

in_memory_table::row_group_appender in_memory_table::get_row_group_appender()
{
  return {&_row_groups, &_row_group_latch};
}

in_memory_table::row_group_appender::row_group_appender(
  std::deque<row_group>* non_owning_row_groups, std::shared_mutex* non_owning_row_group_latch)
  : _non_owning_row_groups(non_owning_row_groups),
    _non_owning_row_group_latch(non_owning_row_group_latch)
{
}

void in_memory_table::row_group_appender::operator()(row_group&& new_row_group)
{
  std::unique_lock latch_guard(*_non_owning_row_group_latch);
  _non_owning_row_groups->push_back(std::move(new_row_group));
}

void in_memory_table::row_group_appender::operator()(std::vector<row_group>&& new_row_groups)
{
  std::unique_lock latch_guard(*_non_owning_row_group_latch);

  for (auto&& rg : new_row_groups) {
    _non_owning_row_groups->push_back(std::move(rg));
  }
}

// Helper to create a column view slice
cudf::column_view slice_column(const cudf::column_view& full_column,
                               const cudf::size_type size,
                               const cudf::size_type offset,
                               const cudf::size_type null_counts)
{
  const auto type      = full_column.type();
  const auto head      = full_column.head();
  const auto null_mask = full_column.null_mask();
  const auto children =
    std::vector<cudf::column_view>(full_column.child_begin(), full_column.child_end());
  return cudf::column_view(type, size, head, null_mask, null_counts, offset, children);
}

// This is called for all columns but always returns the same value.
// We could make pruning_results_t a dedicated type which caches this information. Then we would
// have to adapt the logic that iterates over individual row groups. But that is an opportunity to
// DRY the casting of the column. Probably not worth the effort.
size_t compute_num_rows(const pruning_results_t& pruning_results)
{
  size_t num_rows = 0;
  for (const auto& [row_group, pruning_result] : pruning_results) {
    num_rows += pruning_result.num_rows();
  }
  return num_rows;
}

output_column_helper::output_column_helper(cudf::data_type cudf_type,
                                           std::shared_ptr<pruning_results_t> pruning_results,
                                           cudf::size_type column_idx)
  : _cudf_type(cudf_type),
    _pruning_results(std::move(pruning_results)),
    _column_idx(column_idx),
    _num_rows(compute_num_rows(*_pruning_results))
{
}

std::unique_ptr<output_column_helper> make_output_column_helper(
  const cudf::data_type cudf_type,
  const in_memory_column_type column_type,
  std::shared_ptr<pruning_results_t> pruning_results,
  cudf::size_type column_idx,
  const row_group* first_row_group)
{
  if (column_type == in_memory_column_type::CONTIGUOUS &&
      cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    GQE_LOG_DEBUG(
      "Creating helper for contiguous fixed-width column; column_idx = {}, column_type = {}, "
      "cudf_type = {}",
      column_idx,
      column_type,
      cudf_type);
    return std::make_unique<contiguous_output_column_helper<contiguous_column>>(
      cudf_type, pruning_results, column_idx);
  } else if (column_type == in_memory_column_type::COMPRESSED_SLICED &&
             cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    GQE_LOG_DEBUG(
      "Creating helper for compressed_sliced fixed-width column; column_idx = {}, column_type = "
      "{}, cudf_type = {}",
      column_idx,
      column_type,
      cudf_type);
    return std::make_unique<compressed_sliced_output_column_helper>(
      cudf_type, pruning_results, column_idx);
  } else if (column_type == in_memory_column_type::COMPRESSED_SLICED &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    auto& string_column =
      get_column<string_compressed_sliced_column_base>(first_row_group, column_idx);
    bool is_large_string = string_column.is_large_string();
    GQE_LOG_DEBUG(
      "Creating helper for string_compressed_sliced column; column_idx = {}, is_large_string = {}",
      column_idx,
      is_large_string);
    if (is_large_string) {
      return std::make_unique<string_compressed_sliced_output_column_helper<true>>(
        cudf_type, pruning_results, column_idx);
    } else {
      return std::make_unique<string_compressed_sliced_output_column_helper<false>>(
        cudf_type, pruning_results, column_idx);
    }
  }
  // Use cudf::concatenate for all other columns
  GQE_LOG_DEBUG("Creating concatenating helper; column_idx = {}, column_type = {}, cudf_type = {}",
                column_idx,
                column_type,
                cudf_type);
  return std::make_unique<concatenating_output_column_helper>(
    cudf_type, pruning_results, column_idx);
}

size_t concatenating_output_column_helper::num_copied_buffers(size_t partition_size)
{
  // concatenating_output_column_helper uses cudf::concatenate and does not participate in the
  // batched memcpy
  return 0;
}

void concatenating_output_column_helper::prepare_batched_memcpy(std::byte** dst_ptrs,
                                                                std::byte** src_ptrs,
                                                                size_t* sizes,
                                                                rmm::cuda_stream_view stream)
{
  // concatenating_output_column_helper uses cudf::concatenate and does not participate in the
  // batched memcpy; do nothing
  return;
}

bool concatenating_output_column_helper::decompress_row_group_columns(
  rmm::cuda_stream_view decompression_stream)
{
  bool has_compressed_column = false;
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column = row_group->get_column(_column_idx);
    switch (column.type()) {
      case in_memory_column_type::COMPRESSED_SLICED: {
        throw std::logic_error("Use dedicated compressed_sliced_output_column_helper instead");
        break;
      }
      case in_memory_column_type::CONTIGUOUS:  // Fall through
      case in_memory_column_type::SHARED_CONTIGUOUS:
        // Nothing to do since has_compressed_column is initialized to false
        break;
      // Code duplication below will be removed once we have dedicated column helpers for different
      // column types.
      case in_memory_column_type::COMPRESSED: {
        auto decompressed_column =
          dynamic_cast<compressed_column&>(column).decompress(decompression_stream);
        _decompressed_columns.emplace(&column, std::move(decompressed_column));
        has_compressed_column = true;
        break;
      }
      case in_memory_column_type::SHARED_COMPRESSED: {
        auto decompressed_column =
          dynamic_cast<shared_compressed_column&>(column).decompress(decompression_stream);
        _decompressed_columns.emplace(&column, std::move(decompressed_column));
        has_compressed_column = true;
        break;
      }
      default:
        GQE_LOG_ERROR("Column type not supported by concatenating_output_column_helper: {}",
                      column.type());
        throw std::logic_error("Column type not supported");
    }
  }
  return has_compressed_column;
}

std::unique_ptr<cudf::column> concatenating_output_column_helper::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream)
{
  std::vector<cudf::column_view> pruned_columns;
  pruned_columns.reserve(_pruning_results->size());
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      auto& column = row_group->get_column(_column_idx);
      // Get the (potentially decompressed) column view
      cudf::column_view full_column_view;
      switch (column.type()) {
        case in_memory_column_type::COMPRESSED_SLICED: {
          throw std::logic_error("Use dedicated compressed_sliced_output_column_helper instead");
          break;
        }
        case in_memory_column_type::CONTIGUOUS: {
          full_column_view = dynamic_cast<contiguous_column&>(column).view();
          break;
        }
        case in_memory_column_type::SHARED_CONTIGUOUS: {
          full_column_view = dynamic_cast<shared_contiguous_column&>(column).view();
          break;
        }
        case in_memory_column_type::COMPRESSED:  // Fall through
        case in_memory_column_type::SHARED_COMPRESSED: {
          auto stored_decompressed_column = _decompressed_columns.find(&column);
          if (stored_decompressed_column != _decompressed_columns.end()) {
            full_column_view = stored_decompressed_column->second->view();
          } else {
            throw std::logic_error("Column should have been decompressed already.");
          }
          break;
        }
        default: throw std::logic_error("Unknown column type");
      }
      // Construct the partitioned column view
      const auto size               = partition.end - partition.start;
      const auto offset             = partition.start;
      const auto null_counts        = partition.null_counts[_column_idx];
      const auto pruned_column_view = slice_column(full_column_view, size, offset, null_counts);
      pruned_columns.push_back(std::move(pruned_column_view));
    }
  }

  // Construct the output column from the (partitioned) row group column views using
  // cudf::concatenate
  std::unique_ptr<cudf::column> cudf_column =
    cudf::concatenate(pruned_columns, concatenation_stream, rmm::mr::get_current_device_resource());

  GQE_LOG_DEBUG(
    "Processed column with concatenate: _column_idx = {}, "
    "cudf_column->size() = {}",
    _column_idx,
    cudf_column->size());
  return cudf_column;
}

// Allocate the output buffer belonging to an output_column_helper and return a pointer to it.
std::byte* allocate_output_buffer(
  std::unique_ptr<rmm::device_buffer>& buffer,
  size_t buffer_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  buffer                = std::make_unique<rmm::device_buffer>(buffer_size, stream, mr);
  std::byte* target_ptr = static_cast<std::byte*>(buffer->data());
  return target_ptr;
}

// Allocate the output buffer belonging to an output_column_helper and return a pointer to it.
// The size is also returned for logging.
std::byte* allocate_output_buffer(
  std::unique_ptr<rmm::device_buffer>& buffer,
  size_t num_values,
  size_t value_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  size_t buffer_size    = num_values * value_size;
  std::byte* target_ptr = allocate_output_buffer(buffer, buffer_size, stream, mr);
  return target_ptr;
}

template <typename T>
size_t contiguous_output_column_helper<T>::num_copied_buffers(size_t partition_size)
{
  size_t num_copied_buffers = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    num_copied_buffers += pruning_result.consolidated_partitions().size();
  }
  return num_copied_buffers;
}

template <typename T>
void contiguous_output_column_helper<T>::prepare_batched_memcpy(std::byte** dst_ptrs,
                                                                std::byte** src_ptrs,
                                                                size_t* sizes,
                                                                rmm::cuda_stream_view stream)
{
  std::byte* target_ptr =
    allocate_output_buffer(_output_buffer, _num_rows, cudf::size_of(_cudf_type), stream);
  GQE_LOG_DEBUG("_column_idx = {}, target_ptr = {}, _output_buffer->size = {}",
                _column_idx,
                (void*)target_ptr,
                _output_buffer->size());
  // Fill pointer and size arrays
  size_t buffer_idx = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const T& column        = get_column<T>(row_group, _column_idx);
    const auto column_view = column.view();
    std::byte* source_ptr  = const_cast<std::byte*>(column_view.template data<std::byte>());
    GQE_LOG_DEBUG("Filling array for batched memcpy; _column_idx = {}, buffer_idx, source_ptr = {}",
                  _column_idx,
                  buffer_idx,
                  (void*)source_ptr);
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      const size_t size_in_bytes = (partition.end - partition.start) * cudf::size_of(_cudf_type);
      dst_ptrs[buffer_idx]       = target_ptr;
      src_ptrs[buffer_idx]       = source_ptr + partition.start * cudf::size_of(_cudf_type);
      sizes[buffer_idx]          = size_in_bytes;
#ifndef NDEBUG
      GQE_LOG_DEBUG(
        "_column_idx = {}, buffer_idx = {}, dst_ptrs[buffer_idx] = {}, src_ptrs[buffer_idx] = {}, "
        "size_in_bytes = {}",
        _column_idx,
        buffer_idx,
        (void*)dst_ptrs[buffer_idx],
        (void*)src_ptrs[buffer_idx],
        size_in_bytes);
#endif
      target_ptr += size_in_bytes;
      ++buffer_idx;
    }
  }
}

template <typename T>
bool contiguous_output_column_helper<T>::decompress_row_group_columns(
  rmm::cuda_stream_view decompression_stream)
{
  // Contiguous columns are not compressed; do nothing
  return false;
}

template <typename T>
std::unique_ptr<cudf::column> contiguous_output_column_helper<T>::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream)
{
  auto null_mask             = std::make_unique<rmm::device_buffer>();
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(*_output_buffer), std::move(*null_mask), null_count);
}

size_t compressed_sliced_output_column_helper::num_copied_buffers(size_t partition_size)
{
  size_t num_copied_buffers = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    num_copied_buffers += gqe::utility::divide_round_up(pruning_result.num_rows(), partition_size);
  }
  return num_copied_buffers;
}

void compressed_sliced_output_column_helper::prepare_batched_memcpy(std::byte** dst_ptrs,
                                                                    std::byte** src_ptrs,
                                                                    size_t* sizes,
                                                                    rmm::cuda_stream_view stream)
{
  std::byte* target_ptr =
    allocate_output_buffer(_output_buffer, _num_rows, cudf::size_of(_cudf_type), stream);
  GQE_LOG_DEBUG(
    "Created output buffer; _column_idx = {}, target_ptr = {}, _output_buffer->size() = {}, "
    "num_rows = {}, "
    "_cudf_type = {}",
    _column_idx,
    (void*)target_ptr,
    _output_buffer->size(),
    _num_rows,
    _cudf_type);
  size_t arrays_offset     = 0;
  size_t target_ptr_offset = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const auto& column = get_column<compressed_sliced_column>(row_group, _column_idx);
    column.prepare_batched_memcpy(dst_ptrs + arrays_offset,
                                  src_ptrs + arrays_offset,
                                  sizes + arrays_offset,
                                  _compression_buffers,
                                  pruning_result,
                                  target_ptr + target_ptr_offset,
                                  stream);
    arrays_offset += pruning_result.partition_indexes().size();
    target_ptr_offset += pruning_result.num_rows() * cudf::size_of(_cudf_type);
  }
}

bool compressed_sliced_output_column_helper::decompress_row_group_columns(
  rmm::cuda_stream_view decompression_stream)
{
  bool has_compressed_column  = false;
  size_t decompression_offset = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column          = get_column<compressed_sliced_column>(row_group, _column_idx);
    std::byte* target_ptr = static_cast<std::byte*>(_output_buffer->data()) + decompression_offset;
    rmm::device_buffer* compression_buffer = _compression_buffers[&column].get();
    const auto was_compressed =
      column.decompress(target_ptr, compression_buffer, pruning_result, decompression_stream);
    GQE_LOG_DEBUG(
      "Decompressed row group column; target_ptr = {}, compression_buffer = {}, was_compressed = "
      "{}",
      (void*)target_ptr,
      (void*)compression_buffer,
      was_compressed);
    has_compressed_column |= was_compressed;
    decompression_offset += pruning_result.num_rows() * cudf::size_of(_cudf_type);
  }
  return has_compressed_column;
}

std::unique_ptr<cudf::column> compressed_sliced_output_column_helper::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream)
{
  auto null_mask             = std::make_unique<rmm::device_buffer>();
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(*_output_buffer), std::move(*null_mask), null_count);
}

template <bool large_string_mode>
size_t string_compressed_sliced_output_column_helper<large_string_mode>::num_copied_buffers(
  size_t partition_size)
{
  size_t num_copied_buffers = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    num_copied_buffers += pruning_result.partition_indexes().size();
  }
  // Need to copy the character buffer and offset buffer for each partition
  return 2 * num_copied_buffers;
}

template <bool large_string_mode>
void string_compressed_sliced_output_column_helper<large_string_mode>::prepare_batched_memcpy(
  std::byte** dst_ptrs, std::byte** src_ptrs, size_t* sizes, rmm::cuda_stream_view stream)
{
  // Compute required buffer sizes and allocate char buffer and offset buffer
  size_t char_buffer_size   = 0;
  size_t offset_buffer_size = 0;
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    auto [column_char_buffer_size, column_offset_buffer_size] = column.buffer_sizes(pruning_result);
    char_buffer_size += column_char_buffer_size;
    offset_buffer_size += column_offset_buffer_size;
  }
  // Add one offset for length of last value
  offset_buffer_size += large_string_mode ? cudf::size_of(cudf::data_type(cudf::type_id::INT64))
                                          : cudf::size_of(cudf::data_type(cudf::type_id::INT32));
  auto char_target_ptr   = allocate_output_buffer(_char_buffer, char_buffer_size, stream);
  auto offset_target_ptr = allocate_output_buffer(_offset_buffer, offset_buffer_size, stream);
  GQE_LOG_DEBUG(
    "Created output buffers for COMPRESSED_SLICED string column; _column_idx = {}, char_target_ptr "
    "= {}, char_buffer_size = {}, offset_target_ptr = {}, offset_buffer_size = {}, num_rows = {}",
    _column_idx,
    (void*)char_target_ptr,
    char_buffer_size,
    (void*)offset_target_ptr,
    offset_buffer_size,
    _num_rows);

  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    const auto [copied_buffers, char_buffer_offset, offset_buffer_offset] =
      column.prepare_batched_memcpy(dst_ptrs,
                                    src_ptrs,
                                    sizes,
                                    _char_compression_buffers,
                                    _offset_compression_buffers,
                                    pruning_result,
                                    char_target_ptr,
                                    offset_target_ptr,
                                    stream);
    dst_ptrs += copied_buffers;
    src_ptrs += copied_buffers;
    sizes += copied_buffers;
    char_target_ptr += char_buffer_offset;
    offset_target_ptr += offset_buffer_offset;
  }
}

template <bool large_string_mode>
bool string_compressed_sliced_output_column_helper<large_string_mode>::decompress_row_group_columns(
  rmm::cuda_stream_view decompression_stream)
{
  // TODO Very similar to compressed_sliced_columnn::decompress_row_groups; split and refactor?
  bool has_compressed_column   = false;
  std::byte* char_target_ptr   = static_cast<std::byte*>(_char_buffer->data());
  std::byte* offset_target_ptr = static_cast<std::byte*>(_offset_buffer->data());
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    rmm::device_buffer* char_compression_buffer   = _char_compression_buffers[&column].get();
    rmm::device_buffer* offset_compression_buffer = _offset_compression_buffers[&column].get();
    const auto [was_compressed, char_decompressed_size] =
      column.decompress(char_target_ptr,
                        offset_target_ptr,
                        char_compression_buffer,
                        offset_compression_buffer,
                        pruning_result,
                        decompression_stream);
    const size_t offset_decompressed_size =
      pruning_result.num_rows() * cudf::size_of(offset_element_type);
    GQE_LOG_DEBUG(
      "Decompressed row group COMPRESSED_SLICED string column; char_target_ptr = {}, "
      "char_compression_buffer = {}, offset_target_ptr = {}, offset_compression_buffer = {}, "
      "was_compressed = {}, char_decompressed_size = {}, offset_decompressed_size = {}",
      (void*)char_target_ptr,
      (void*)char_compression_buffer,
      (void*)offset_target_ptr,
      (void*)offset_compression_buffer,
      was_compressed,
      char_decompressed_size,
      offset_decompressed_size);
    has_compressed_column |= was_compressed;
    char_target_ptr += char_decompressed_size;
    offset_target_ptr += offset_decompressed_size;
  }
  return has_compressed_column;
}

template <bool large_string_mode>
std::unique_ptr<cudf::column>
string_compressed_sliced_output_column_helper<large_string_mode>::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream)
{
  // Adjust offsets
  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  size_t num_partitions     = 0;
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    num_partitions += pruning_result.partition_indexes().size();
  }

  rmm::device_buffer partition_char_offsets_buffer(
    num_partitions * sizeof(offsets_type), concatenation_stream, cudf_pinned_resource);
  offsets_type* partition_char_offsets =
    reinterpret_cast<offsets_type*>(partition_char_offsets_buffer.data());
  rmm::device_buffer partition_row_offsets_buffer(
    num_partitions * sizeof(cudf::size_type), concatenation_stream, cudf_pinned_resource);
  auto* partition_row_offsets =
    reinterpret_cast<cudf::size_type*>(partition_row_offsets_buffer.data());

  concatenation_stream.synchronize();

  offsets_type char_offset    = 0;
  cudf::size_type row_offset  = 0;
  size_t partition_offset_idx = 0;
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    column.fill_partition_offsets(partition_char_offsets,
                                  partition_row_offsets,
                                  char_offset,
                                  row_offset,
                                  pruning_result,
                                  partition_offset_idx);
    partition_offset_idx += pruning_result.partition_indexes().size();
  }

  // create a copy on device before adjust_offsets since we will do a binary search over this buffer
  auto d_partition_row_offsets_buffer =
    rmm::device_buffer(partition_row_offsets_buffer, concatenation_stream);
  auto d_partition_row_offsets =
    reinterpret_cast<cudf::size_type*>(d_partition_row_offsets_buffer.data());
  adjust_offsets_api(reinterpret_cast<offsets_type*>(_offset_buffer->data()),
                     _num_rows,
                     num_partitions,
                     d_partition_row_offsets,
                     partition_char_offsets,
                     _char_buffer->size(),
                     concatenation_stream);
  GQE_LOG_DEBUG(
    "Adjusted offsets; _column_idx = {}, num_partitions = {}, char_offset = {}, row_offset = {}, "
    "_num_rows = {}, _char_buffer->size() = {}",
    _column_idx,
    num_partitions,
    char_offset,
    row_offset,
    _num_rows,
    _char_buffer->size());

  // Create CUDF column
  auto offset_column = std::make_unique<cudf::column>(
    offset_element_type, _num_rows + 1, std::move(*_offset_buffer), rmm::device_buffer(), 0);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(offset_column));
  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::STRING),
                                        _num_rows,
                                        std::move(*_char_buffer),
                                        rmm::device_buffer(),
                                        0,
                                        std::move(children));
}

template class string_compressed_sliced_output_column_helper<false>;
template class string_compressed_sliced_output_column_helper<true>;

in_memory_read_task::in_memory_read_task(context_reference ctx_ref,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::vector<const row_group*> row_groups,
                                         std::vector<cudf::size_type> column_indexes,
                                         std::vector<cudf::data_type> data_types,
                                         memory_kind::type memory_kind,
                                         std::unique_ptr<gqe::expression> partial_filter,
                                         std::vector<std::shared_ptr<task>> subquery_tasks,
                                         bool force_zero_copy_disable)
  : read_task_base(ctx_ref, task_id, stage_id, std::move(subquery_tasks)),
    _row_groups(std::move(row_groups)),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _memory_kind(std::move(memory_kind)),
    _partial_filter(std::move(partial_filter)),
    _force_zero_copy_disable(force_zero_copy_disable)
{
}

bool in_memory_read_task::can_prune_partitions() const
{
  if (!_partial_filter) { return false; }
  if (get_query_context()->parameters.zone_map_partition_size == 0) { return false; }
  return get_query_context()->parameters.use_partition_pruning;
}

std::unique_ptr<pruning_results_t> in_memory_read_task::evaluate_partial_filter()
{
  auto pruning_results           = std::make_unique<pruning_results_t>();
  cudf::size_type partition_size = get_query_context()->parameters.zone_map_partition_size;
  if (can_prune_partitions()) {
    std::for_each(_row_groups.begin(),
                  _row_groups.end(),
                  [this, &pruning_results, &partition_size](const auto* rg) {
                    if (!rg->zone_map()) {
                      throw std::logic_error("Row group should have a zone map but none exists");
                    }
                    // TODO I don't like passing the query context paramters here
                    const auto partitions =
                      rg->zone_map()->evaluate(get_query_context()->parameters, *_partial_filter);
                    const auto has_unpruned_partitions =
                      std::find_if(partitions.begin(), partitions.end(), [](const auto& partition) {
                        return !partition.pruned;
                      }) != partitions.end();
                    if (has_unpruned_partitions) {
                      // TODO Change output of evaluate to pruning_result?
                      pruning_results->emplace_back(rg,
                                                    pruning_result_t{partitions, partition_size});
                    }
                  });
    GQE_LOG_DEBUG("Row groups after pruning: {}", pruning_results->size());
    return pruning_results;
  } else {
    GQE_LOG_DEBUG("No partial filter; processing all row groups");
    pruning_results->reserve(_row_groups.size());
    std::transform(
      _row_groups.begin(),
      _row_groups.end(),
      std::back_inserter(*pruning_results),
      [this, &partition_size](const auto* rg) {
        std::vector<cudf::size_type> null_counts(rg->num_columns(), 0);
        std::for_each(
          _column_indexes.begin(), _column_indexes.end(), [&null_counts, &rg](const auto i) {
            null_counts[i] = rg->get_column(i).null_count();
          });
        const zone_map::partition partition{.pruned      = false,
                                            .start       = 0,
                                            .end         = static_cast<cudf::size_type>(rg->size()),
                                            .null_counts = null_counts};
        return row_group_with_pruning_result_t{rg, pruning_result_t{{partition}, partition_size}};
      });
    return pruning_results;
  }
}

bool in_memory_read_task::is_zero_copy_possible(const pruning_results_t& pruning_results) const
{
  // The function emit_zero_copy_result assumes the positive outcome of the checks below.
  if (!get_query_context()->parameters.read_zero_copy_enable) {
    GQE_LOG_DEBUG("Zero-copy read is disabled by the query context parameters");
    return false;
  }
  if (!memory_kind::is_gpu_accessible(_memory_kind)) {
    GQE_LOG_WARN(
      "Zero-copy read is not possible because the GPU cannot access pageable memory on this "
      "system");
    return false;
  }
  if (pruning_results.size() > 1) {
    GQE_LOG_DEBUG(
      "Zero-copy read is not possible because there are multiple row groups with "
      "candidate partitions");
    return false;
  }
  if (pruning_results.size() == 0) {
    throw std::logic_error("There should be at least a single candidate partition");
  }
  auto& [row_group, pruning_result] = pruning_results.front();
  for (auto column_idx : _column_indexes) {
    auto type = row_group->get_column(column_idx).type();
    if (type == in_memory_column_type::COMPRESSED ||
        type == in_memory_column_type::COMPRESSED_SLICED ||
        type == in_memory_column_type::SHARED_COMPRESSED) {
      GQE_LOG_DEBUG(
        "Zero-copy read is not possible because at least one column is compressed: "
        "type={}",
        type);
      return false;
    }
  }
  auto num_partitions = pruning_result.consolidated_partitions().size();
  if (num_partitions > 1) {
    GQE_LOG_DEBUG(
      "Zero-copy read is not possible because there are multiple discontiguous "
      "candidate partitions: num_partitions={}",
      num_partitions);
    return false;
  }
  if (num_partitions != 1) {
    throw std::logic_error("There should be exactly one contiguous candidate partition");
  }
  return true;
}

void in_memory_read_task::emit_empty_table()
{
  GQE_LOG_DEBUG("Returning empty table");
  std::vector<std::unique_ptr<cudf::column>> empty_columns;
  for (const auto& data_type : _data_types) {
    // cudf::make_empty_column will fail for nested types, e.g., lists and structs. In this
    // case, we have to use cudf::empty_like on an existing column, or reproduce it.
    // TODO There is no test of the entire query execution pipeline when an empty table is
    // returned
    auto empty_column = cudf::make_empty_column(data_type);
    empty_columns.push_back(std::move(empty_column));
  }
  auto empty_table = std::make_unique<cudf::table>(std::move(empty_columns));
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=empty_table, "
    "output_size={}.",
    task_id(),
    stage_id(),
    empty_table->num_rows());
  emit_result(std::move(empty_table));
}

// This method throws std::logic_error if certain preconditions are not met. These are checked
// by is_zero_copy_possible. They are duplicated here to ensure that the code doesn't break if
// is_zero_copy_possible is changed.
void in_memory_read_task::emit_zero_copy_result(std::unique_ptr<pruning_results_t> pruning_results)
{
  GQE_LOG_DEBUG("Performing zero-copy read");

  if (pruning_results->size() != 1) {
    throw std::logic_error("There must be a single row group with candidate partitions");
  }
  const auto& [row_group, pruning_result] = pruning_results->front();

  if (pruning_result.consolidated_partitions().size() != 1) {
    throw std::logic_error("There must be a single consolidated partition");
  }
  const auto& partition = pruning_result.consolidated_partitions().front();

  std::vector<cudf::column_view> columns;
  for (auto column_idx : _column_indexes) {
    cudf::column_view full_column_view;
    auto& column = row_group->get_column(column_idx);
    if (column.type() == in_memory_column_type::CONTIGUOUS) {
      full_column_view = dynamic_cast<contiguous_column&>(column).view();
    } else if (column.type() == in_memory_column_type::SHARED_CONTIGUOUS) {
      full_column_view = dynamic_cast<shared_contiguous_column&>(column).view();
    } else {
      throw std::logic_error("The column must be contiguous");
    }
    const auto size                    = partition.end - partition.start;
    const auto offset                  = partition.start;
    const auto null_counts             = partition.null_counts[column_idx];
    const auto partitioned_column_view = slice_column(full_column_view, size, offset, null_counts);
    columns.push_back(partitioned_column_view);
  }
  cudf::table_view result(columns);
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_reference, "
    "output_size={}.",
    task_id(),
    stage_id(),
    result.num_rows());
  emit_result(result);
}

void in_memory_read_task::emit_copied_result(std::unique_ptr<pruning_results_t> pruning_results)
{
  GQE_LOG_DEBUG("Copying data to GPU");

  // Create helpers to create output columns from the columns of individual row groups
  std::vector<std::unique_ptr<output_column_helper>> output_columns;
  const size_t num_columns = _column_indexes.size();
  // Share the pruning result across all output columns
  std::shared_ptr<pruning_results_t> shared_pruning_result(std::move(pruning_results));
  for (size_t i = 0; i < num_columns; ++i) {
    const auto column_idx       = _column_indexes[i];
    const auto cudf_type        = _data_types[i];
    const auto* first_row_group = shared_pruning_result->front().first;
    const auto& first_column    = first_row_group->get_column(column_idx);
    const auto column_type      = first_column.type();
    auto output_column          = make_output_column_helper(
      cudf_type, column_type, shared_pruning_result, column_idx, first_row_group);
    output_columns.push_back(std::move(output_column));
  }

  // Lock on the shared copy engine stream
  std::unique_lock<std::mutex> ce_lock;
  // Event used to synchronize the CUDF default stream and the shared copy engine stream.
  cudaEvent_t ce_evt;
  // Stream for decompression. Either the dedicated copy_engine_stream or the CUDF default stream.
  rmm::cuda_stream_view decompression_stream;

  // Lock the shared copy engine stream and determine decompression stream
  bool should_use_overlap_mtx = get_query_context()->parameters.use_overlap_mtx;
  if (should_use_overlap_mtx) {
    GQE_LOG_DEBUG("Using overlap mutex for in_memory_read_task");
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&ce_evt, cudaEventDisableTiming));
    ce_lock =
      std::unique_lock{*get_context_reference()._task_manager_context->copy_engine_stream.mtx};
    decompression_stream = get_context_reference()._task_manager_context->copy_engine_stream.stream;
  } else {
    decompression_stream = cudf::get_default_stream();
  }

  // Determine number of buffers for batched memcpy
  const auto partition_size = get_query_context()->parameters.zone_map_partition_size;
  const size_t num_copied_buffers =
    std::accumulate(output_columns.begin(),
                    output_columns.end(),
                    0,
                    [partition_size](size_t num_copied_buffers, const auto& column) {
                      return num_copied_buffers + column->num_copied_buffers(partition_size);
                    });
  GQE_LOG_DEBUG("Determined number of buffers for batched memcpy; num_copied_buffers = {}",
                num_copied_buffers);

  if (num_copied_buffers == 0) {
    GQE_LOG_DEBUG("No buffers to copy, skipping batched memcpy");
  } else {
    utility::nvtx_scoped_range nvtx_range("Filling cudaMemcpyBatchAsync arrays");

    // Allocate pointer and size arrays for batched memcpy
    auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
    auto stream               = cudf::get_default_stream();

    rmm::device_buffer sizes_buffer(
      sizeof(size_t) * num_copied_buffers, stream, cudf_pinned_resource);
    size_t* sizes = reinterpret_cast<size_t*>(sizes_buffer.data());
    rmm::device_buffer dst_ptrs_buffer(
      sizeof(std::byte*) * num_copied_buffers, stream, cudf_pinned_resource);
    std::byte** dst_ptrs = reinterpret_cast<std::byte**>(dst_ptrs_buffer.data());
    rmm::device_buffer src_ptrs_buffer(
      sizeof(std::byte*) * num_copied_buffers, stream, cudf_pinned_resource);
    std::byte** src_ptrs = reinterpret_cast<std::byte**>(src_ptrs_buffer.data());
    GQE_CUDA_TRY(cudaStreamSynchronize(stream));

    GQE_LOG_DEBUG(
      "Created pointer arrays for batched memcpy; src_ptrs = {}, dst_ptrs = {}, sizes = {}",
      (void*)src_ptrs,
      (void*)dst_ptrs,
      (void*)sizes);

    GQE_LOG_DEBUG("Filling pointer arrays for batched memcpy");
    size_t offset = 0;
    for (const auto& output_column : output_columns) {
      output_column->prepare_batched_memcpy(
        dst_ptrs + offset, src_ptrs + offset, sizes + offset, stream);
      // TODO Can I get rid of the duplicate use of partition_size?
      // Return num_copied_buffers from prepare_batched_memcpy
      offset += output_column->num_copied_buffers(partition_size);
    }

    GQE_LOG_DEBUG("Batched memcpy");
    stream.synchronize();
    {
      utility::semaphore_acquire_guard guard(
        get_context_reference()._task_manager_context->batched_memcpy_semaphore);
      gqe::utility::do_batched_memcpy(
        (void**)dst_ptrs, (void**)src_ptrs, sizes, num_copied_buffers, stream);
      stream.synchronize();
    }  // Release the semaphore.
  }

  GQE_LOG_DEBUG("Decompressing columns");
  bool has_compressed_columns = false;
  {
    utility::nvtx_scoped_range nvtx_range("Decompressing columns");
    decompression_stream.synchronize();
    {
      utility::semaphore_acquire_guard guard(
        get_context_reference()._task_manager_context->decompress_semaphore);
      for (auto& column_creator : output_columns) {
        bool was_compressed = column_creator->decompress_row_group_columns(decompression_stream);
        has_compressed_columns |= was_compressed;
      }
      decompression_stream.synchronize();
    }  // Release the semaphore.
    GQE_LOG_DEBUG("Decompressed columns; has_compressed_columns = {}", has_compressed_columns);
  }

  if (has_compressed_columns && should_use_overlap_mtx) {
    // Wait on the CUDF default stream until data transfers of the shared copy engine are
    // finished.
    auto default_stream = cudf::get_default_stream().value();
    auto shared_ce_stream =
      get_context_reference()._task_manager_context->copy_engine_stream.stream.value();
    GQE_CUDA_TRY(cudaEventRecord(ce_evt, shared_ce_stream));
    GQE_CUDA_TRY(cudaStreamWaitEvent(default_stream, ce_evt));
  }

  // If the column is not compressed we do a CE copy in concatenate, we want this copy to be done on
  // the shared CE stream to allow for pipelining. Otherwise, it is better to do this on the thread
  // specific stream to avoid false Note: The shared CE stream guarantees serial execution, if more
  // than one row group is compressed then the ce_evt will have overwrites. This is intended as we
  // only need to wait on the "last" scheduled copy on the shared CE stream.
  rmm::cuda_stream_view concatenation_stream;
  if (not has_compressed_columns && should_use_overlap_mtx) {
    concatenation_stream = decompression_stream;
  } else {
    concatenation_stream = cudf::get_default_stream();
  }

  // Create output cudf::columns
  auto cudf_column_view =
    output_columns | std::views::transform([concatenation_stream](auto& output_column) {
      utility::nvtx_scoped_range nvtx_range("Creating cuDF column");
      return output_column->make_cudf_column(concatenation_stream);
    });
  std::vector<std::unique_ptr<cudf::column>> cudf_columns(cudf_column_view.begin(),
                                                          cudf_column_view.end());

  // Sync on the decompression stream if it was used to concatenate the columns
  if (not has_compressed_columns && should_use_overlap_mtx) {
    GQE_CUDA_TRY(cudaEventRecord(ce_evt, decompression_stream.value()));
    GQE_CUDA_TRY(cudaStreamWaitEvent(cudf::get_default_stream().value(), ce_evt));
  }

  // We sync while holding the lock to avoid aggressive memory allocation leading to OOM errors.
  // This is a hotfix so the behaviour is same as that for cuDF 24.12.
  if (should_use_overlap_mtx) {
    get_context_reference()._task_manager_context->copy_engine_stream.stream.synchronize();
    ce_lock.unlock();
    GQE_CUDA_TRY(cudaEventDestroy(ce_evt));
  }

  // Emit table
  auto result = std::make_unique<cudf::table>(std::move(cudf_columns));
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_value, output_size={}.",
    task_id(),
    stage_id(),
    result->num_rows());
  emit_result(std::move(result));
}

void in_memory_read_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range in_memory_read_task_range("in_memory_read_task");

  auto pruning_results = evaluate_partial_filter();

  if (pruning_results->empty()) {
    emit_empty_table();
  } else if (is_zero_copy_possible(*pruning_results)) {
    emit_zero_copy_result(std::move(pruning_results));
  } else {
    emit_copied_result(std::move(pruning_results));
  }

  GQE_LOG_DEBUG("Finishing read task");
  remove_dependencies();
  GQE_LOG_DEBUG("Finished read task");
}

in_memory_write_task::in_memory_write_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> input,
  rmm::mr::device_memory_resource* non_owned_memory_resource,
  in_memory_table::row_group_appender appender,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> column_indexes,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
  : write_task_base(ctx_ref, task_id, stage_id, input),
    _non_owned_memory_resource(non_owned_memory_resource),
    _appender(std::move(appender)),
    _column_indexes(std::move(column_indexes)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types)),
    _memory_kind(memory_kind),
    _statistics(statistics)
{
}

void in_memory_write_task::execute_default()
{
  GQE_LOG_TRACE("Write task execute");
  prepare_dependencies();

  utility::nvtx_scoped_range in_memory_write_task_range("in_memory_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  // Check if input schema matches output schema
  auto num_columns = _column_indexes.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data types must have the same length as the number of columns");
  }
  if (static_cast<decltype(num_columns)>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  // Create columns and insert data into them

  // Specify the stream to enqueue copy on
  auto stream = cudf::get_default_stream();

  const auto partition_size            = get_query_context()->parameters.zone_map_partition_size;
  const bool partition_pruning_enabled = partition_size > 0;

  // FIXME: Maybe we can zero-copy construct (i.e., move to) the new columns if
  // the input_table columns have the same memory type as the result columns.
  // This would require us to destroy the result cache here, but that should be
  // ok because `write` is the root task.
  std::vector<std::unique_ptr<column_base>> new_columns(_column_indexes.size());

  // Local table statistics for the current row group.
  table_statistics row_group_table_stats(input_table.num_rows(), _data_types);

  for (decltype(input_table.num_columns()) column_idx = 0; column_idx < input_table.num_columns();
       ++column_idx) {
    auto const& input_column = input_table.column(column_idx);
    auto cudf_column         = cudf::column(input_column, stream, _non_owned_memory_resource);

    auto comp_format = get_query_context()->parameters.in_memory_table_compression_format;
    auto const compression_chunk_size =
      get_query_context()->parameters.in_memory_table_compression_chunk_size;
    auto compression_ratio_threshold =
      get_query_context()->parameters.in_memory_table_compression_ratio_threshold;
    auto secondary_comp_format =
      get_query_context()->parameters.in_memory_table_secondary_compression_format;
    auto secondary_compression_ratio_threshold =
      get_query_context()->parameters.in_memory_table_secondary_compression_ratio_threshold;
    auto secondary_compression_multiplier_threshold =
      get_query_context()->parameters.in_memory_table_secondary_compression_multiplier_threshold;
    auto use_cpu_compression = get_query_context()->parameters.use_cpu_compression;
    auto compression_level   = get_query_context()->parameters.compression_level;
    auto decompress_backend  = get_query_context()->parameters.decompress_backend;

    auto dtype = cudf_column.type().id();

    if ((comp_format == gqe::compression_format::best_compression_ratio) ||
        (comp_format == gqe::compression_format::best_decompression_speed)) {
      best_compression_config(
        dtype,
        comp_format,
        (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
    }

    if (comp_format == compression_format::none) {
      GQE_LOG_TRACE("Uncompressed column size {}", cudf_column.size());
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<contiguous_column>(std::move(cudf_column));
    } else {
      if (not partition_pruning_enabled) {
        GQE_LOG_TRACE("Compressed column size {}", cudf_column.size());
        auto cudf_type = cudf_column.type();
        new_columns[_column_indexes[column_idx]] =
          std::make_unique<compressed_column>(std::move(cudf_column),
                                              comp_format,
                                              decompress_backend,
                                              stream,
                                              _non_owned_memory_resource,
                                              compression_chunk_size,
                                              compression_ratio_threshold,
                                              use_cpu_compression,
                                              compression_level,
                                              _column_names[column_idx],
                                              cudf_type);
      } else if (dtype == cudf::type_id::STRING) {
        // Get a string view of the column
        cudf::strings_column_view strings_column_view(cudf_column);
        int64_t chars_size = strings_column_view.chars_size(stream);
        GQE_LOG_TRACE("String column size {}", chars_size);
        if (chars_size > std::numeric_limits<int32_t>::max()) {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<true>>(
              std::move(cudf_column),
              partition_size,
              _memory_kind,
              comp_format,
              secondary_comp_format,
              decompress_backend,
              compression_chunk_size,
              compression_ratio_threshold,
              secondary_compression_ratio_threshold,
              secondary_compression_multiplier_threshold,
              use_cpu_compression,
              compression_level,
              stream,
              _non_owned_memory_resource,
              _column_names[column_idx]);
        } else {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<false>>(
              std::move(cudf_column),
              partition_size,
              _memory_kind,
              comp_format,
              secondary_comp_format,
              decompress_backend,
              compression_chunk_size,
              compression_ratio_threshold,
              secondary_compression_ratio_threshold,
              secondary_compression_multiplier_threshold,
              use_cpu_compression,
              compression_level,
              stream,
              _non_owned_memory_resource,
              _column_names[column_idx]);
        }
      } else {
        GQE_LOG_TRACE("Compressed sliced column size {}", cudf_column.size());
        auto cudf_type = cudf_column.type();
        new_columns[_column_indexes[column_idx]] =
          std::make_unique<compressed_sliced_column>(std::move(cudf_column),
                                                     partition_size,
                                                     _memory_kind,
                                                     comp_format,
                                                     secondary_comp_format,
                                                     decompress_backend,
                                                     compression_chunk_size,
                                                     compression_ratio_threshold,
                                                     secondary_compression_ratio_threshold,
                                                     secondary_compression_multiplier_threshold,
                                                     use_cpu_compression,
                                                     compression_level,
                                                     stream,
                                                     _non_owned_memory_resource,
                                                     _column_names[column_idx],
                                                     cudf_type);
      }
    }
    column_statistics col_stats = {
      .column_id         = static_cast<size_t>(_column_indexes[column_idx]),
      .compression_stats = new_columns[_column_indexes[column_idx]]->get_compression_stats()};
    row_group_table_stats.add_column_statistics(col_stats);
  }

  // Create zone map

  std::unique_ptr<gqe::zone_map> zone_map =
    partition_pruning_enabled ? std::make_unique<gqe::zone_map>(input_table, partition_size)
                              : nullptr;

  // Create row group from columns
  auto row_group = storage::row_group(std::move(new_columns), std::move(zone_map));

  // Append row group to table
  _appender(std::move(row_group));
  _statistics->append_table_statistics(row_group_table_stats);

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

void in_memory_write_task::execute_shared_memory()
{
  prepare_dependencies();

  utility::nvtx_scoped_range in_memory_write_task_range("in_memory_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  // Check if input schema matches output schema
  auto num_columns = _column_indexes.size();

  cudf::table_view input_table;

  // Other ranks will have empty result table
  if (gqe::utility::multi_process::mpi_rank_zero()) {
    input_table = *dependent_tasks[0]->result();
    if (_data_types.size() != num_columns) {
      throw std::length_error("Data types must have the same length as the number of columns");
    }
    if (static_cast<decltype(num_columns)>(input_table.num_columns()) != num_columns) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }

    for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
      if (_data_types[column_idx] != input_table.column(column_idx).type()) {
        throw std::invalid_argument("Query result schema must match Parquet output schema");
      }
    }
  }

  // Create columns and insert data into them

  // Specify the stream to enqueue copy on
  auto stream = cudf::get_default_stream();

  // FIXME: Maybe we can zero-copy construct (i.e., move to) the new columns if
  // the input_table columns have the same memory type as the result columns.
  // This would require us to destroy the result cache here, but that should be
  // ok because `write` is the root task.
  std::vector<std::unique_ptr<column_base>> new_columns(_column_indexes.size());

  auto& segment =
    static_cast<memory_resource::boost_shared_memory_resource*>(_non_owned_memory_resource)
      ->segment();

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    auto comp_format = get_query_context()->parameters.in_memory_table_compression_format;
    auto compression_ratio_threshold =
      get_query_context()->parameters.in_memory_table_compression_ratio_threshold;

    std::string shared_column_name = [this, &column_idx] {
      std::ostringstream oss;
      oss << "shared_column_" << stage_id() << "_" << task_id() << "_" << column_idx << "_"
          << _column_names[column_idx];
      return oss.str();
    }();

    cudf::column_view input_column = [&input_table, &column_idx]() {
      if (gqe::utility::multi_process::mpi_rank_zero()) {
        return input_table.column(column_idx);
      } else {
        return cudf::column_view{};
      }
    }();

    if (comp_format == compression_format::none) {
      if (gqe::utility::multi_process::mpi_rank_zero()) {
        segment.construct<gqe::storage::shared_column>(shared_column_name.c_str())(
          input_column, segment, stream, _non_owned_memory_resource);
      }

      new_columns[_column_indexes[column_idx]] =
        std::make_unique<shared_contiguous_column>(shared_column_name, segment);
    } else {
      if (gqe::utility::multi_process::mpi_rank_zero()) {
        cudf::column cudf_column(input_column, stream, rmm::mr::get_current_device_resource());

        auto const chunk_size =
          get_query_context()->parameters.in_memory_table_compression_chunk_size;

        auto const use_cpu_compression = get_query_context()->parameters.use_cpu_compression;
        auto const compression_level   = get_query_context()->parameters.compression_level;
        auto const decompress_backend  = get_query_context()->parameters.decompress_backend;

        auto dtype = input_column.type().id();

        if ((comp_format == gqe::compression_format::best_compression_ratio) ||
            (comp_format == gqe::compression_format::best_decompression_speed)) {
          best_compression_config(
            dtype,
            comp_format,
            (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
        }

        // TODO: Not safe to pass the cudf::data_type to shared memory, need to investigate why
        // It's only used for logging.
        gqe::storage::compressed_column compressed_column(std::move(cudf_column),
                                                          comp_format,
                                                          decompress_backend,
                                                          stream,
                                                          rmm::mr::get_current_device_resource(),
                                                          chunk_size,
                                                          compression_ratio_threshold,
                                                          use_cpu_compression,
                                                          compression_level,
                                                          _column_names[column_idx],
                                                          cudf::data_type{cudf::type_id::EMPTY});
        segment.construct<gqe::storage::shared_compressed_column_base>(shared_column_name.c_str())(
          std::move(compressed_column), segment, stream, _non_owned_memory_resource);
      }

      new_columns[_column_indexes[column_idx]] =
        std::make_unique<shared_compressed_column>(shared_column_name, segment);
    }
  }

  // Create row group from columns
  // Append row group to table
  const auto partition_size = get_query_context()->parameters.zone_map_partition_size;
  std::unique_ptr<gqe::zone_map> zone_map;
  if (partition_size <= 0) {
    zone_map = nullptr;
  } else {
    std::string shared_table_name = "shared_zone_map_table_" + _column_names[0] +
                                    std::to_string(stage_id()) + "_" + std::to_string(task_id());

    if (gqe::utility::multi_process::mpi_rank_zero()) {
      segment.construct<gqe::shared_zone_map_table>(shared_table_name.c_str())(
        input_table, partition_size, &segment, stream, _non_owned_memory_resource);
    }

    zone_map = std::make_unique<gqe::shared_zone_map>(partition_size, shared_table_name, &segment);
  }

  if (gqe::utility::multi_process::mpi_rank_zero()) {
    _statistics->add_rows(input_table.num_rows());
  }

  auto row_group = storage::row_group(std::move(new_columns), std::move(zone_map));
  _appender(std::move(row_group));

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

void in_memory_write_task::execute()
{
  if (get_query_context()->parameters.use_in_memory_table_multigpu) {
    execute_shared_memory();
  } else {
    execute_default();
  }
}

in_memory_readable_view::in_memory_readable_view(in_memory_table* non_owning_table)
  : readable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<read_task_base>> in_memory_readable_view::get_read_tasks(
  std::vector<readable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types)
{
  assert(!task_parameters.empty() && "Must have at least one read task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");

  assert(task_parameters.size() <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto parallelism = static_cast<int64_t>(task_parameters.size());

  // Transform column names into column indexes
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](const std::string& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  // Acquire shared read latch on row groups and release on function return
  std::shared_lock latch_guard(_non_owning_table->_row_group_latch);

  assert(_non_owning_table->_row_groups.size() <=
         static_cast<decltype(_non_owning_table->_row_groups.size())>(
           std::numeric_limits<int64_t>::max()));
  const auto nrow_groups = static_cast<int64_t>(_non_owning_table->_row_groups.size());

  const auto max_nrow_groups_per_instance = utility::divide_round_up(nrow_groups, parallelism);

  // Transform partial filter to zone map filter. Assumes that the partial filter is the same for
  // all tasks.
  const auto zone_map_filter =
    transform_partial_filter(task_parameters.front().partial_filter.get());

  // Create tasks
  std::vector<std::unique_ptr<read_task_base>> read_tasks;
  read_tasks.reserve(task_parameters.size());
  {
    // Iterate over multiple iterators at the same time
    auto task            = task_parameters.begin();
    int64_t begin_offset = 0;

    while (task != task_parameters.end() && begin_offset < nrow_groups) {
      const auto end_offset = std::min(begin_offset + max_nrow_groups_per_instance, nrow_groups);

      // Get a chunk of row groups
      std::vector<const row_group*> row_groups_chunk;
      assert(end_offset >= begin_offset);
      row_groups_chunk.reserve(end_offset - begin_offset);
      std::transform(_non_owning_table->_row_groups.cbegin() + begin_offset,
                     _non_owning_table->_row_groups.cbegin() + end_offset,
                     std::back_inserter(row_groups_chunk),
                     [](const row_group& rg) { return &rg; });

      auto zone_map_filter_copy = zone_map_filter ? zone_map_filter->clone() : nullptr;

      // Create a new read task
      auto read_task = std::make_unique<in_memory_read_task>(ctx_ref,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(row_groups_chunk),
                                                             column_indexes,
                                                             data_types,
                                                             _non_owning_table->_memory_kind,
                                                             std::move(zone_map_filter_copy),
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
}

std::unique_ptr<gqe::expression> in_memory_readable_view::transform_partial_filter(
  gqe::expression* partial_filter)
{
  if (!partial_filter) { return nullptr; }
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  if (zone_map_filter) {
    GQE_LOG_DEBUG(
      "Using partial filter to prune results\nPartial filter:\n{}\nZone map filter:\n{}",
      expression_json_formatter::to_json(*partial_filter),
      expression_json_formatter::to_json(*zone_map_filter));
  }
  return zone_map_filter;
}

in_memory_writeable_view::in_memory_writeable_view(in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<write_task_base>> in_memory_writeable_view::get_write_tasks(
  std::vector<writeable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
{
  assert(!task_parameters.empty() && "Must have at least one write task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");

  // Transform column names into column indexes
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](const std::string& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  std::vector<std::unique_ptr<write_task_base>> write_tasks;
  write_tasks.reserve(task_parameters.size());

  for (auto& task_parameter : task_parameters) {
    // Create appender, which will acquire a write latch to the row groups vector
    auto appender = _non_owning_table->get_row_group_appender();

    // Get the memory resource from task_manager_context
    auto* memory_resource = _non_owning_table->_task_manager_context->get_table_memory_resource_ptr(
      _non_owning_table->_memory_kind);

    /// Create a new write task
    auto write_task = std::make_unique<in_memory_write_task>(ctx_ref,
                                                             task_parameter.task_id,
                                                             stage_id,
                                                             std::move(task_parameter.input),
                                                             memory_resource,
                                                             std::move(appender),
                                                             _non_owning_table->_memory_kind,
                                                             column_indexes,
                                                             column_names,
                                                             data_types,
                                                             statistics);
    write_tasks.push_back(std::move(write_task));
  }

  return write_tasks;
}

}  // namespace storage

}  // namespace gqe
