/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <gqe/executor/optimization_parameters.hpp>
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
#include <gqe/utility/multi_process_helpers.hpp>
#include <gqe/utility/serialization.hpp>

#include <cudf_test/default_stream.hpp>

#include <cuda/__barrier/barrier_arrive_tx.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <fmt/format.h>
#include <nvcomp.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock
#include <ranges>
#include <shared_mutex>  // std::shared_lock
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
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
    case in_memory_column_type::COMPRESSED_SLICED: return "COMPRESSED_SLICED";
    case in_memory_column_type::SHARED_CONTIGUOUS: return "SHARED_CONTIGUOUS";
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
  _candidate_partitions.reserve(partitions.size());
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

  // Compute vector of partition indexes when partitioning is enabled. Unpartitioned results keep
  // only row ranges; compressed-sliced materialization requires a positive partition size.
  if (_partition_size > 0) {
    size_t num_partition_indexes = 0;
    for (const auto& partition : _consolidated_partitions) {
      const size_t start_idx = partition.start / _partition_size;
      const size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      num_partition_indexes += end_idx - start_idx;
    }
    _partition_indexes.reserve(num_partition_indexes);

    for (const auto& partition : _consolidated_partitions) {
      size_t start_idx = partition.start / _partition_size;
      size_t end_idx   = gqe::utility::divide_round_up(partition.end, _partition_size);
      for (size_t partition_idx = start_idx; partition_idx < end_idx; ++partition_idx) {
        _partition_indexes.emplace_back(partition_idx);
      }
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
    _mr(mr),
    _allocation_stream(stream)
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

  // Allocate data and copy from device to host. We always pass
  // `rmm::CUDA_ALLOCATION_ALIGNMENT` (256), which is the alignment the
  // underlying device resources use anyway.
  _data = static_cast<std::byte*>(mr.allocate(stream, _data_size, rmm::CUDA_ALLOCATION_ALIGNMENT));
  GQE_CUDA_TRY(cudaMemcpy(_data.get(), col.data<std::byte>(), _data_size, cudaMemcpyDeviceToHost));

  // Allocate and copy null mask from device to host
  if (_null_count > 0) {
    _null_mask_size = col.size() * sizeof(cudf::bitmask_type);
    _null_mask      = static_cast<cudf::bitmask_type*>(
      mr.allocate(stream, _null_mask_size, rmm::CUDA_ALLOCATION_ALIGNMENT));
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
    _mr.deallocate(_allocation_stream, _data.get(), _data_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    _data      = nullptr;
    _data_size = 0;
  }
  if (_null_mask && _null_mask_size > 0) {
    _mr.deallocate(
      _allocation_stream, _null_mask.get(), _null_mask_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
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
  if (gqe::utility::multi_process::nvshmem_rank_zero()) {
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
    _column_names(std::move(column_names)),
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

namespace {

[[nodiscard]] std::string sanitize_path(std::string value)
{
  for (char& c : value) {
    if (c == '/' || c == '\\') { c = '_'; }
  }
  return value;
}

/** On-disk column snapshot: `<rg_dir>/<sanitized_column_name>.bin`. */
[[nodiscard]] std::filesystem::path serialized_column_path(std::filesystem::path const& rg_dir,
                                                           std::string const& column_name)
{
  return rg_dir / (sanitize_path(column_name) + ".bin");
}

/** Metadata sidecar: `<rg_dir>/<sanitized_column_name>.json`. */
[[nodiscard]] std::filesystem::path serialized_column_metadata_path(
  std::filesystem::path const& rg_dir, std::string const& column_name)
{
  return rg_dir / (sanitize_path(column_name) + ".json");
}

[[nodiscard]] std::string sanitize_json_string(std::string_view value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '"' || c == '\\') { escaped.push_back('\\'); }
    escaped.push_back(c);
  }
  return escaped;
}

//**
// Write column metadata to a JSON file. Used for serialization.
// @param json_path The path to the JSON file.
// @param column_name The name of the column.
// @param row_group_index The index of the row group.
// @param opt_params The optimization parameters.
// @param column The column to write metadata for.
// @return True if the metadata was written successfully, false otherwise.
// */
void write_column_metadata(std::filesystem::path const& json_path,
                           std::string const& column_name,
                           size_t row_group_index,
                           optimization_parameters const& opt_params,
                           compressed_sliced_column const& column)
{
  auto const format = column.data_buffer_view().compression_format();
  std::ofstream out(json_path);
  if (!out) {
    throw std::runtime_error("Could not open metadata file for writing: " + json_path.string());
  }

  // TODO: Add documentation for version number for Serialization.
  out << fmt::format(
    "{{\n"
    R"(  "serialization_version_major": {},)"
    "\n"
    R"(  "serialization_version_minor": {},)"
    "\n"
    R"(  "column_name": "{}",)"
    "\n"
    R"(  "row_group": {},)"
    "\n"
    R"(  "zone_map_size": {},)"
    "\n"
    R"(  "compression_format": "{}",)"
    "\n"
    R"(  "compression_level": {},)"
    "\n"
    R"(  "uncompressed_size": {},)"
    "\n"
    R"(  "compressed_size": {},)"
    "\n"
    R"(  "is_compressed": {})"
    "\n"
    "}}\n",
    utility::serialization_version_major,
    utility::serialization_version_minor,
    sanitize_json_string(column_name),
    row_group_index,
    opt_params.zone_map_partition_size,
    gqe::to_string(format),
    opt_params.compression_level,
    column.get_uncompressed_size(),
    column.get_compressed_size(),
    column.is_compressed() ? "true" : "false");

  if (!out) { throw std::runtime_error("Failed writing metadata file: " + json_path.string()); }
}

/**
 * @brief Numeric suffix of `rg-<n>` directory names.
 *
 * @throw std::invalid_argument if @p is not `rg-<non-negative-integer>`.
 */
[[nodiscard]] int64_t row_group_directory_index(std::filesystem::path const& p)
{
  const std::string name = p.filename().string();
  static constexpr std::string_view k_prefix{"rg-"};
  if (name.size() <= k_prefix.size() || !name.starts_with(k_prefix)) {
    throw std::invalid_argument("row group directory must be named rg-<index>, got: " + name);
  }

  int64_t n         = 0;
  const auto* first = name.data() + k_prefix.size();
  const auto* last  = name.data() + name.size();
  if (auto const r = std::from_chars(first, last, n); r.ec != std::errc{} || r.ptr != last) {
    throw std::invalid_argument("row group directory index is not a plain integer: " + name);
  }
  if (n < 0) {
    throw std::invalid_argument("row group directory index must be non-negative: " + name);
  }
  return n;
}

/** Column indices to serialize (compressed-sliced only); computed before any disk I/O. */
[[nodiscard]] std::vector<std::vector<size_t>> serializable_column_indices_per_row_group(
  std::deque<row_group> const& row_groups, std::string const& table_name)
{
  std::vector<std::vector<size_t>> indices_per_row_group(row_groups.size());
  for (size_t rg_idx = 0; rg_idx < row_groups.size(); ++rg_idx) {
    auto const ncols = static_cast<size_t>(row_groups[rg_idx].num_columns());
    for (size_t col_idx = 0; col_idx < ncols; ++col_idx) {
      column_base const& col = row_groups[rg_idx].get_column(static_cast<cudf::size_type>(col_idx));
      if (col.type() != in_memory_column_type::COMPRESSED_SLICED) {
        GQE_LOG_TRACE("skipping column idx={} type_id={} table='{}'",
                      col_idx,
                      static_cast<int>(col.type()),
                      table_name);
        continue;
      }
      indices_per_row_group[rg_idx].push_back(col_idx);
    }
  }
  return indices_per_row_group;
}

/** Serialize row-group columns under `table_serialized_data_root/rg-<i>/{compression}/{chunk}/`. */
void serialize_row_group_columns(row_group const& rg,
                                 size_t row_group_index,
                                 std::string const& table_serialized_data_root,
                                 std::string const& table_name,
                                 std::vector<std::string> const& column_names,
                                 std::span<size_t const> column_indices,
                                 optimization_parameters const& opt_params,
                                 rmm::cuda_stream_view stream)
{
  std::filesystem::path const column_chunk_dir = utility::serialized_row_group_column_root(
    std::filesystem::path(table_serialized_data_root), row_group_index, opt_params);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(column_chunk_dir, mkdir_ec);

  GQE_LOG_TRACE(
    "serialize_row_group_columns: begin table='{}' row_group={} table_serialized_data_root='{}' "
    "column_dir='{}' num_columns={} mkdir_ec={}",
    table_name,
    row_group_index,
    table_serialized_data_root,
    column_chunk_dir.string(),
    rg.num_columns(),
    mkdir_ec.value());

  if (mkdir_ec) { throw std::runtime_error("Could not create directories: " + mkdir_ec.message()); }

  for (size_t col_idx : column_indices) {
    auto const& sliced = static_cast<compressed_sliced_column const&>(
      rg.get_column(static_cast<cudf::size_type>(col_idx)));
    std::string label = column_names[col_idx];

    std::filesystem::path const file_path = serialized_column_path(column_chunk_dir, label);
    std::filesystem::path const metadata_path =
      serialized_column_metadata_path(column_chunk_dir, label);
    GQE_LOG_TRACE("column='{}' idx={} bin='{}' json='{}' table='{}'",
                  label,
                  col_idx,
                  file_path.string(),
                  metadata_path.string(),
                  table_name);
    write_column_metadata(metadata_path, label, row_group_index, opt_params, sliced);
    sliced.serialize_to_disk(label, file_path, row_group_index, stream);
  }

  GQE_LOG_TRACE(
    "Serialization completed successfully: table='{}' row_group={}", table_name, row_group_index);
}

/** Serialize row-group zone maps under `table_serialized_data_root/rg-<i>/zone_maps/`. */
void serialize_zone_maps_to_disk(std::deque<row_group> const& row_groups,
                                 std::filesystem::path const& table_serialized_data_root,
                                 std::string const& table_name,
                                 rmm::cuda_stream_view stream)
{
  (void)stream;
  for (size_t i = 0; i < row_groups.size(); ++i) {
    std::filesystem::path const zm_dir =
      utility::serialized_row_group_zone_maps_root(table_serialized_data_root, i);
    std::error_code ec;
    std::filesystem::create_directories(zm_dir, ec);
    if (ec) {
      throw std::runtime_error("Could not create zone_maps directory '" + zm_dir.string() +
                               "': " + ec.message());
    }
    GQE_LOG_TRACE("Created zone_maps directory: {}", zm_dir.string());
    // TODO: write zone map payload for row_groups[i] into zm_dir
  }
  GQE_LOG_TRACE(
    "serialize_zone_maps_to_disk: directories created; payload write not implemented table='{}' "
    "table_serialized_data_root='{}'",
    table_name,
    table_serialized_data_root.string());
  throw std::runtime_error(
    "serialize_zone_maps_to_disk: zone map payload write not implemented for table '" + table_name +
    "'");
}

/**
 * Load row-group zone maps from `table_serialized_data_root/rg-<i>/zone_maps/` before column
 * snapshots (not
 * implemented yet).
 */
void deserialize_zone_maps_from_disk(std::deque<row_group>& row_groups,
                                     std::filesystem::path const& table_serialized_data_root,
                                     std::vector<std::filesystem::path> const& rg_dirs,
                                     std::string const& table_name,
                                     rmm::cuda_stream_view stream)
{
  (void)row_groups;
  (void)table_serialized_data_root;
  (void)rg_dirs;
  (void)stream;
  GQE_LOG_TRACE(
    "deserialize_zone_maps_from_disk: not implemented yet table='{}' "
    "table_serialized_data_root='{}'",
    table_name,
    table_serialized_data_root.string());
  throw std::runtime_error("deserialize_zone_maps_from_disk: not implemented yet for table '" +
                           table_name + "'");
}

}  // namespace

void in_memory_table::deserialize_table_from_disk(std::string const& table_serialized_data_root,
                                                  std::string const& table_name,
                                                  rmm::cuda_stream_view stream)
{
  (void)table_name;
  std::unique_lock latch_guard(_row_group_latch);
  _row_groups.clear();

  std::filesystem::path const root(table_serialized_data_root);
  if (!std::filesystem::is_directory(root)) {
    throw std::runtime_error("passed path is not a directory: " + table_serialized_data_root);
  }

  std::vector<std::filesystem::path> rg_dirs;
  for (auto const& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_directory()) { continue; }
    std::string const fn = entry.path().filename().string();
    if (fn.starts_with("rg-")) { rg_dirs.push_back(entry.path()); }
  }

  try {
    // `directory_iterator` order is unspecified; load row groups in rg-0, rg-1, … numeric order.
    std::ranges::sort(rg_dirs, {}, row_group_directory_index);

    deserialize_zone_maps_from_disk(_row_groups, root, rg_dirs, table_name, stream);

    auto const& opt_params = _task_manager_context->get_optimization_parameters();
    rmm::device_async_resource_ref mr =
      _task_manager_context->get_table_memory_resource(_memory_kind);

    for (std::filesystem::path const& rg_path : rg_dirs) {
      auto const row_group_index = static_cast<size_t>(row_group_directory_index(rg_path));
      std::filesystem::path const column_chunk_dir =
        utility::serialized_row_group_column_root(root, row_group_index, opt_params);
      std::vector<std::unique_ptr<column_base>> columns;
      columns.reserve(_column_names.size());

      for (std::string const& col_name : _column_names) {
        std::filesystem::path const bin = serialized_column_path(column_chunk_dir, col_name);
        if (!std::filesystem::exists(bin)) {
          throw std::runtime_error("Deserialize: Missing binary file: " + bin.string());
        }
        std::ifstream probe(bin, std::ios::binary);
        std::uint32_t magic = 0;
        probe.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        probe.close();

        // TODO: Update magic number to something standardized.
        if (magic == 0x47514E01u) {
          auto col = compressed_sliced_column::deserialize_from_disk(
            bin, col_name, row_group_index, stream, mr);
          if (!col) {
            throw std::runtime_error("deserialize: failed column '" + col_name + "' in " +
                                     bin.string());
          }
          columns.push_back(std::move(col));
        } else {
          throw std::runtime_error("deserialize: unsupported magic " + fmt::format("{:x}", magic) +
                                   " for column '" + col_name + "'");
        }
      }

      _row_groups.push_back(row_group(std::move(columns)));
    }
  } catch (std::invalid_argument const& e) {
    throw std::runtime_error(std::string("Invalid row-group directory under '") + root.string() +
                             "': " + e.what());
  }
}

void in_memory_table::serialize_table_to_disk(std::string const& table_serialized_data_root,
                                              std::string const& table_name,
                                              rmm::cuda_stream_view stream)
{
  std::shared_lock latch_guard(_row_group_latch);
  auto const& opt_params = _task_manager_context->get_optimization_parameters();

  auto const column_indices_per_row_group =
    serializable_column_indices_per_row_group(_row_groups, table_name);

  serialize_zone_maps_to_disk(
    _row_groups, std::filesystem::path(table_serialized_data_root), table_name, stream);

  for (size_t i = 0; i < _row_groups.size(); ++i) {
    serialize_row_group_columns(_row_groups[i],
                                i,
                                table_serialized_data_root,
                                table_name,
                                _column_names,
                                column_indices_per_row_group[i],
                                opt_params,
                                stream);
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

output_column_builder::output_column_builder(cudf::data_type cudf_type,
                                             std::shared_ptr<pruning_results_t> pruning_results,
                                             cudf::size_type column_idx)
  : _cudf_type(cudf_type),
    _pruning_results(std::move(pruning_results)),
    _column_idx(column_idx),
    _num_rows(compute_num_rows(*_pruning_results))
{
}

void output_column_builder::submit_materialization_requests(utility::copy_batch&,
                                                            decompression_batch&,
                                                            rmm::cuda_stream_view)
{
}

std::unique_ptr<cudf::column> concatenating_output_column_builder::make_cudf_column(
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
          throw std::logic_error("Use dedicated compressed_sliced_output_column_builder instead");
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
  std::unique_ptr<cudf::column> cudf_column = cudf::concatenate(
    pruned_columns, concatenation_stream, rmm::mr::get_current_device_resource_ref());

  GQE_LOG_DEBUG(
    "Processed column with concatenate: _column_idx = {}, "
    "cudf_column->size() = {}",
    _column_idx,
    cudf_column->size());
  return cudf_column;
}

namespace {

std::unique_ptr<output_column_builder> make_output_column_builder(
  const cudf::data_type cudf_type,
  const in_memory_column_type column_type,
  std::shared_ptr<pruning_results_t> pruning_results,
  cudf::size_type column_idx,
  const row_group* representative_row_group)
{
  if (column_type == in_memory_column_type::CONTIGUOUS &&
      cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<contiguous_output_column_builder<contiguous_column>>(
      cudf_type, pruning_results, column_idx);
  } else if (column_type == in_memory_column_type::CONTIGUOUS &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<concatenating_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (column_type == in_memory_column_type::COMPRESSED_SLICED &&
             cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<compressed_sliced_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (column_type == in_memory_column_type::COMPRESSED_SLICED &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    auto& string_column =
      get_column<string_compressed_sliced_column_base>(representative_row_group, column_idx);
    if (string_column.is_large_string()) {
      return std::make_unique<string_compressed_sliced_output_column_builder<true>>(
        cudf_type, pruning_results, column_idx);
    } else {
      return std::make_unique<string_compressed_sliced_output_column_builder<false>>(
        cudf_type, pruning_results, column_idx);
    }
  }

  return std::make_unique<concatenating_output_column_builder>(
    cudf_type, pruning_results, column_idx);
}

rmm::device_buffer make_output_buffer(
  size_t buffer_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  return rmm::device_buffer(buffer_size, stream, mr);
}

rmm::device_buffer make_output_buffer(
  size_t num_values,
  size_t value_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  return make_output_buffer(num_values * value_size, stream, mr);
}

void submit_buffer_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch& d_batch,
  std::deque<rmm::device_buffer>& staging_buffers,
  compressed_sliced_column::buffer_view host_source_buffer,
  compression_manager const& compression_manager,
  std::span<size_t const> partition_indexes,
  std::byte* output_ptr,
  rmm::cuda_stream_view stream)
{
  if (partition_indexes.empty()) { return; }

  if (host_source_buffer.is_compressed()) {
    // Each selected partition is copied into staging, then the whole staging buffer is represented
    // by one decompression request.
    size_t staging_buffer_size{0};
    for (size_t partition_idx : partition_indexes) {
      staging_buffer_size +=
        rmm::align_up(host_source_buffer.get_partition(partition_idx).size(), size_t{8});
    }

    staging_buffers.emplace_back(
      staging_buffer_size, stream, rmm::mr::get_current_device_resource_ref());
    auto* device_staging_buffer = &staging_buffers.back();
    auto* staging_ptr           = reinterpret_cast<std::byte*>(device_staging_buffer->data());
    c_batch.reserve(c_batch.size() + partition_indexes.size());
    for (size_t partition_idx : partition_indexes) {
      auto partition = host_source_buffer.get_partition(partition_idx);
      c_batch.add(
        staging_ptr, reinterpret_cast<std::byte const*>(partition.data()), partition.size());
      staging_ptr += rmm::align_up(partition.size(), size_t{8});
    }

    d_batch.add(host_source_buffer,
                &compression_manager,
                std::vector<size_t>(partition_indexes.begin(), partition_indexes.end()),
                device_staging_buffer,
                output_ptr);
    return;
  }

  // Each selected uncompressed partition is copied directly into the output buffer.
  c_batch.reserve(c_batch.size() + partition_indexes.size());

  auto* current_output_ptr = output_ptr;
  for (size_t partition_idx : partition_indexes) {
    auto partition = host_source_buffer.get_partition(partition_idx);
    c_batch.add(
      current_output_ptr, reinterpret_cast<std::byte const*>(partition.data()), partition.size());
    current_output_ptr += partition.size();
  }
}

}  // namespace

template <typename T>
rmm::device_buffer contiguous_output_column_builder<T>::allocate_output_buffer(
  rmm::cuda_stream_view stream)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream);
}

template <typename T>
void contiguous_output_column_builder<T>::submit_materialization_requests(
  utility::copy_batch& c_batch, decompression_batch&, rmm::cuda_stream_view stream)
{
  size_t num_copy_requests{0};
  for (const auto& [/* row_group */ _, pruning_result] : *_pruning_results) {
    num_copy_requests += pruning_result.consolidated_partitions().size();
  }
  c_batch.reserve(c_batch.size() + num_copy_requests);

  _output_buffer        = allocate_output_buffer(stream);
  std::byte* target_ptr = reinterpret_cast<std::byte*>(_output_buffer.data());
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const T& column        = get_column<T>(row_group, _column_idx);
    const auto column_view = column.view();
    auto* source_ptr = reinterpret_cast<std::byte const*>(column_view.template data<std::byte>());
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      const size_t size_in_bytes = (partition.end - partition.start) * cudf::size_of(_cudf_type);
      auto const* partition_source_ptr = source_ptr + partition.start * cudf::size_of(_cudf_type);
      c_batch.add(target_ptr, partition_source_ptr, size_in_bytes);
      target_ptr += size_in_bytes;
    }
  }
}

template <typename T>
std::unique_ptr<cudf::column> contiguous_output_column_builder<T>::make_cudf_column(
  rmm::cuda_stream_view /* concatenation_stream */)
{
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), null_count);
}

rmm::device_buffer compressed_sliced_output_column_builder::allocate_output_buffer(
  rmm::cuda_stream_view stream)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream);
}

void compressed_sliced_output_column_builder::submit_materialization_requests(
  utility::copy_batch& c_batch, decompression_batch& d_batch, rmm::cuda_stream_view stream)
{
  _output_buffer        = allocate_output_buffer(stream);
  std::byte* target_ptr = reinterpret_cast<std::byte*>(_output_buffer.data());
  size_t target_ptr_offset{0};
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    if (pruning_result.partition_size() <= 0) {
      throw std::invalid_argument(
        "Compressed-sliced materialization requires zone_map_partition_size > 0.");
    }
    const auto& column              = get_column<compressed_sliced_column>(row_group, _column_idx);
    auto const data_buffer          = column.data_buffer_view();
    auto const& compression_manager = column.get_compression_manager();
    auto const& partition_indexes   = pruning_result.partition_indexes();
    auto* row_group_target_ptr      = target_ptr + target_ptr_offset;

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _staging_buffers,
                                           data_buffer,
                                           compression_manager,
                                           partition_indexes,
                                           row_group_target_ptr,
                                           stream);

    target_ptr_offset += pruning_result.num_rows() * cudf::size_of(_cudf_type);
  }
}

std::unique_ptr<cudf::column> compressed_sliced_output_column_builder::make_cudf_column(
  rmm::cuda_stream_view /* concatenation_stream */)
{
  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), null_count);
}

template <bool large_string_mode>
std::pair<rmm::device_buffer, rmm::device_buffer>
string_compressed_sliced_output_column_builder<large_string_mode>::allocate_output_buffers(
  size_t char_buffer_size, size_t offset_buffer_size, rmm::cuda_stream_view stream)
{
  return {make_output_buffer(char_buffer_size, stream),
          make_output_buffer(offset_buffer_size, stream)};
}

template <bool large_string_mode>
void string_compressed_sliced_output_column_builder<
  large_string_mode>::submit_materialization_requests(utility::copy_batch& c_batch,
                                                      decompression_batch& d_batch,
                                                      rmm::cuda_stream_view stream)
{
  size_t char_buffer_size{0};
  size_t offset_buffer_size = sizeof(
    typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    if (pruning_result.partition_size() <= 0) {
      throw std::invalid_argument(
        "Compressed-sliced string materialization requires zone_map_partition_size > 0.");
    }
    const auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    for (size_t partition_idx : pruning_result.partition_indexes()) {
      char_buffer_size += column.get_char_partition_output_size(partition_idx);
    }
    offset_buffer_size +=
      pruning_result.num_rows() *
      sizeof(
        typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
  }

  std::tie(_char_buffer, _offset_buffer) =
    allocate_output_buffers(char_buffer_size, offset_buffer_size, stream);
  auto* char_target_ptr   = reinterpret_cast<std::byte*>(_char_buffer.data());
  auto* offset_target_ptr = reinterpret_cast<std::byte*>(_offset_buffer.data());

  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const auto& column =
      get_column<string_compressed_sliced_column<large_string_mode>>(row_group, _column_idx);
    auto const char_buffer          = column.char_buffer_view();
    auto const offset_buffer        = column.offset_buffer_view();
    auto const& compression_manager = column.get_compression_manager();
    auto const& partition_indexes   = pruning_result.partition_indexes();

    size_t char_output_size{0};
    for (size_t partition_idx : partition_indexes) {
      char_output_size += column.get_char_partition_output_size(partition_idx);
    }

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _char_staging_buffers,
                                           char_buffer,
                                           compression_manager,
                                           partition_indexes,
                                           char_target_ptr,
                                           stream);
    char_target_ptr += char_output_size;

    const size_t offset_output_size =
      pruning_result.num_rows() *
      sizeof(
        typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _offset_staging_buffers,
                                           offset_buffer,
                                           compression_manager,
                                           partition_indexes,
                                           offset_target_ptr,
                                           stream);
    offset_target_ptr += offset_output_size;
  }
}

template <bool large_string_mode>
std::unique_ptr<cudf::column>
string_compressed_sliced_output_column_builder<large_string_mode>::make_cudf_column(
  rmm::cuda_stream_view concatenation_stream)
{
  // Adjust offsets
  auto cudf_pinned_resource = cudf::get_pinned_memory_resource();
  size_t num_partitions     = 0;
  for (auto const& [/* row_group */ _, pruning_result] : *_pruning_results) {
    num_partitions += pruning_result.partition_indexes().size();
  }

  rmm::device_uvector<offsets_type> partition_char_offsets_buffer(
    num_partitions, concatenation_stream, cudf_pinned_resource);
  auto* partition_char_offsets = partition_char_offsets_buffer.data();
  rmm::device_uvector<cudf::size_type> partition_row_offsets_buffer(
    num_partitions, concatenation_stream, cudf_pinned_resource);
  auto* partition_row_offsets = partition_row_offsets_buffer.data();

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
  rmm::device_uvector<cudf::size_type> d_partition_row_offsets_buffer(partition_row_offsets_buffer,
                                                                      concatenation_stream);
  auto d_partition_row_offsets = d_partition_row_offsets_buffer.data();

  adjust_offsets_api(reinterpret_cast<offsets_type*>(_offset_buffer.data()),
                     _num_rows,
                     num_partitions,
                     d_partition_row_offsets,
                     partition_char_offsets,
                     _char_buffer.size(),
                     concatenation_stream);
  GQE_LOG_DEBUG(
    "Adjusted offsets; _column_idx = {}, num_partitions = {}, char_offset = {}, row_offset = {}, "
    "_num_rows = {}, _char_buffer.size() = {}",
    _column_idx,
    num_partitions,
    char_offset,
    row_offset,
    _num_rows,
    _char_buffer.size());

  // Create CUDF column
  auto offset_column = std::make_unique<cudf::column>(
    offset_element_type, _num_rows + 1, std::move(_offset_buffer), rmm::device_buffer(), 0);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(1);
  children.push_back(std::move(offset_column));
  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::STRING),
                                        _num_rows,
                                        std::move(_char_buffer),
                                        rmm::device_buffer(),
                                        0,
                                        std::move(children));
}

template class string_compressed_sliced_output_column_builder<false>;
template class string_compressed_sliced_output_column_builder<true>;

output_table_builder::output_table_builder(context_reference ctx_ref,
                                           std::vector<cudf::size_type> column_indexes,
                                           std::vector<cudf::data_type> data_types,
                                           std::unique_ptr<pruning_results_t> pruning_results)
  : _ctx_ref(ctx_ref),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _pruning_results(std::shared_ptr<pruning_results_t>(std::move(pruning_results)))
{
}

std::unique_ptr<cudf::table> output_table_builder::build()
{
  // Select one builder per projected column. The first row group is only used to inspect storage
  // details that must be consistent across row groups, such as string offset width.
  _output_columns.clear();
  _output_columns.reserve(_column_indexes.size());

  const auto* representative_row_group = _pruning_results->front().first;
  for (size_t i = 0; i < _column_indexes.size(); ++i) {
    const auto column_idx             = _column_indexes[i];
    const auto cudf_type              = _data_types[i];
    const auto& representative_column = representative_row_group->get_column(column_idx);
    _output_columns.push_back(make_output_column_builder(cudf_type,
                                                         representative_column.type(),
                                                         _pruning_results,
                                                         column_idx,
                                                         representative_row_group));
  }

  std::unique_lock<std::mutex> overlap_mtx_lock;
  auto default_stream       = cudf::get_default_stream();
  auto copy_stream          = default_stream;
  auto decompression_stream = default_stream;
  auto column_build_stream  = default_stream;

  auto& shared_stream = _ctx_ref._task_manager_context->copy_engine_stream;
  if (_ctx_ref._query_context->parameters.use_overlap_mtx) {
    overlap_mtx_lock     = std::unique_lock{*shared_stream.mtx};
    decompression_stream = shared_stream.stream;
  }

  // Prepare all materialization work first. Builders allocate their output/staging buffers here
  // and enqueue copy/decompression requests, but do not create cuDF columns yet.
  utility::copy_batch c_batch;
  decompression_batch d_batch;
  for (auto& output_column : _output_columns) {
    output_column->submit_materialization_requests(c_batch, d_batch, copy_stream);
  }

  const bool has_compressed_buffers = !d_batch.empty();
  if (_ctx_ref._query_context->parameters.use_overlap_mtx && !has_compressed_buffers) {
    // For contiguous columns, run cudf::concatenate on the shared stream since we do not have
    // batched memcpy for this case.
    column_build_stream = shared_stream.stream;
  }

  // Copy selected contiguous data and compressed partition bytes into their destination/staging
  // buffers. Decompression, if needed, consumes those staging buffers afterward.
  if (!c_batch.empty()) {
    utility::semaphore_acquire_guard guard(
      _ctx_ref._task_manager_context->batched_memcpy_semaphore);
    c_batch.execute(copy_stream,
                    _ctx_ref._query_context->parameters.in_memory_dummy_copy_multiplier);
    copy_stream.synchronize();
  }

  if (has_compressed_buffers) {
    utility::nvtx_scoped_range nvtx_range("Decompressing columns");
    decompression_stream.synchronize();
    utility::semaphore_acquire_guard guard(_ctx_ref._task_manager_context->decompress_semaphore);
    d_batch.execute_async(decompression_stream);
    decompression_stream.synchronize();
  }

  auto cudf_column_view =
    _output_columns | std::views::transform([column_build_stream](auto& output_column) {
      utility::nvtx_scoped_range nvtx_range("Creating cuDF column");
      return output_column->make_cudf_column(column_build_stream);
    });
  std::vector<std::unique_ptr<cudf::column>> cudf_columns(cudf_column_view.begin(),
                                                          cudf_column_view.end());

  cudaEvent_t columns_ready_evt{};
  GQE_CUDA_TRY(cudaEventCreateWithFlags(&columns_ready_evt, cudaEventDisableTiming));
  GQE_CUDA_TRY(cudaEventRecord(columns_ready_evt, column_build_stream.value()));
  GQE_CUDA_TRY(cudaStreamWaitEvent(default_stream.value(), columns_ready_evt));

  if (_ctx_ref._query_context->parameters.use_overlap_mtx) {
    shared_stream.stream.synchronize();
    overlap_mtx_lock.unlock();
  }
  GQE_CUDA_TRY(cudaEventDestroy(columns_ready_evt));

  return std::make_unique<cudf::table>(std::move(cudf_columns));
}

in_memory_read_task::in_memory_read_task(context_reference ctx_ref,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::vector<const row_group*> row_groups,
                                         std::vector<cudf::size_type> column_indexes,
                                         std::vector<cudf::data_type> data_types,
                                         memory_kind::type memory_kind,
                                         std::optional<arrow::compute::Expression> partial_filter,
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
    pruning_results->reserve(_row_groups.size());
    std::for_each(_row_groups.begin(),
                  _row_groups.end(),
                  [this, &pruning_results, &partition_size](const auto* rg) {
                    if (!rg->zone_map()) {
                      throw std::logic_error("Row group should have a zone map but none exists");
                    }
                    const auto partitions = rg->zone_map()->evaluate(*_partial_filter);
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
        // When zone map partition size is disabled
        // (<= 0), compressed_sliced columns use one slice of row-group size. Avoid _partition_size
        // == 0 in pruning_result_t
        const auto const_row_group_rows = static_cast<cudf::size_type>(rg->size());
        const cudf::size_type effective_partition_size =
          partition_size > 0 ? partition_size
                             : std::max(static_cast<cudf::size_type>(1), const_row_group_rows);
        return row_group_with_pruning_result_t{
          rg, pruning_result_t{{partition}, effective_partition_size}};
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
    if (type == in_memory_column_type::COMPRESSED_SLICED) {
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
  empty_columns.reserve(_data_types.size());
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
  columns.reserve(_column_indexes.size());
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
  output_table_builder builder(
    get_context_reference(), _column_indexes, _data_types, std::move(pruning_results));
  auto result = builder.build();
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

  if (_column_indexes.empty()) {
    // No columns projected (e.g. SELECT COUNT(*) FROM t). A 0-column table has
    // num_rows()==0 in cudf, so we must carry the row count via a phantom EMPTY column.
    // Using EMPTY avoids allocating ~1 byte/row (INT8 would); the public cudf::column
    // constructor is the supported way to size an EMPTY column with a zero-size data buffer.
    cudf::size_type total_rows = 0;
    for (auto const& [rg, pr] : *pruning_results)
      total_rows += pr.num_rows();
    std::vector<std::unique_ptr<cudf::column>> phantom;
    phantom.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::EMPTY},
                                                     total_rows,
                                                     rmm::device_buffer{},
                                                     rmm::device_buffer{},
                                                     0));
    emit_result(std::make_unique<cudf::table>(std::move(phantom)));
  } else if (pruning_results->empty()) {
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

in_memory_write_task::in_memory_write_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<task> input,
                                           rmm::device_async_resource_ref non_owned_memory_resource,
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
  auto const configured_comp_format =
    get_query_context()->parameters.in_memory_table_compression_format;
  if (configured_comp_format != compression_format::none && partition_size <= 0) {
    throw std::invalid_argument(
      "Compressed in-memory writes require zone_map_partition_size > 0 because "
      "compressed_sliced_column is the only supported compressed storage layout.");
  }

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

    auto comp_format = configured_comp_format;
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

    if (comp_format == compression_format::none) {
      GQE_LOG_TRACE("Uncompressed column size {}", cudf_column.size());
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<contiguous_column>(std::move(cudf_column));
    } else {
      auto compression_config = compression_configuration{
        .primary_compression_format                 = comp_format,
        .secondary_compression_format               = secondary_comp_format,
        .compression_chunk_size                     = compression_chunk_size,
        .compression_ratio_threshold                = compression_ratio_threshold,
        .secondary_compression_ratio_threshold      = secondary_compression_ratio_threshold,
        .secondary_compression_multiplier_threshold = secondary_compression_multiplier_threshold,
        .use_cpu_compression                        = use_cpu_compression,
        .compression_level                          = compression_level,
        .decompress_backend                         = decompress_backend,
        .cudf_type                                  = cudf_column.type()};
      if (dtype == cudf::type_id::STRING) {
        // Get a string view of the column
        cudf::strings_column_view strings_column_view(cudf_column);
        int64_t chars_size = strings_column_view.chars_size(stream);
        GQE_LOG_TRACE("String column size {}", chars_size);
        if (chars_size > std::numeric_limits<int32_t>::max()) {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<true>>(std::move(cudf_column),
                                                                    partition_size,
                                                                    _memory_kind,
                                                                    std::move(compression_config),
                                                                    stream,
                                                                    _non_owned_memory_resource);
        } else {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<false>>(std::move(cudf_column),
                                                                     partition_size,
                                                                     _memory_kind,
                                                                     std::move(compression_config),
                                                                     stream,
                                                                     _non_owned_memory_resource);
        }
      } else {
        GQE_LOG_TRACE("Compressed sliced column size {}", cudf_column.size());
        new_columns[_column_indexes[column_idx]] =
          std::make_unique<compressed_sliced_column>(std::move(cudf_column),
                                                     partition_size,
                                                     _memory_kind,
                                                     std::move(compression_config),
                                                     stream,
                                                     _non_owned_memory_resource);
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
  auto const comp_format = get_query_context()->parameters.in_memory_table_compression_format;
  if (comp_format != compression_format::none) {
    throw std::invalid_argument(
      "Shared-memory in-memory tables do not support compression. "
      "shared_contiguous_column is the only supported shared-memory storage layout.");
  }

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  // Check if input schema matches output schema
  auto num_columns = _column_indexes.size();

  cudf::table_view input_table;

  // Other ranks will have empty result table
  if (gqe::utility::multi_process::nvshmem_rank_zero()) {
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

  // The type-erased `_non_owned_memory_resource` does not expose the underlying
  // `boost_shared_memory_resource`; reach the typed instance through the side
  // channel kept on the task manager context.
  auto* boost_shared_mr =
    get_context_reference()._task_manager_context->get_boost_shared_resource();
  GQE_EXPECTS(boost_shared_mr != nullptr,
              "boost_shared memory resource has not been initialised on this task manager context");
  auto& segment = boost_shared_mr->segment();

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    std::string shared_column_name = [this, &column_idx] {
      std::ostringstream oss;
      oss << "shared_column_" << stage_id() << "_" << task_id() << "_" << column_idx << "_"
          << _column_names[column_idx];
      return oss.str();
    }();

    cudf::column_view input_column = [&input_table, &column_idx]() {
      if (gqe::utility::multi_process::nvshmem_rank_zero()) {
        return input_table.column(column_idx);
      } else {
        return cudf::column_view{};
      }
    }();

    if (gqe::utility::multi_process::nvshmem_rank_zero()) {
      segment.construct<gqe::storage::shared_column>(shared_column_name.c_str())(
        input_column, segment, stream, _non_owned_memory_resource);
    }
    new_columns[_column_indexes[column_idx]] =
      std::make_unique<shared_contiguous_column>(shared_column_name, segment);
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

    if (gqe::utility::multi_process::nvshmem_rank_zero()) {
      segment.construct<gqe::shared_zone_map_table>(shared_table_name.c_str())(
        input_table, partition_size, &segment);
    }

    zone_map = std::make_unique<gqe::shared_zone_map>(partition_size, shared_table_name, &segment);
  }

  if (gqe::utility::multi_process::nvshmem_rank_zero()) {
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

      // Create a new read task
      auto read_task = std::make_unique<in_memory_read_task>(ctx_ref,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(row_groups_chunk),
                                                             column_indexes,
                                                             data_types,
                                                             _non_owning_table->_memory_kind,
                                                             zone_map_filter,
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
}

std::optional<arrow::compute::Expression> in_memory_readable_view::transform_partial_filter(
  gqe::expression* partial_filter)
{
  if (!partial_filter) { return std::nullopt; }
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  if (zone_map_filter) {
    GQE_LOG_DEBUG(
      "Using partial filter to prune results\nPartial filter:\n{}\nZone map filter:\n{}",
      expression_json_formatter::to_json(*partial_filter),
      zone_map_filter->ToString());
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
    auto memory_resource = _non_owning_table->_task_manager_context->get_table_memory_resource(
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
