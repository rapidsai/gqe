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

#include "in_memory_common.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/chunked_in_memory_writer.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/storage/in_memory_readable_view.hpp>
#include <gqe/storage/oneshot_in_memory_reader.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
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
#include <cudf/column/column_stream.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
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

#include <nvcomp.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock, std::scoped_lock
#include <optional>
#include <ranges>
#include <shared_mutex>  // std::shared_lock
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

namespace storage {

// Get the size of a string column in bytes, can only be called on string columns. Returns a pair of
// (chars_size, offsets_size).
template <typename ColumnStorage>
std::pair<int64_t, int64_t> get_string_column_size(ColumnStorage const& column)
{
  if (column.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("get_string_column_size can only be called on string columns");
  }
  if (column.num_children() < 1) {
    // cudf::make_empty_column(STRING) has no offsets child. Its character and offset buffers are
    // both empty, so avoid accessing child(0).
    if (column.size() == 0) { return {0, 0}; }
    throw std::runtime_error("String column storage must contain an offsets child");
  }
  auto const& offsets  = column.child(0);
  int64_t chars_size   = static_cast<int64_t>(column.data_size());
  int64_t offsets_size = static_cast<int64_t>(offsets.data_size());
  return std::make_pair(chars_size, offsets_size);
}

// Get the size of a column in bytes, string column includes both chars and offsets size.
template <typename ColumnStorage>
int64_t get_column_size(ColumnStorage const& column)
{
  auto const type = column.type();
  if (type.id() == cudf::type_id::STRING) {
    auto [chars_size, offsets_size] = get_string_column_size(column);
    return chars_size + offsets_size;
  }
  return static_cast<int64_t>(column.data_size());
}

const char* format_as(const column_layout layout)
{
  switch (layout) {
    case column_layout::contiguous: return "contiguous";
    case column_layout::compressed_sliced: return "compressed_sliced";
    case column_layout::string_compressed_sliced: return "string_compressed_sliced";
    case column_layout::large_string_compressed_sliced: return "large_string_compressed_sliced";
  }
  // Throw an exception so this will get fixed quickly when we add a new type.
  throw std::runtime_error("column_layout not supported for log formatting");
}

// Cast a column from a row group to a desired type
template <typename T>
const T& get_column(const row_group_interface* row_group, size_t column_idx)
{
  const auto& column_base = row_group->get_column(column_idx);
  const T& column         = static_cast<const T&>(column_base);
  return column;
}

// Common implementation for local and shared contiguous column wrappers.

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::size() const
{
  assert(static_cast<std::size_t>(_storage_handle.get().size()) <=
         std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(_storage_handle.get().size());
}

template <class StorageHandle>
cudf::size_type basic_contiguous_column<StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <class StorageHandle>
bool basic_contiguous_column<StorageHandle>::is_compressed() const
{
  return false;
}

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::get_compressed_size() const
{
  return get_uncompressed_size();
}

template <class StorageHandle>
int64_t basic_contiguous_column<StorageHandle>::get_uncompressed_size() const
{
  return get_column_size(_storage_handle.get());
}

template <class StorageHandle>
column_compression_statistics basic_contiguous_column<StorageHandle>::get_compression_stats() const
{
  auto const& column = _storage_handle.get();
  if (column.type().id() == cudf::type_id::STRING) {
    string_compression_statistics string_stats;
    auto [chars_size, offsets_size]              = get_string_column_size(column);
    string_stats.offsets_stats.compressed_size   = offsets_size;
    string_stats.offsets_stats.uncompressed_size = offsets_size;
    string_stats.chars_stats.compressed_size     = chars_size;
    string_stats.chars_stats.uncompressed_size   = chars_size;

    return column_compression_statistics(string_stats);
  }

  fixed_width_compression_statistics fixed_width_stats;
  fixed_width_stats.compressed_size           = get_column_size(column);
  fixed_width_stats.uncompressed_size         = fixed_width_stats.compressed_size;
  fixed_width_stats.primary_compressed_size   = fixed_width_stats.compressed_size;
  fixed_width_stats.secondary_compressed_size = fixed_width_stats.uncompressed_size;

  return column_compression_statistics(fixed_width_stats);
}

template <class StorageHandle>
cudf::column_view basic_contiguous_column<StorageHandle>::view() const
{
  return _storage_handle.get().view();
}

template <class StorageHandle>
void basic_contiguous_column<StorageHandle>::deallocate(cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

// Common implementation for local and shared compressed-sliced column wrappers.

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::size() const
{
  return _storage_handle.get().size();
}

template <class StorageHandle>
cudf::size_type basic_compressed_sliced_column<StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <class StorageHandle>
bool basic_compressed_sliced_column<StorageHandle>::is_compressed() const
{
  return _storage_handle.get().is_compressed();
}

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::get_compressed_size() const
{
  return _storage_handle.get().get_compressed_size();
}

template <class StorageHandle>
int64_t basic_compressed_sliced_column<StorageHandle>::get_uncompressed_size() const
{
  return _storage_handle.get().get_uncompressed_size();
}

template <class StorageHandle>
column_compression_statistics basic_compressed_sliced_column<StorageHandle>::get_compression_stats()
  const
{
  return _storage_handle.get().get_compression_stats();
}

template <class StorageHandle>
typename basic_compressed_sliced_column<StorageHandle>::buffer_view
basic_compressed_sliced_column<StorageHandle>::data_buffer_view() const
{
  return _storage_handle.get().data_buffer_view();
}

template <class StorageHandle>
gqe::compression::configuration const&
basic_compressed_sliced_column<StorageHandle>::get_compression_config() const
{
  return _storage_handle.get().get_compression_config();
}

template <class StorageHandle>
void basic_compressed_sliced_column<StorageHandle>::deallocate(cuda::stream_ref stream,
                                                               rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

// Common implementation for local and shared string compressed-sliced column wrappers.

template <bool large_string_mode, class StorageHandle>
column_layout basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::layout()
  const
{
  if constexpr (large_string_mode) {
    return column_layout::large_string_compressed_sliced;
  } else {
    return column_layout::string_compressed_sliced;
  }
}

template <bool large_string_mode, class StorageHandle>
int64_t basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::size() const
{
  return _storage_handle.get().size();
}

template <bool large_string_mode, class StorageHandle>
cudf::size_type
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::null_count() const
{
  return _storage_handle.get().null_count();
}

template <bool large_string_mode, class StorageHandle>
bool basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::is_compressed() const
{
  return _storage_handle.get().is_compressed();
}

template <bool large_string_mode, class StorageHandle>
int64_t
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compressed_size() const
{
  return _storage_handle.get().get_compressed_size();
}

template <bool large_string_mode, class StorageHandle>
int64_t basic_string_compressed_sliced_column<large_string_mode,
                                              StorageHandle>::get_uncompressed_size() const
{
  return _storage_handle.get().get_uncompressed_size();
}

template <bool large_string_mode, class StorageHandle>
column_compression_statistics
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compression_stats()
  const
{
  return _storage_handle.get().get_compression_stats();
}

template <bool large_string_mode, class StorageHandle>
bool basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::is_large_string()
  const
{
  return _storage_handle.get().is_large_string();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::data_buffer_view() const
{
  return _storage_handle.get().data_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::char_buffer_view() const
{
  return data_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
typename basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::buffer_view
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::offset_buffer_view() const
{
  return _storage_handle.get().offset_buffer_view();
}

template <bool large_string_mode, class StorageHandle>
size_t basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::
  get_char_partition_output_size(size_t partition_idx) const
{
  return _storage_handle.get().get_char_partition_output_size(partition_idx);
}

template <bool large_string_mode, class StorageHandle>
gqe::compression::configuration const&
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::get_compression_config()
  const
{
  return _storage_handle.get().get_compression_config();
}

template <bool large_string_mode, class StorageHandle>
void basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::deallocate(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  _storage_handle.deallocate(stream, mr);
}

template <class StorageHandle>
void basic_compressed_sliced_column<StorageHandle>::serialize_to_disk(
  std::string const& column_name,
  std::filesystem::path const& file_path,
  cuda::stream_ref stream) const
{
  _storage_handle.get().serialize_to_disk(column_name, file_path, stream);
}

template <bool large_string_mode, class StorageHandle>
void basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::serialize_to_disk(
  std::string const& column_name,
  std::filesystem::path const& file_path,
  cuda::stream_ref stream) const
{
  _storage_handle.get().serialize_to_disk(column_name, file_path, stream);
}

// Recompress (secondary-compression-on-load): the storage decompresses and re-compresses its
// buffers into a fresh storage object, and the wrapper adopts it as a new local column.
template <class StorageHandle>
std::unique_ptr<compressed_sliced_column_interface>
basic_compressed_sliced_column<StorageHandle>::recompress_buffers(
  gqe::compression::configuration target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  // Only the local instantiation is ever called. The shared branch exists solely to compile.
  if constexpr (std::is_same_v<storage_type, compressed_sliced_column_storage<>>) {
    return std::make_unique<compressed_sliced_column>(
      _storage_handle.get().recompress(target_config, memory_kind, stream, mr));
  } else {
    throw std::logic_error("recompress_buffers is only supported for local (non-shared) storage");
  }
}

template <bool large_string_mode, class StorageHandle>
std::unique_ptr<compressed_sliced_column_interface>
basic_string_compressed_sliced_column<large_string_mode, StorageHandle>::recompress_buffers(
  gqe::compression::configuration target_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  // Only the local instantiation is ever called. The shared branch exists solely to compile.
  if constexpr (std::is_same_v<storage_type,
                               string_compressed_sliced_column_storage<large_string_mode>>) {
    return std::make_unique<string_compressed_sliced_column<large_string_mode>>(
      _storage_handle.get().recompress(target_config, memory_kind, stream, mr));
  } else {
    throw std::logic_error("recompress_buffers is only supported for local (non-shared) storage");
  }
}

// Explicit instantiations for the storage handles and wrapper implementations

template class local_storage_handle<contiguous_column_storage>;
template class shared_storage_handle<shared_contiguous_column_storage>;
template class local_storage_handle<compressed_sliced_column_storage<>>;
template class shared_storage_handle<shared_compressed_sliced_column_storage>;
template class local_storage_handle<string_compressed_sliced_column_storage<false>>;
template class local_storage_handle<string_compressed_sliced_column_storage<true>>;
template class shared_storage_handle<shared_string_compressed_sliced_column_storage<false>>;
template class shared_storage_handle<shared_string_compressed_sliced_column_storage<true>>;

template class basic_contiguous_column<local_storage_handle<contiguous_column_storage>>;
template class basic_contiguous_column<shared_storage_handle<shared_contiguous_column_storage>>;
template class basic_compressed_sliced_column<
  local_storage_handle<compressed_sliced_column_storage<>>>;
template class basic_compressed_sliced_column<
  shared_storage_handle<shared_compressed_sliced_column_storage>>;
template class basic_string_compressed_sliced_column<
  false,
  local_storage_handle<string_compressed_sliced_column_storage<false>>>;
template class basic_string_compressed_sliced_column<
  true,
  local_storage_handle<string_compressed_sliced_column_storage<true>>>;
template class basic_string_compressed_sliced_column<
  false,
  shared_storage_handle<shared_string_compressed_sliced_column_storage<false>>>;
template class basic_string_compressed_sliced_column<
  true,
  shared_storage_handle<shared_string_compressed_sliced_column_storage<true>>>;

column_storage_handle::column_storage_handle(column_layout layout_,
                                             shared_memory_handle_t storage_handle_)
  : layout(layout_), storage_handle(storage_handle_)
{
}

namespace {

/**
 * @brief Validate a handle against the segment and resolve it to `Storage`.
 *
 * Checks that the handle lies inside the segment, that the address is suitably aligned, and that
 * the magic number every shared storage type keeps as its first data member matches, before
 * casting the address. Reading the magic assumes that first-member placement, which the storage
 * constructors verify.
 *
 * @param[in] expected_magic_number Magic number the caller expects to find at the handle.
 * @param[in] description Storage kind, used to build the exception messages.
 * @param[in] retired_magic_number Magic number marking storage that is being torn down, when the
 * storage type has one. Finding it is reported separately because it names the likely cause: use
 * of a handle to storage the owner has already started releasing.
 */
template <typename Storage>
Storage* resolve_shared_storage(shared_segment& segment,
                                shared_memory_handle_t handle,
                                std::uint64_t expected_magic_number,
                                std::string_view description,
                                std::optional<std::uint64_t> retired_magic_number = std::nullopt)
{
  auto invalid = [description](std::string_view problem) {
    return std::invalid_argument(std::format("{} handle {}", description, problem));
  };

  if (handle <= shared_memory_handle_t{}) { throw invalid("is outside the segment"); }

  auto const offset       = static_cast<std::uintmax_t>(handle);
  auto const segment_size = static_cast<std::uintmax_t>(segment.get_size());
  if (offset > segment_size || sizeof(Storage) > segment_size - offset) {
    throw invalid("is outside the segment");
  }

  auto* address = segment.get_address_from_handle(handle);
  if (reinterpret_cast<std::uintptr_t>(address) % alignof(Storage) != 0) {
    throw invalid("is improperly aligned");
  }

  std::uint64_t magic_number;
  std::memcpy(&magic_number, address, sizeof(magic_number));
  if (magic_number != expected_magic_number) {
    if (retired_magic_number && magic_number == *retired_magic_number) {
      throw invalid("refers to storage that is being deallocated");
    }
    throw invalid("has an invalid magic number");
  }

  return static_cast<Storage*>(address);
}

/**
 * @brief Resolve a row-group storage handle, validating the segment bounds, alignment, and magic.
 *
 * @param[in] expected_state Storage state the caller expects. Only cleanup paths, which release
 * storage already detached from its table, pass `retired`.
 */
shared_row_group_storage* resolve_shared_row_group_storage(
  shared_segment& segment,
  shared_memory_handle_t handle,
  row_group_storage_state expected_state = row_group_storage_state::live)
{
  return resolve_shared_storage<shared_row_group_storage>(
    segment,
    handle,
    expected_state == row_group_storage_state::live
      ? shared_row_group_storage::shared_row_group_storage_magic_number
      : shared_row_group_storage::shared_row_group_storage_retired_magic_number,
    "Shared row-group storage");
}

/**
 * @brief Stamp a row group's storage as retired so stale handles no longer resolve as live.
 *
 * Called while the table mutex is held, immediately after the handle is dropped from the table, so
 * the storage never reads as valid once it is unreachable through the table. Failures are logged
 * and contained: the handle has already been detached and cleanup still owns the storage.
 */
void mark_row_group_storage_retired(shared_segment& segment, shared_memory_handle_t handle) noexcept
{
  try {
    auto* storage_ptr = resolve_shared_row_group_storage(segment, handle);
    storage_ptr->magic_number =
      shared_row_group_storage::shared_row_group_storage_retired_magic_number;
  } catch (std::exception const& error) {
    GQE_LOG_ERROR("Failed to mark detached shared row group as retired: {}", error.what());
  } catch (...) {
    GQE_LOG_ERROR("Failed to mark detached shared row group as retired: unknown error");
  }
}

/** @brief Resolve a table storage handle, validating the segment bounds, alignment, and magic. */
shared_in_memory_table_storage* resolve_shared_in_memory_table_storage(
  shared_segment& segment, shared_memory_handle_t handle)
{
  return resolve_shared_storage<shared_in_memory_table_storage>(
    segment,
    handle,
    shared_in_memory_table_storage::shared_in_memory_table_storage_magic_number,
    "Shared table storage",
    shared_in_memory_table_storage::shared_in_memory_table_storage_retired_magic_number);
}

}  // namespace

shared_row_group_storage::shared_row_group_storage(cudf::size_type num_rows_,
                                                   allocator_type const& alloc)
  : num_rows(num_rows_), columns(rebind_allocator<column_storage_handle>(alloc))
{
  if (static_cast<void const*>(this) != static_cast<void const*>(&magic_number)) {
    throw std::logic_error("Shared row-group storage magic number must be the first data member");
  }
}

shared_in_memory_table_storage::shared_in_memory_table_storage(allocator_type const& alloc)
  : _row_group_storage_handles(rebind_allocator<shared_memory_handle_t>(alloc))
{
  if (static_cast<void const*>(this) != static_cast<void const*>(&_magic_number)) {
    throw std::logic_error("Shared table storage magic number must be the first data member");
  }
}

void shared_in_memory_table_storage::throw_if_retired() const
{
  if (_magic_number != shared_in_memory_table_storage_magic_number) {
    throw std::invalid_argument("Shared table storage is being deallocated");
  }
}

shared_memory_handle_t shared_in_memory_table_storage::append(shared_segment& segment,
                                                              shared_row_group_storage&& row_group)
{
  auto guard = lock_shared_memory_mutex(_mutex, "shared_in_memory_table_storage::append");
  throw_if_retired();
  auto* storage_ptr = segment.template construct<shared_row_group_storage>(
    boost::interprocess::anonymous_instance)(std::move(row_group));
  try {
    auto const handle = segment.get_handle_from_address(storage_ptr);
    _row_group_storage_handles.push_back(handle);
    ++_storage_revision;
    return handle;
  } catch (...) {
    segment.destroy_ptr(storage_ptr);
    throw;
  }
}

shared_in_memory_table::row_group_appender::row_group_appender(shared_segment* non_owning_segment,
                                                               shared_memory_handle_t table_handle)
  : _non_owning_segment(non_owning_segment), _table_handle(table_handle)
{
}

void shared_in_memory_table::row_group_appender::operator()(shared_row_group_storage&& row_group)
{
  if (_non_owning_segment == nullptr || _table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row-group appender is not initialized");
  }
  auto* table_storage = resolve_shared_in_memory_table_storage(*_non_owning_segment, _table_handle);
  table_storage->append(*_non_owning_segment, std::move(row_group));
}

shared_segment& shared_in_memory_table::row_group_appender::segment() const
{
  if (_non_owning_segment == nullptr) {
    throw std::logic_error("Shared row-group appender segment is not initialized");
  }
  return *_non_owning_segment;
}

contiguous_column::contiguous_column(cudf::column&& cudf_column)
  : contiguous_column(std::move(cudf_column),
                      cudf::get_default_stream(),
                      rmm::mr::get_current_device_resource_ref())
{
}

contiguous_column::contiguous_column(cudf::column&& cudf_column,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr,
                                     storage_type::allocator_type const& alloc)
  : base_type(cudf_column.view(), stream, mr, alloc)
{
}

cudf::mutable_column_view contiguous_column::mutable_view()
{
  return _storage_handle.get().mutable_view();
}

row_group::row_group(cudf::size_type num_rows,
                     std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map)
  : _num_rows(num_rows), _columns(std::move(columns)), _zone_map(std::move(zone_map))
{
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map)
  : row_group(columns.empty() ? 0 : static_cast<cudf::size_type>(columns.front()->size()),
              std::move(columns),
              std::move(zone_map))
{
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns)
  : row_group(std::move(columns), nullptr)
{
}

int64_t row_group::size() const { return _num_rows; }

int64_t row_group::num_columns() const { return _columns.size(); }

column_base& row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

gqe::zone_map* row_group::zone_map() const { return _zone_map.get(); }

void row_group::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  for (auto& column : _columns) {
    column->deallocate(stream, mr);
  }
  _columns.clear();
  if (_zone_map) { _zone_map->deallocate(stream, mr); }
  _zone_map.reset();
  _num_rows = 0;
}

shared_row_group::shared_row_group(shared_segment& segment,
                                   shared_memory_handle_t shared_row_group_storage_handle)
  : shared_row_group(segment, shared_row_group_storage_handle, row_group_storage_state::live)
{
}

shared_row_group::shared_row_group(shared_segment& segment,
                                   shared_memory_handle_t shared_row_group_storage_handle,
                                   row_group_storage_state state)
  : _segment(&segment),
    _shared_row_group_storage_handle(shared_row_group_storage_handle),
    _state(state)
{
  auto const& shared_storage = storage();
  _columns.reserve(shared_storage.columns.size());
  for (auto const& column : shared_storage.columns) {
    switch (column.layout) {
      case column_layout::contiguous:
        _columns.push_back(
          std::make_unique<shared_contiguous_column>(*_segment, column.storage_handle));
        break;
      case column_layout::compressed_sliced:
        _columns.push_back(
          std::make_unique<shared_compressed_sliced_column>(*_segment, column.storage_handle));
        break;
      case column_layout::string_compressed_sliced:
        _columns.push_back(std::make_unique<shared_string_compressed_sliced_column<false>>(
          *_segment, column.storage_handle));
        break;
      case column_layout::large_string_compressed_sliced:
        _columns.push_back(std::make_unique<shared_string_compressed_sliced_column<true>>(
          *_segment, column.storage_handle));
        break;
      default: throw std::logic_error("Unsupported shared column storage layout");
    }
  }
  if (shared_storage.zone_map) {
    _zone_map = std::make_unique<gqe::shared_zone_map>(*_segment, *shared_storage.zone_map);
  }
}

shared_row_group_storage const& shared_row_group::storage() const
{
  if (_segment == nullptr || _shared_row_group_storage_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row group storage is not initialized");
  }
  return *resolve_shared_row_group_storage(*_segment, _shared_row_group_storage_handle, _state);
}

int64_t shared_row_group::size() const { return storage().num_rows; }

int64_t shared_row_group::num_columns() const { return _columns.size(); }

column_base& shared_row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

gqe::zone_map* shared_row_group::zone_map() const { return _zone_map.get(); }

void shared_row_group::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  if (_segment == nullptr || _shared_row_group_storage_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared row group storage is not initialized");
  }

  auto* row_group_storage =
    resolve_shared_row_group_storage(*_segment, _shared_row_group_storage_handle, _state);

  for (auto& column : _columns) {
    column->deallocate(stream, mr);
  }
  _columns.clear();
  if (_zone_map) { _zone_map->deallocate(stream, mr); }
  _zone_map.reset();

  // Zero the magic before the object is destroyed so this path does not deliberately leave an
  // accepted magic value in the block. What freed segment memory ends up holding is not
  // guaranteed, so this narrows the window for a surviving handle rather than closing it.
  row_group_storage->magic_number = 0;
  _segment->destroy_ptr(row_group_storage);
  _segment                         = nullptr;
  _shared_row_group_storage_handle = shared_memory_handle_t{};
}

in_memory_table::in_memory_table(memory_kind::type memory_kind,
                                 std::vector<std::string> const& column_names,
                                 std::vector<cudf::data_type> const& column_types,
                                 task_manager_context* ctx)
  : in_memory_table_interface(),
    _task_manager_context(ctx),
    _memory_kind(memory_kind),
    _column_name_to_index(),
    _column_names(column_names),
    _column_types(column_types),
    _row_groups(),
    _row_group_latch()
{
  if (!_task_manager_context) {
    throw std::invalid_argument("in_memory_table requires a non-null task_manager_context");
  }

  // Populate column name-to-index map
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }
}

in_memory_table::~in_memory_table() = default;

void in_memory_table::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::unique_lock latch_guard(_row_group_latch);
  for (auto& row_group : _row_groups) {
    row_group.deallocate(stream, mr);
  }
  _row_groups.clear();
}

shared_in_memory_table::shared_in_memory_table(shared_segment* non_owning_segment,
                                               std::vector<std::string> const& column_names,
                                               std::vector<cudf::data_type> const& column_types,
                                               shared_memory_handle_t table_handle)
  : in_memory_table_interface(),
    _memory_kind(memory_kind::shared_segment{}),
    _column_types(column_types),
    _row_groups(),
    _row_group_latch(),
    _shared_segment(non_owning_segment),
    _shared_table_handle(table_handle)
{
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }
  if (_shared_segment == nullptr) {
    throw std::invalid_argument("Cannot attach shared in-memory table without segment");
  }
  if (_shared_table_handle == shared_memory_handle_t{}) {
    throw std::invalid_argument("Cannot attach shared in-memory table without table handle");
  }
  reattach_row_groups();
}

void shared_in_memory_table::reattach_row_groups()
{
  std::vector<shared_memory_handle_t> row_group_storage_handles;
  {
    auto const& table_storage = storage();
    auto storage_guard =
      lock_shared_memory_mutex(table_storage._mutex, "shared_in_memory_table::reattach_row_groups");
    // Resolution above happens before the lock, so attaching can be ordered after a teardown that
    // began in between. Re-check here rather than snapshotting handles whose row groups are being
    // released; the wrappers built below would otherwise read freed columns.
    table_storage.throw_if_retired();
    row_group_storage_handles.assign(table_storage._row_group_storage_handles.begin(),
                                     table_storage._row_group_storage_handles.end());
  }

  std::unique_lock facade_guard(_row_group_latch);
  _row_groups.clear();
  for (auto const shared_row_group_storage_handle : row_group_storage_handles) {
    _row_groups.emplace_back(*_shared_segment, shared_row_group_storage_handle);
  }
}

void shared_in_memory_table::deallocate(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::vector<shared_memory_handle_t> row_group_storage_handles;
  auto& storage_ref = storage();
  {
    auto storage_guard =
      lock_shared_memory_mutex(storage_ref._mutex, "shared_in_memory_table::deallocate");
    // A second facade over the same table would otherwise snapshot these handles again and release
    // every row group twice.
    storage_ref.throw_if_retired();
    row_group_storage_handles.assign(storage_ref._row_group_storage_handles.begin(),
                                     storage_ref._row_group_storage_handles.end());
    // Mark the storage before releasing anything it points at: the row-group teardown below is not
    // instantaneous, and a handle resolved during it would otherwise hand out a table whose row
    // groups are being freed. The mark is set under the mutex and every mutator re-checks it under
    // the same mutex, so no publication can commit against the handles snapshotted just above.
    storage_ref._magic_number =
      shared_in_memory_table_storage::shared_in_memory_table_storage_retired_magic_number;
    // Retire the row groups under the same lock, so a reader that resolved the table before this
    // teardown cannot resolve one of its row groups while the columns below are being freed.
    for (auto const handle : row_group_storage_handles) {
      mark_row_group_storage_retired(*_shared_segment, handle);
    }
  }
  {
    std::unique_lock local_lock(_row_group_latch);
    _row_groups.clear();
  }

  for (auto const handle : row_group_storage_handles) {
    shared_row_group row_group(*_shared_segment, handle, row_group_storage_state::retired);
    row_group.deallocate(stream, mr);
  }

  // The retired magic stamped above is the last value written here, and no resolution path accepts
  // it, so destruction leaves no accepted magic behind for a surviving handle.
  _shared_segment->destroy_ptr(&storage_ref);
  _shared_table_handle = shared_memory_handle_t{};
}

std::vector<shared_memory_handle_t> shared_in_memory_table::replace_row_groups(
  shared_row_group_storage&& new_row_group)
{
  auto& storage_ref = storage();

  // Construct the replacement row-group storage first so that a failure below leaves the table
  // unchanged (strong exception guarantee).
  auto* new_storage_ptr = _shared_segment->construct<shared_row_group_storage>(
    boost::interprocess::anonymous_instance)(std::move(new_row_group));

  std::vector<shared_memory_handle_t> old_handles;
  try {
    auto const new_handle = _shared_segment->get_handle_from_address(new_storage_ptr);

    auto guard =
      lock_shared_memory_mutex(storage_ref._mutex, "shared_in_memory_table::replace_row_groups");
    // Checked under the lock: the catch below releases the replacement storage constructed above.
    storage_ref.throw_if_retired();
    // Reserve before the destructive clear/push_back so the push_back cannot reallocate (and thus
    // cannot throw) after we have already dropped the old handles.
    storage_ref._row_group_storage_handles.reserve(1);
    old_handles.assign(storage_ref._row_group_storage_handles.begin(),
                       storage_ref._row_group_storage_handles.end());
    storage_ref._row_group_storage_handles.clear();
    storage_ref._row_group_storage_handles.push_back(new_handle);
    ++storage_ref._storage_revision;
    // Stamp the detached storage before releasing the mutex so a stale handle can never resolve as
    // a live row group, even if cleanup is delayed or fails.
    for (auto const handle : old_handles) {
      mark_row_group_storage_retired(*_shared_segment, handle);
    }
  } catch (...) {
    _shared_segment->destroy_ptr(new_storage_ptr);
    throw;
  }

  return old_handles;
}

void shared_in_memory_table::cleanup_detached_row_groups(
  std::vector<shared_memory_handle_t> const& handles,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) noexcept
{
  // Drop stale local wrappers before destroying the shared storage they reference. If locking
  // fails, retain both the wrappers and detached storage rather than leaving dangling wrappers.
  try {
    std::unique_lock local_lock(_row_group_latch);
    _row_groups.clear();
  } catch (std::exception const& error) {
    GQE_LOG_ERROR("Failed to drop local row-group wrappers after publication: {}", error.what());
    return;
  } catch (...) {
    GQE_LOG_ERROR("Failed to drop local row-group wrappers after publication: unknown error");
    return;
  }

  for (auto const handle : handles) {
    try {
      shared_row_group row_group(*_shared_segment, handle, row_group_storage_state::retired);
      row_group.deallocate(stream, mr);
    } catch (std::exception const& error) {
      record_if_shared_memory_unrecoverable(std::current_exception(),
                                            "shared_in_memory_table::cleanup_detached_row_groups");
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: {}", error.what());
    } catch (...) {
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: unknown error");
    }
  }
}

void shared_in_memory_table::clear_row_groups(cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto& storage_ref = storage();

  std::vector<shared_memory_handle_t> old_handles;
  {
    auto guard =
      lock_shared_memory_mutex(storage_ref._mutex, "shared_in_memory_table::clear_row_groups");
    storage_ref.throw_if_retired();
    old_handles.assign(storage_ref._row_group_storage_handles.begin(),
                       storage_ref._row_group_storage_handles.end());
    storage_ref._row_group_storage_handles.clear();
    ++storage_ref._storage_revision;
    // See replace_row_groups(): retire the detached storage under the lock that unpublished it.
    for (auto const handle : old_handles) {
      mark_row_group_storage_retired(*_shared_segment, handle);
    }
  }

  cleanup_detached_row_groups(old_handles, stream, mr);
}

void shared_in_memory_table::replace_contents(cudf::table_view input_table,
                                              rmm::device_async_resource_ref memory_resource,
                                              cuda::stream_ref stream)
{
  // A zero-row (or zero-column) input leaves the table empty: a zero-row column cannot be stored
  // as a shared column.
  if (input_table.num_columns() == 0 || input_table.num_rows() == 0) {
    clear_row_groups(stream, memory_resource);
    return;
  }

  if (static_cast<std::size_t>(input_table.num_columns()) != _column_types.size()) {
    throw std::invalid_argument(
      "shared_in_memory_table::replace_contents: input column count does not match the table "
      "schema");
  }

  auto& segment = *_shared_segment;
  shared_memory_allocator alloc(segment.get_segment_manager());
  shared_row_group_storage new_row_group(input_table.num_rows(), alloc);
  new_row_group.columns.reserve(input_table.num_columns());

  // Until replace_row_groups() takes ownership, the shared column storage constructed below lives
  // in the segment referenced only by handle. Track the handles so that any failure before the
  // publish completes (column construction, stream sync, or the publish itself) releases them
  // rather than leaking shared/device memory.
  std::vector<shared_memory_handle_t> staged_handles;
  staged_handles.reserve(input_table.num_columns());

  try {
    // Statistics tables are tiny and uncompressed: every column is stored contiguously and no zone
    // map is built (they are never predicate-pruned). Rank 0 constructs the backing storage in the
    // segment; the copies are asynchronous on `stream` and synchronized before publishing.
    for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
      auto* column_storage = segment.construct<shared_contiguous_column_storage>(
        boost::interprocess::anonymous_instance)(
        input_table.column(column_idx), stream, memory_resource, alloc);
      auto const handle = segment.get_handle_from_address(column_storage);
      new_row_group.columns.emplace_back(column_layout::contiguous, handle);
      staged_handles.push_back(handle);
    }

    stream.sync();
    auto const old_handles = replace_row_groups(std::move(new_row_group));

    // Publication transferred ownership of the staged columns to shared metadata. Release the
    // rollback list before cleanup of the detached old row groups.
    staged_handles.clear();
    cleanup_detached_row_groups(old_handles, stream, memory_resource);
  } catch (...) {
    // replace_row_groups only throws before publication, so these columns remain caller-owned.
    for (auto const handle : staged_handles) {
      shared_contiguous_column staged(segment, handle);
      staged.deallocate(stream, memory_resource);
    }
    throw;
  }
}

shared_in_memory_table_storage const& shared_in_memory_table::storage() const
{
  if (_shared_segment == nullptr || _shared_table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared in-memory table storage is not initialized");
  }
  return *resolve_shared_in_memory_table_storage(*_shared_segment, _shared_table_handle);
}

shared_in_memory_table_storage& shared_in_memory_table::storage()
{
  if (_shared_segment == nullptr || _shared_table_handle == shared_memory_handle_t{}) {
    throw std::logic_error("Shared in-memory table storage is not initialized");
  }
  return *resolve_shared_in_memory_table_storage(*_shared_segment, _shared_table_handle);
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

/**
 * @brief Validate a serialized-table root and return its `rg-<n>` directories in row-group order.
 *
 * Callers run this before mutating the table, so a malformed snapshot is rejected while the
 * existing contents are still intact.
 *
 * @throw std::invalid_argument if @p root is not a directory, holds no row-group directories, or
 *        holds one that is not named `rg-<non-negative-integer>`.
 */
[[nodiscard]] std::vector<std::filesystem::path> sorted_row_group_directories(
  std::filesystem::path const& root)
{
  if (!std::filesystem::is_directory(root)) {
    throw std::invalid_argument("deserialize: path is not a directory: " + root.string());
  }

  std::vector<std::filesystem::path> rg_dirs;
  for (auto const& entry : std::filesystem::directory_iterator(root)) {
    if (entry.is_directory() && entry.path().filename().string().starts_with("rg-")) {
      rg_dirs.push_back(entry.path());
    }
  }
  if (rg_dirs.empty()) {
    throw std::invalid_argument("deserialize: no row-group directories under '" + root.string() +
                                "'");
  }

  try {
    std::ranges::sort(rg_dirs, {}, row_group_directory_index);
  } catch (std::invalid_argument const& e) {
    throw std::invalid_argument("Invalid row-group directory under '" + root.string() +
                                "': " + e.what());
  }
  return rg_dirs;
}

/** Column indices to serialize; computed before any disk I/O. */
[[nodiscard]] std::vector<std::vector<size_t>> serializable_column_indices_per_row_group(
  in_memory_table_interface const& table, std::string const& table_name)
{
  auto const num_row_groups = table.num_row_groups();
  std::vector<std::vector<size_t>> indices_per_row_group(num_row_groups);
  for (size_t rg_idx = 0; rg_idx < num_row_groups; ++rg_idx) {
    row_group_interface const& row_group = *table.row_group_at(rg_idx);
    auto const ncols                     = static_cast<size_t>(row_group.num_columns());
    for (size_t col_idx = 0; col_idx < ncols; ++col_idx) {
      column_base const& col = row_group.get_column(static_cast<cudf::size_type>(col_idx));
      if (dynamic_cast<compressed_sliced_column_interface const*>(&col) == nullptr) {
        // deserialize requires a `.bin` for every column, so only compressed-sliced columns are
        // supported.
        throw std::runtime_error("serialize: column index " + std::to_string(col_idx) +
                                 " in row group " + std::to_string(rg_idx) + " of table '" +
                                 table_name + "' is not a compressed-sliced column (layout=" +
                                 std::to_string(static_cast<int>(col.layout())) +
                                 "); serialization supports only compressed-sliced columns");
      }
      indices_per_row_group[rg_idx].push_back(col_idx);
    }
  }
  return indices_per_row_group;
}

/** Serialize row-group columns under `table_serialized_data_root/rg-<i>/`. */
void serialize_row_group_columns(row_group_interface const& rg,
                                 size_t row_group_index,
                                 std::string const& table_serialized_data_root,
                                 std::string const& table_name,
                                 std::vector<std::string> const& column_names,
                                 std::span<size_t const> column_indices,
                                 cuda::stream_ref stream)
{
  GQE_LOG_INFO("Serializing row group {} of table '{}' ({} columns) to '{}'",
               row_group_index,
               table_name,
               rg.num_columns(),
               table_serialized_data_root);

  std::filesystem::path const column_chunk_dir = utility::serialized_row_group_column_root(
    std::filesystem::path(table_serialized_data_root), row_group_index);
  std::error_code mkdir_ec;
  std::filesystem::create_directories(column_chunk_dir, mkdir_ec);
  if (mkdir_ec) { throw std::runtime_error("Could not create directories: " + mkdir_ec.message()); }

  for (size_t col_idx : column_indices) {
    auto const& col = dynamic_cast<compressed_sliced_column_interface const&>(
      rg.get_column(static_cast<cudf::size_type>(col_idx)));
    std::string label = column_names[col_idx];

    std::filesystem::path const file_path = serialized_column_path(column_chunk_dir, label);
    GQE_LOG_TRACE(
      "column='{}' idx={} bin='{}' table='{}'", label, col_idx, file_path.string(), table_name);
    col.serialize_to_disk(label, file_path, stream);
  }
}

/** Serialize row-group zone maps under `table_serialized_data_root/rg-<i>/zone_maps/`. */
void serialize_zone_maps_to_disk(in_memory_table_interface const& table,
                                 std::filesystem::path const& table_serialized_data_root,
                                 std::string const& table_name)
{
  for (size_t i = 0; i < table.num_row_groups(); ++i) {
    gqe::zone_map* const zm = table.row_group_at(i)->zone_map();
    if (zm == nullptr) {
      // The row group was registered without partition pruning, so there is no zone map to write.
      // Deserialization detects the absent file and reconstructs the row group without a zone map.
      GQE_LOG_TRACE(
        "serialize_zone_maps_to_disk: row group {} has no zone map; skipping table='{}'",
        i,
        table_name);
      continue;
    }
    std::filesystem::path const zm_dir =
      utility::serialized_row_group_zone_maps_root(table_serialized_data_root, i);
    zm->serialize_to_disk(zm_dir);
    GQE_LOG_TRACE(
      "serialize_zone_maps_to_disk: wrote zone map for row group {} table='{}' dir='{}'",
      i,
      table_name,
      zm_dir.string());
  }
}

/**
 * Load row-group zone maps from `table_serialized_data_root/rg-<i>/zone_maps/`.
 *
 * @return One zone map per entry of @p rg_dirs, in the same order. An entry is nullptr if the
 * corresponding row group was serialized without a zone map.
 */
[[nodiscard]] std::vector<std::unique_ptr<gqe::zone_map>> deserialize_zone_maps_from_disk(
  std::filesystem::path const& table_serialized_data_root,
  std::vector<std::filesystem::path> const& rg_dirs,
  std::string const& table_name,
  bool require_zone_map,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<gqe::zone_map>> zone_maps;
  zone_maps.reserve(rg_dirs.size());
  try {
    for (std::filesystem::path const& rg_path : rg_dirs) {
      auto const row_group_index = static_cast<size_t>(row_group_directory_index(rg_path));
      std::filesystem::path const zm_dir =
        utility::serialized_row_group_zone_maps_root(table_serialized_data_root, row_group_index);
      zone_maps.push_back(gqe::local_zone_map::deserialize_from_disk(zm_dir, stream));
      // Absence is valid only when partition pruning is off. With pruning on, every row group
      // must carry a zone map, so a missing file signals an incomplete snapshot.
      if (require_zone_map && !zone_maps.back()) {
        throw std::invalid_argument("deserialize: missing zone map for row group " +
                                    std::to_string(row_group_index) + " under '" + zm_dir.string() +
                                    "' (partition pruning is enabled)");
      }
      GQE_LOG_TRACE("deserialize_zone_maps_from_disk: row group {} table='{}' (zone_map {})",
                    row_group_index,
                    table_name,
                    zone_maps.back() ? "present" : "absent");
    }
  } catch (...) {
    // Deallocate the zone maps already loaded before `zone_maps` is unwound, because their
    // destructors do not free the underlying IPC buffers.
    for (auto& zone_map : zone_maps) {
      if (zone_map) { zone_map->deallocate(stream, mr); }
    }
    throw;
  }
  return zone_maps;
}

// Shared serialize path for both local and shared in-memory tables. Works entirely through
// the row-group interface (num_row_groups / row_group_at / zone_map) and the virtual column
// serialize_to_disk, so it is storage-kind agnostic. Callers hold a shared latch on their own
// row-group container for the duration of the pass. @p column_names must be index-ordered.
void serialize_in_memory_table_to_disk(in_memory_table_interface const& table,
                                       std::vector<std::string> const& column_names,
                                       optimization_parameters const& opt_params,
                                       std::string const& table_serialized_data_root,
                                       std::string const& table_name,
                                       cuda::stream_ref stream)
{
  // Serialization does not support secondary compression, so reject it explicitly.
  if (opt_params.secondary_compression_format != compression_format::none) {
    throw std::runtime_error(
      "serialize_table_to_disk: serialization is not supported when secondary compression is "
      "enabled (secondary_compression_format is not 'none') for table '" +
      table_name + "'");
  }

  auto const column_indices_per_row_group =
    serializable_column_indices_per_row_group(table, table_name);

  serialize_zone_maps_to_disk(table, std::filesystem::path(table_serialized_data_root), table_name);

  for (size_t i = 0; i < table.num_row_groups(); ++i) {
    serialize_row_group_columns(*table.row_group_at(i),
                                i,
                                table_serialized_data_root,
                                table_name,
                                column_names,
                                column_indices_per_row_group[i],
                                stream);
  }
}

/// Compression configuration used to reconstruct columns on deserialize, from the active params.
[[nodiscard]] gqe::compression::configuration make_compression_config(
  optimization_parameters const& opt_params)
{
  return gqe::compression::configuration{
    .primary_compression_format            = opt_params.compression_format,
    .secondary_compression_format          = opt_params.secondary_compression_format,
    .compression_chunk_size                = opt_params.compression_chunk_size,
    .compression_ratio_threshold           = opt_params.compression_ratio_threshold,
    .secondary_compression_ratio_threshold = opt_params.secondary_compression_ratio_threshold,
    .secondary_compression_multiplier_threshold =
      opt_params.secondary_compression_multiplier_threshold,
    .use_cpu_compression = opt_params.use_cpu_compression,
    .compression_level   = opt_params.compression_level,
    .decompress_backend  = opt_params.decompression_backend};
}

/// Restore per-row-group statistics for the cost-based optimizer, mirroring the write task. Without
/// them a deserialized table reports ~0 rows to the estimator, so join reordering mis-chooses the
/// hash build side and can OOM. Call only on success, after the row groups are fully built.
template <typename RowGroups>
void restore_table_statistics(RowGroups const& row_groups,
                              std::vector<cudf::data_type> const& column_types,
                              table_statistics_manager* statistics)
{
  if (statistics == nullptr) { return; }
  for (auto const& built_row_group : row_groups) {
    table_statistics row_group_table_stats(built_row_group.size(), column_types);
    auto const built_num_columns = built_row_group.num_columns();
    for (int64_t column_idx = 0; column_idx < built_num_columns; ++column_idx) {
      column_statistics const col_stats = {
        .column_id         = static_cast<size_t>(column_idx),
        .compression_stats = built_row_group.get_column(static_cast<cudf::size_type>(column_idx))
                               .get_compression_stats()};
      row_group_table_stats.add_column_statistics(col_stats);
    }
    statistics->append_table_statistics(row_group_table_stats);
  }
}

}  // namespace

void in_memory_table::deserialize_table_from_disk(std::string const& table_serialized_data_root,
                                                  std::string const& table_name,
                                                  optimization_parameters const& opt_params,
                                                  rmm::device_async_resource_ref mr,
                                                  cuda::stream_ref stream,
                                                  table_statistics_manager* statistics)
{
  std::filesystem::path const root(table_serialized_data_root);
  auto const rg_dirs = sorted_row_group_directories(root);

  // Source is usable: take the exclusive latch and replace the existing row groups.
  std::unique_lock latch_guard(_row_group_latch);
  for (auto& row_group : _row_groups) {
    row_group.deallocate(stream, mr);
  }
  _row_groups.clear();

  // Declared before the try so failure cleanup can deallocate loaded-but-not-yet-owned zone maps.
  // Column and zone-map destructors do not free their buffers (deallocate() does), so a failure
  // mid-deserialize must explicitly free everything built so far.
  std::vector<std::unique_ptr<gqe::zone_map>> row_group_zone_maps;

  try {
    // One zone map per row-group directory (nullptr where none was serialized).
    row_group_zone_maps = deserialize_zone_maps_from_disk(
      root, rg_dirs, table_name, opt_params.use_partition_pruning, stream, mr);

    auto const compression_config = make_compression_config(opt_params);

    for (size_t rg_loop_index = 0; rg_loop_index < rg_dirs.size(); ++rg_loop_index) {
      std::filesystem::path const& rg_path = rg_dirs[rg_loop_index];
      auto const row_group_index = static_cast<size_t>(row_group_directory_index(rg_path));
      std::filesystem::path const column_chunk_dir =
        utility::serialized_row_group_column_root(root, row_group_index);
      std::vector<std::unique_ptr<column_base>> columns;
      columns.reserve(_column_names.size());

      try {
        for (std::string const& col_name : _column_names) {
          std::filesystem::path const bin = serialized_column_path(column_chunk_dir, col_name);

          if (!std::filesystem::exists(bin)) {
            throw std::invalid_argument("Deserialize: Missing binary file: " + bin.string());
          }

          columns.push_back(compressed_sliced_column::deserialize_from_disk(
            bin, col_name, compression_config, stream, mr, _memory_kind));
        }
      } catch (...) {
        // Deallocate the columns already built for this row group before `columns` is unwound;
        // their destructors do not free the underlying device buffers.
        for (auto& col : columns) {
          if (col) { col->deallocate(stream, mr); }
        }
        throw;
      }

      _row_groups.push_back(
        row_group(std::move(columns), std::move(row_group_zone_maps[rg_loop_index])));
    }
  } catch (...) {
    // Free the partially built table: published row groups (columns + their zone map) and any zone
    // maps not yet moved into one.
    for (auto& row_group : _row_groups) {
      row_group.deallocate(stream, mr);
    }
    _row_groups.clear();
    for (auto& zone_map : row_group_zone_maps) {
      if (zone_map) { zone_map->deallocate(stream, mr); }
    }

    throw;
  }

  restore_table_statistics(_row_groups, _column_types, statistics);
}

void in_memory_table::serialize_table_to_disk(std::string const& table_serialized_data_root,
                                              std::string const& table_name,
                                              optimization_parameters const& opt_params,
                                              cuda::stream_ref stream)
{
  std::shared_lock latch_guard(_row_group_latch);
  serialize_in_memory_table_to_disk(
    *this, _column_names, opt_params, table_serialized_data_root, table_name, stream);
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

memory_kind::type in_memory_table::get_memory_kind() const { return _memory_kind; }

std::size_t in_memory_table::num_row_groups() const { return _row_groups.size(); }

row_group_interface const* in_memory_table::row_group_at(std::size_t index) const
{
  return &_row_groups.at(index);
}

std::shared_mutex& in_memory_table::row_group_latch() const { return _row_group_latch; }

in_memory_table::row_group_appender in_memory_table::get_row_group_appender()
{
  return {&_row_groups, &_row_group_latch};
}

bool shared_in_memory_table::is_readable() const { return true; }

bool shared_in_memory_table::is_writeable() const { return true; }

std::uint64_t shared_in_memory_table::storage_revision() const
{
  auto const& table_storage = storage();
  auto guard =
    lock_shared_memory_mutex(table_storage._mutex, "shared_in_memory_table::storage_revision");
  return table_storage._storage_revision;
}

int32_t shared_in_memory_table::max_concurrent_readers() const
{
  std::shared_lock latch_guard(_row_group_latch);
  auto const nrow_groups = _row_groups.size();
  assert(nrow_groups <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));
  return static_cast<int32_t>(nrow_groups);
}

int32_t shared_in_memory_table::max_concurrent_writers() const
{
  return std::numeric_limits<int32_t>::max();
}

std::unique_ptr<readable_view> shared_in_memory_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new in_memory_readable_view(this));
}

std::unique_ptr<writeable_view> shared_in_memory_table::writeable_view()
{
  return std::unique_ptr<storage::writeable_view>(new in_memory_writeable_view(this));
}

shared_in_memory_table::row_group_appender shared_in_memory_table::get_row_group_appender()
{
  return {_shared_segment, _shared_table_handle};
}

shared_memory_handle_t shared_in_memory_table::shared_table_handle() const
{
  return _shared_table_handle;
}

cudf::size_type shared_in_memory_table::get_column_index(std::string const& column_name) const
{
  return _column_name_to_index.at(column_name);
}

memory_kind::type shared_in_memory_table::get_memory_kind() const { return _memory_kind; }

std::size_t shared_in_memory_table::num_row_groups() const { return _row_groups.size(); }

row_group_interface const* shared_in_memory_table::row_group_at(std::size_t index) const
{
  return &_row_groups.at(index);
}

std::shared_mutex& shared_in_memory_table::row_group_latch() const { return _row_group_latch; }

void shared_in_memory_table::serialize_table_to_disk(std::string const& table_serialized_data_root,
                                                     std::string const& table_name,
                                                     optimization_parameters const& opt_params,
                                                     cuda::stream_ref stream)
{
  // The catalog's table object is built empty at registration and is not refreshed when a writer
  // (task-manager rank 0) appends row groups to the shared storage. Reattach to the currently
  // published row groups first, or the serializer would observe an empty table and write nothing.
  reattach_row_groups();

  std::shared_lock latch_guard(_row_group_latch);
  serialize_in_memory_table_to_disk(*this,
                                    index_ordered_column_names(),
                                    opt_params,
                                    table_serialized_data_root,
                                    table_name,
                                    stream);
}

std::vector<std::string> shared_in_memory_table::index_ordered_column_names() const
{
  std::vector<std::string> column_names(_column_name_to_index.size());
  for (auto const& [name, index] : _column_name_to_index) {
    column_names.at(static_cast<size_t>(index)) = name;
  }
  return column_names;
}

namespace {

// Build a process-local wrapper over a shared column storage identified by @p handle so its buffers
// can be released through the wrapper's virtual deallocate. Used to reclaim columns constructed in
// the segment by a deserialize that then fails before publishing.
std::unique_ptr<column_base> make_shared_column_wrapper(shared_segment& segment,
                                                        column_storage_handle const& handle)
{
  switch (handle.layout) {
    case column_layout::contiguous:
      return std::make_unique<shared_contiguous_column>(segment, handle.storage_handle);
    case column_layout::compressed_sliced:
      return std::make_unique<shared_compressed_sliced_column>(segment, handle.storage_handle);
    case column_layout::string_compressed_sliced:
      return std::make_unique<shared_string_compressed_sliced_column<false>>(segment,
                                                                             handle.storage_handle);
    case column_layout::large_string_compressed_sliced:
      return std::make_unique<shared_string_compressed_sliced_column<true>>(segment,
                                                                            handle.storage_handle);
  }
  throw std::logic_error("make_shared_column_wrapper: unsupported column layout");
}

// Recompress a just-deserialized shared column with the run's secondary codec (recompress-on-load
// for shared tables). Builds the secondary representation in the segment via the storage's
// recompress(), keeps whichever representation wins the ratio thresholds, frees the loser, and
// returns the chosen handle. Mirrors compressed_sliced_column::deserialize_from_disk, the
// local-table path.
template <typename StorageT>
[[nodiscard]] column_storage_handle recompress_shared_column_if_better(
  shared_segment& segment,
  column_storage_handle const& primary_handle,
  gqe::compression::configuration const& compression_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  gqe::compression::configuration secondary_config = compression_config;
  secondary_config.primary_compression_format   = compression_config.secondary_compression_format;
  secondary_config.secondary_compression_format = gqe::compression_format::none;

  auto* const primary_ptr =
    static_cast<StorageT*>(segment.get_address_from_handle(primary_handle.storage_handle));
  // Build the recompressed storage directly in the segment (never move a stack object into it,
  // which would leave the storage's boost offset_ptr members dangling).
  auto* const secondary_ptr =
    primary_ptr->recompress_into_segment(segment, secondary_config, memory_kind, stream, mr);

  column_storage_handle const secondary_handle(primary_handle.layout,
                                               segment.get_handle_from_address(secondary_ptr));
  double const primary_ratio =
    make_shared_column_wrapper(segment, primary_handle)->get_compression_ratio();

  double const secondary_ratio =
    make_shared_column_wrapper(segment, secondary_handle)->get_compression_ratio();
  bool const secondary_passes =
    secondary_ratio > compression_config.secondary_compression_ratio_threshold;
  bool const secondary_better =
    secondary_ratio >
    (primary_ratio * compression_config.secondary_compression_multiplier_threshold);
  bool const keep_secondary = secondary_passes && secondary_better;

  // Free the loser. The wrapper's deallocate releases its buffers and destroys its storage object
  // in the segment.
  auto const loser_handle = keep_secondary ? primary_handle : secondary_handle;
  make_shared_column_wrapper(segment, loser_handle)->deallocate(stream, mr);

  return keep_secondary ? secondary_handle : primary_handle;
}

// Apply recompress-on-load to a deserialized shared column, dispatching on its layout. A no-op when
// no secondary codec is configured. Uncompressed (contiguous) columns are never produced by
// serialization, so they pass through unchanged.
[[nodiscard]] column_storage_handle recompress_shared_column_if_secondary(
  shared_segment& segment,
  column_storage_handle const& handle,
  gqe::compression::configuration const& compression_config,
  memory_kind::type memory_kind,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (compression_config.secondary_compression_format == gqe::compression_format::none) {
    return handle;
  }
  switch (handle.layout) {
    case column_layout::compressed_sliced:
      return recompress_shared_column_if_better<shared_compressed_sliced_column_storage>(
        segment, handle, compression_config, memory_kind, stream, mr);
    case column_layout::string_compressed_sliced:
      return recompress_shared_column_if_better<
        shared_string_compressed_sliced_column_storage<false>>(
        segment, handle, compression_config, memory_kind, stream, mr);
    case column_layout::large_string_compressed_sliced:
      return recompress_shared_column_if_better<
        shared_string_compressed_sliced_column_storage<true>>(
        segment, handle, compression_config, memory_kind, stream, mr);
    case column_layout::contiguous: return handle;
  }
  return handle;
}

}  // namespace

void shared_in_memory_table::replace_all_row_groups(
  std::vector<shared_row_group_storage>&& new_row_groups,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto& storage_ref = storage();

  // 1. Construct all replacement row-group storages in the segment first, so a failure here leaves
  //    the table unchanged (strong guarantee). Track the pointers to free them on any failure
  //    below.
  std::vector<shared_row_group_storage*> new_ptrs;
  new_ptrs.reserve(new_row_groups.size());
  // Runs while another exception is propagating, so a failure to free is recorded, not thrown.
  auto const destroy_new = [&]() noexcept {
    for (auto* ptr : new_ptrs) {
      try {
        _shared_segment->destroy_ptr(ptr);
      } catch (std::exception const& error) {
        record_if_shared_memory_unrecoverable(std::current_exception(),
                                              "shared_in_memory_table::replace_all_row_groups");
        GQE_LOG_ERROR("Failed to free unpublished shared row group: {}", error.what());
      } catch (...) {
        GQE_LOG_ERROR("Failed to free unpublished shared row group: unknown error");
      }
    }
  };
  try {
    for (auto& row_group : new_row_groups) {
      new_ptrs.push_back(_shared_segment->construct<shared_row_group_storage>(
        boost::interprocess::anonymous_instance)(std::move(row_group)));
    }
  } catch (...) {
    destroy_new();
    throw;
  }

  std::vector<shared_memory_handle_t> new_handles;
  new_handles.reserve(new_ptrs.size());
  for (auto* ptr : new_ptrs) {
    new_handles.push_back(_shared_segment->get_handle_from_address(ptr));
  }

  // 2. Swap the published handle list under the storage mutex (mirrors replace_row_groups). Reserve
  //    before the destructive clear so the push_back cannot reallocate/throw after old handles are
  //    dropped.
  std::vector<shared_memory_handle_t> old_handles;
  try {
    auto guard = lock_shared_memory_mutex(storage_ref._mutex,
                                          "shared_in_memory_table::replace_all_row_groups");
    storage_ref.throw_if_retired();
    storage_ref._row_group_storage_handles.reserve(new_handles.size());
    old_handles.assign(storage_ref._row_group_storage_handles.begin(),
                       storage_ref._row_group_storage_handles.end());
    storage_ref._row_group_storage_handles.clear();
    for (auto const handle : new_handles) {
      storage_ref._row_group_storage_handles.push_back(handle);
    }
    ++storage_ref._storage_revision;
    // Retire the detached storage under the lock that unpublished it, as replace_row_groups does.
    for (auto const handle : old_handles) {
      mark_row_group_storage_retired(*_shared_segment, handle);
    }
  } catch (...) {
    destroy_new();
    throw;
  }

  // 3. Rebuild this facade's local wrappers so the owning instance reads the new contents. Unlike
  //    cleanup_detached_row_groups, the facade is left populated (it now holds the new row groups).
  {
    std::unique_lock latch_guard(_row_group_latch);
    _row_groups.clear();
    for (auto const handle : new_handles) {
      _row_groups.emplace_back(*_shared_segment, handle);
    }
  }

  // 4. Reclaim the detached (now retired) storages. They are distinct from the freshly published
  //    row groups, so freeing them does not disturb the rebuilt facade. Failures are logged, not
  //    thrown -- ownership of the replacement has already transferred to shared metadata.
  for (auto const handle : old_handles) {
    try {
      shared_row_group detached(*_shared_segment, handle, row_group_storage_state::retired);
      detached.deallocate(stream, mr);
    } catch (std::exception const& error) {
      record_if_shared_memory_unrecoverable(std::current_exception(),
                                            "shared_in_memory_table::replace_all_row_groups");
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: {}", error.what());
    } catch (...) {
      GQE_LOG_ERROR("Failed to reclaim detached shared row group: unknown error");
    }
  }
}

void shared_in_memory_table::deserialize_table_from_disk(
  std::string const& table_serialized_data_root,
  std::string const& table_name,
  optimization_parameters const& opt_params,
  rmm::device_async_resource_ref mr,
  cuda::stream_ref stream,
  table_statistics_manager* statistics)
{
  GQE_LOG_INFO(
    "deserialize shared: ENTER table='{}' root='{}'", table_name, table_serialized_data_root);
  auto& segment = *_shared_segment;
  shared_memory_allocator alloc(segment.get_segment_manager());

  auto const column_names = index_ordered_column_names();

  // Validate and scan the source directory before mutating the table.
  std::filesystem::path const root(table_serialized_data_root);
  auto const rg_dirs = sorted_row_group_directories(root);

  // Compression configuration used to reconstruct columns.
  auto const compression_config = make_compression_config(opt_params);

  // Build every row group's columns and zone map in the segment. Track every constructed segment
  // storage so a partial failure frees them. Nothing is published until step 4, so the table is
  // left unchanged on failure (build-then-swap).
  std::vector<shared_row_group_storage> built;
  std::vector<column_storage_handle> built_columns;
  std::vector<shared_memory_handle_t> built_zone_maps;
  auto const cleanup_partial = [&]() noexcept {
    for (auto const handle : built_zone_maps) {
      try {
        gqe::shared_zone_map(segment, handle).deallocate(stream, mr);
      } catch (std::exception const& error) {
        record_if_shared_memory_unrecoverable(
          std::current_exception(), "shared_in_memory_table::deserialize_table_from_disk");
        GQE_LOG_ERROR("deserialize cleanup: failed to free shared zone map: {}", error.what());
      } catch (...) {
      }
    }
    for (auto const& handle : built_columns) {
      try {
        make_shared_column_wrapper(segment, handle)->deallocate(stream, mr);
      } catch (std::exception const& error) {
        record_if_shared_memory_unrecoverable(
          std::current_exception(), "shared_in_memory_table::deserialize_table_from_disk");
        GQE_LOG_ERROR("deserialize cleanup: failed to free shared column: {}", error.what());
      } catch (...) {
      }
    }
  };

  // Publication transfers ownership of the built column/zone-map storages to the published row
  // groups, so partial cleanup must run only if we fail before it succeeds.
  bool published = false;
  try {
    built.reserve(rg_dirs.size());
    for (auto const& rg_path : rg_dirs) {
      auto const rg_index = static_cast<size_t>(row_group_directory_index(rg_path));
      std::filesystem::path const column_chunk_dir =
        utility::serialized_row_group_column_root(root, rg_index);

      std::vector<column_storage_handle> columns;
      columns.reserve(column_names.size());
      int64_t num_rows = 0;
      for (auto const& name : column_names) {
        std::filesystem::path const bin = serialized_column_path(column_chunk_dir, name);
        if (!std::filesystem::exists(bin)) {
          throw std::invalid_argument("deserialize: missing binary file: " + bin.string());
        }

        auto const built_column =
          deserialize_column_into_segment(segment, alloc, bin, compression_config, stream, mr);

        // Track the primary for failure cleanup before recompression, which may free and replace
        // it.
        built_columns.push_back(built_column.handle);
        auto const column_handle = recompress_shared_column_if_secondary(
          segment, built_column.handle, compression_config, get_memory_kind(), stream, mr);

        if (column_handle.storage_handle != built_column.handle.storage_handle) {
          // Recompression kept the secondary copy and already freed the primary, so track the
          // survivor.
          built_columns.back() = column_handle;
        }
        columns.push_back(column_handle);
        num_rows = built_column.num_rows;
      }

      shared_row_group_storage row_group(static_cast<cudf::size_type>(num_rows), alloc);
      row_group.columns.reserve(columns.size());
      for (auto const& handle : columns) {
        row_group.columns.push_back(handle);
      }

      auto const zm_dir = utility::serialized_row_group_zone_maps_root(root, rg_index);
      auto const zone_map_handle =
        deserialize_zone_map_into_segment(segment, alloc, zm_dir, stream, mr);
      // A row group pruned at read time must have a zone map, so an absent one means the
      // snapshot is incomplete rather than simply zone-map-free.
      if (opt_params.use_partition_pruning && !zone_map_handle) {
        throw std::invalid_argument("deserialize: missing zone map for row group " +
                                    std::to_string(rg_index) + " under '" + zm_dir.string() +
                                    "' (partition pruning is enabled)");
      }
      if (zone_map_handle) {
        built_zone_maps.push_back(*zone_map_handle);
        row_group.zone_map = *zone_map_handle;
      }

      built.push_back(std::move(row_group));
    }

    // Publish atomically, rebuilding this facade. Kept inside the guarded region so a failure
    // here also frees the built-but-unpublished storages.

    stream.sync();
    replace_all_row_groups(std::move(built), stream, mr);
    published = true;
    GQE_LOG_TRACE("deserialize shared: table='{}' published", table_name);
  } catch (...) {
    // Only unpublished storages need freeing. Once published, the row groups own them.
    if (!published) { cleanup_partial(); }
    throw;
  }

  // Restore statistics after publication so it reads the rebuilt facade.

  restore_table_statistics(_row_groups, _column_types, statistics);
}

in_memory_table::row_group_appender::row_group_appender(
  std::deque<row_group>* non_owning_row_groups, std::shared_mutex* non_owning_row_group_latch)
  : _non_owning_row_groups(non_owning_row_groups),
    _non_owning_row_group_latch(non_owning_row_group_latch)
{
}

void in_memory_table::row_group_appender::operator()(row_group&& new_row_group)
{
  std::scoped_lock latch_guard(*_non_owning_row_group_latch);
  _non_owning_row_groups->push_back(std::move(new_row_group));
}

void in_memory_table::row_group_appender::operator()(std::vector<row_group>&& new_row_groups)
{
  std::scoped_lock latch_guard(*_non_owning_row_group_latch);
  for (auto&& rg : new_row_groups) {
    _non_owning_row_groups->push_back(std::move(rg));
  }
}

cudf::size_type detail::checked_row_count(pruning_results_t const& pruning_results)
{
  int64_t total_rows = 0;
  for (auto const& [/* row_group */ _, pruning_result] : pruning_results) {
    auto const rows = static_cast<int64_t>(pruning_result.num_rows());
    GQE_EXPECTS(
      rows >= 0 &&
        total_rows <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) - rows,
      "in-memory result row count exceeds cudf::size_type",
      std::overflow_error);
    total_rows += rows;
  }
  return static_cast<cudf::size_type>(total_rows);
}

output_column_builder::output_column_builder(cudf::data_type cudf_type,
                                             std::shared_ptr<pruning_results_t> pruning_results,
                                             cudf::size_type column_idx)
  : _cudf_type(cudf_type),
    _pruning_results(std::move(pruning_results)),
    _column_idx(column_idx),
    _num_rows(detail::checked_row_count(*_pruning_results))
{
}

void output_column_builder::submit_materialization_requests(utility::copy_batch&,
                                                            decompression_batch&,
                                                            cuda::stream_ref,
                                                            rmm::device_async_resource_ref)
{
}

std::unique_ptr<cudf::column> concatenating_output_column_builder::make_cudf_column(
  cuda::stream_ref concatenation_stream, rmm::device_async_resource_ref mr)
{
  std::vector<cudf::column_view> pruned_columns;
  pruned_columns.reserve(_pruning_results->size());
  for (auto& [row_group, pruning_result] : *_pruning_results) {
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      auto& column = row_group->get_column(_column_idx);
      if (column.layout() != column_layout::contiguous) {
        throw std::logic_error("Use dedicated compressed_sliced_output_column_builder instead");
      }
      auto const full_column_view = dynamic_cast<contiguous_column_interface&>(column).view();
      // Construct the partitioned column view
      const auto size        = partition.end - partition.start;
      const auto offset      = partition.start;
      const auto null_counts = partition.null_counts[_column_idx];
      const auto pruned_column_view =
        detail::slice_column(full_column_view, size, offset, null_counts);
      pruned_columns.push_back(std::move(pruned_column_view));
    }
  }

  // Construct the output column from the (partitioned) row group column views using
  // cudf::concatenate
  std::unique_ptr<cudf::column> cudf_column =
    cudf::concatenate(pruned_columns, concatenation_stream, mr);

  GQE_LOG_DEBUG(
    "Processed column with concatenate: _column_idx = {}, "
    "cudf_column->size() = {}",
    _column_idx,
    cudf_column->size());
  return cudf_column;
}

bool detail::is_compressed_string_layout(column_layout layout)
{
  return layout == column_layout::string_compressed_sliced ||
         layout == column_layout::large_string_compressed_sliced;
}

namespace {

[[nodiscard]] column_layout select_output_column_layout(cudf::data_type cudf_type,
                                                        pruning_results_t const& pruning_results,
                                                        cudf::size_type column_idx)
{
  auto const first_layout = pruning_results.front().first->get_column(column_idx).layout();
  if (cudf_type != cudf::data_type(cudf::type_id::STRING) ||
      !detail::is_compressed_string_layout(first_layout)) {
    return first_layout;
  }

  bool requires_large_offsets = false;
  size_t selected_chars_size  = 0;
  for (auto const& [row_group, pruning_result] : pruning_results) {
    auto const source_layout = row_group->get_column(column_idx).layout();
    GQE_EXPECTS(detail::is_compressed_string_layout(source_layout),
                "Compressed-sliced string materialization requires compatible row-group layouts");
    auto const& column =
      get_column<string_compressed_sliced_column_interface>(row_group, column_idx);
    requires_large_offsets = requires_large_offsets || column.is_large_string();

    for (auto partition_idx : pruning_result.partition_indexes()) {
      auto const partition_chars_size = column.get_char_partition_output_size(partition_idx);
      if (partition_chars_size > std::numeric_limits<size_t>::max() - selected_chars_size) {
        throw std::overflow_error(
          "Compressed-sliced string character buffer size overflows size_t");
      }
      selected_chars_size += partition_chars_size;
    }
  }

  requires_large_offsets =
    requires_large_offsets ||
    selected_chars_size > static_cast<size_t>(std::numeric_limits<int32_t>::max());
  return requires_large_offsets ? column_layout::large_string_compressed_sliced
                                : column_layout::string_compressed_sliced;
}

std::unique_ptr<output_column_builder> make_output_column_builder(
  const cudf::data_type cudf_type,
  const column_layout layout,
  std::shared_ptr<pruning_results_t> pruning_results,
  cudf::size_type column_idx)
{
  if (layout == column_layout::contiguous && cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<contiguous_output_column_builder<contiguous_column_interface>>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::contiguous &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<concatenating_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::compressed_sliced &&
             cudf_type != cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<compressed_sliced_output_column_builder>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::string_compressed_sliced &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<string_compressed_sliced_output_column_builder<false>>(
      cudf_type, pruning_results, column_idx);
  } else if (layout == column_layout::large_string_compressed_sliced &&
             cudf_type == cudf::data_type(cudf::type_id::STRING)) {
    return std::make_unique<string_compressed_sliced_output_column_builder<true>>(
      cudf_type, pruning_results, column_idx);
  }

  return std::make_unique<concatenating_output_column_builder>(
    cudf_type, pruning_results, column_idx);
}

rmm::device_buffer make_output_buffer(size_t buffer_size,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return rmm::device_buffer(buffer_size, stream, mr);
}

rmm::device_buffer make_output_buffer(size_t num_values,
                                      size_t value_size,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return make_output_buffer(num_values * value_size, stream, mr);
}

void submit_buffer_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch& d_batch,
  std::deque<rmm::device_buffer>& staging_buffers,
  compressed_sliced_buffer_view host_source_buffer,
  decompress_unit unit,
  gqe::compression::configuration const& compression_config,
  std::span<size_t const> partition_indexes,
  std::byte* output_ptr,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
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

    staging_buffers.emplace_back(staging_buffer_size, stream, mr);
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
                unit,
                compression_config,
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

/**
 * @brief Sum the null counts that the zone map recorded for a column over all unpruned partitions.
 *
 * The batched copy materializes data buffers only, so column builders use this count both to
 * decide whether a null mask has to be reconstructed and to set the resulting null count.
 */
cudf::size_type count_nulls(const pruning_results_t& pruning_results, cudf::size_type column_idx)
{
  cudf::size_type total_null_count = 0;
  for (const auto& [/* row_group */ _, pruning_result] : pruning_results) {
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      total_null_count += partition.null_counts[column_idx];
    }
  }
  return total_null_count;
}

}  // namespace

template <typename T>
rmm::device_buffer contiguous_output_column_builder<T>::allocate_output_buffer(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream, mr);
}

template <typename T>
void contiguous_output_column_builder<T>::submit_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch&,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  size_t num_copy_requests{0};
  for (const auto& [/* row_group */ _, pruning_result] : *_pruning_results) {
    num_copy_requests += pruning_result.consolidated_partitions().size();
  }
  c_batch.reserve(c_batch.size() + num_copy_requests);

  _output_buffer        = allocate_output_buffer(stream, mr);
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
  cuda::stream_ref concatenation_stream, rmm::device_async_resource_ref mr)
{
  // The batched copy materializes the data buffer only, so the null mask has to be reconstructed
  // separately from the source columns.
  const cudf::size_type total_null_count = count_nulls(*_pruning_results, _column_idx);

  if (total_null_count == 0) {
    return std::make_unique<cudf::column>(
      _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), 0);
  }

  // Build sliced column views that carry the source null masks. The views are needed only for
  // their masks, so cudf::concatenate_masks merges them without touching the values, which the
  // batched copy has already materialized.
  std::vector<cudf::column_view> sliced_views;
  sliced_views.reserve(_pruning_results->size());
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const T& column      = get_column<T>(row_group, _column_idx);
    const auto full_view = column.view();
    for (const auto& partition : pruning_result.consolidated_partitions()) {
      sliced_views.push_back(detail::slice_column(full_view,
                                                  partition.end - partition.start,
                                                  partition.start,
                                                  partition.null_counts[_column_idx]));
    }
  }

  // A single partition needs no merge, so its mask is copied directly. This avoids the stream
  // synchronization that cudf::concatenate_masks performs to compute a null count that this caller
  // already has from the zone map.
  rmm::device_buffer null_mask =
    sliced_views.size() == 1 ? cudf::copy_bitmask(sliced_views.front(), concatenation_stream, mr)
                             : cudf::concatenate_masks(sliced_views, concatenation_stream, mr);

  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), std::move(null_mask), total_null_count);
}

rmm::device_buffer compressed_sliced_output_column_builder::allocate_output_buffer(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return make_output_buffer(_num_rows, cudf::size_of(_cudf_type), stream, mr);
}

void compressed_sliced_output_column_builder::submit_materialization_requests(
  utility::copy_batch& c_batch,
  decompression_batch& d_batch,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  _output_buffer        = allocate_output_buffer(stream, mr);
  std::byte* target_ptr = reinterpret_cast<std::byte*>(_output_buffer.data());
  size_t target_ptr_offset{0};
  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    if (pruning_result.partition_size() <= 0) {
      throw std::invalid_argument(
        "Compressed-sliced materialization requires zone_map_partition_size > 0.");
    }
    const auto& column     = get_column<compressed_sliced_column_interface>(row_group, _column_idx);
    auto const data_buffer = column.data_buffer_view();
    auto const& compression_config = column.get_compression_config();
    auto const& partition_indexes  = pruning_result.partition_indexes();
    auto* row_group_target_ptr     = target_ptr + target_ptr_offset;

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _staging_buffers,
                                           data_buffer,
                                           decompress_unit::data,
                                           compression_config,
                                           partition_indexes,
                                           row_group_target_ptr,
                                           stream,
                                           mr);

    target_ptr_offset += pruning_result.num_rows() * cudf::size_of(_cudf_type);
  }
}

std::unique_ptr<cudf::column> compressed_sliced_output_column_builder::make_cudf_column(
  cuda::stream_ref /* concatenation_stream */, rmm::device_async_resource_ref /* mr */)
{
  // TODO: Copy the null bitmask from the source columns, as
  // contiguous_output_column_builder::make_cudf_column does. Compressed columns do not expose a
  // view(), so the null mask has to be extracted from the decompressed data or stored separately.
  // Until then, fail instead of silently returning a non-nullable column for nullable input.
  GQE_EXPECTS(count_nulls(*_pruning_results, _column_idx) == 0,
              "Compressed-sliced materialization does not support nullable columns yet");

  cudf::size_type null_count = 0;
  return std::make_unique<cudf::column>(
    _cudf_type, _num_rows, std::move(_output_buffer), rmm::device_buffer(), null_count);
}

template <bool large_string_mode>
std::pair<rmm::device_buffer, rmm::device_buffer>
string_compressed_sliced_output_column_builder<large_string_mode>::allocate_output_buffers(
  size_t char_buffer_size,
  size_t offset_buffer_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return {make_output_buffer(char_buffer_size, stream, mr),
          make_output_buffer(offset_buffer_size, stream, mr)};
}

template <bool large_string_mode>
void string_compressed_sliced_output_column_builder<
  large_string_mode>::submit_materialization_requests(utility::copy_batch& c_batch,
                                                      decompression_batch& d_batch,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
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
      get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    for (size_t partition_idx : pruning_result.partition_indexes()) {
      char_buffer_size += column.get_char_partition_output_size(partition_idx);
    }
    offset_buffer_size +=
      pruning_result.num_rows() *
      sizeof(
        typename string_compressed_sliced_output_column_builder<large_string_mode>::offsets_type);
  }

  std::tie(_char_buffer, _offset_buffer) =
    allocate_output_buffers(char_buffer_size, offset_buffer_size, stream, mr);
  auto* char_target_ptr   = reinterpret_cast<std::byte*>(_char_buffer.data());
  auto* offset_target_ptr = reinterpret_cast<std::byte*>(_offset_buffer.data());

  for (const auto& [row_group, pruning_result] : *_pruning_results) {
    const auto& column =
      get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    auto const char_buffer         = column.char_buffer_view();
    auto const offset_buffer       = column.offset_buffer_view();
    auto const& compression_config = column.get_compression_config();
    auto const& partition_indexes  = pruning_result.partition_indexes();

    size_t char_output_size{0};
    for (size_t partition_idx : partition_indexes) {
      char_output_size += column.get_char_partition_output_size(partition_idx);
    }

    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _char_staging_buffers,
                                           char_buffer,
                                           decompress_unit::chars,
                                           compression_config,
                                           partition_indexes,
                                           char_target_ptr,
                                           stream,
                                           mr);
    char_target_ptr += char_output_size;

    const size_t offset_output_size     = pruning_result.num_rows() * sizeof(offsets_type);
    auto* offset_materialization_target = offset_target_ptr;
    if constexpr (large_string_mode) {
      if (!column.is_large_string()) {
        _offset_conversion_buffers.emplace_back(
          pruning_result.num_rows() * sizeof(int32_t), stream, mr);
        offset_materialization_target =
          reinterpret_cast<std::byte*>(_offset_conversion_buffers.back().data());
        _offset_widening_requests.push_back(
          {reinterpret_cast<int32_t const*>(offset_materialization_target),
           reinterpret_cast<int64_t*>(offset_target_ptr),
           static_cast<size_t>(pruning_result.num_rows())});
      }
    } else {
      GQE_EXPECTS(!column.is_large_string(),
                  "32-bit string output cannot materialize 64-bit source offsets");
    }
    submit_buffer_materialization_requests(c_batch,
                                           d_batch,
                                           _offset_staging_buffers,
                                           offset_buffer,
                                           decompress_unit::offsets,
                                           compression_config,
                                           partition_indexes,
                                           offset_materialization_target,
                                           stream,
                                           mr);
    offset_target_ptr += offset_output_size;
  }
}

template <bool large_string_mode>
std::unique_ptr<cudf::column>
string_compressed_sliced_output_column_builder<large_string_mode>::make_cudf_column(
  cuda::stream_ref concatenation_stream, rmm::device_async_resource_ref /* mr */)
{
  // TODO: Copy the null bitmask from the source columns, as
  // contiguous_output_column_builder::make_cudf_column does. Until then, fail instead of silently
  // returning a non-nullable column for nullable input.
  GQE_EXPECTS(count_nulls(*_pruning_results, _column_idx) == 0,
              "Compressed-sliced materialization does not support nullable string columns yet");

  for (auto const& request : _offset_widening_requests) {
    widen_string_offsets(
      request.destination, request.source, request.num_offsets, concatenation_stream);
  }

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

  concatenation_stream.sync();

  offsets_type char_offset    = 0;
  cudf::size_type row_offset  = 0;
  size_t partition_offset_idx = 0;
  auto fill_partition_offsets = [&](const string_compressed_sliced_column_interface& column,
                                    const pruning_result_t& pruning_result,
                                    size_t partition_offset_idx) {
    auto chars_partition_offset_idx = partition_offset_idx;
    for (auto partition_idx : pruning_result.partition_indexes()) {
#ifndef NDEBUG
      GQE_LOG_DEBUG("partition_idx = {}, char_offset = {}", partition_idx, char_offset);
#endif
      auto const partition_char_size = column.get_char_partition_output_size(partition_idx);
      auto const max_char_offset = static_cast<size_t>(std::numeric_limits<offsets_type>::max());
      auto const current_char_offset = static_cast<size_t>(char_offset);
      if (partition_char_size > max_char_offset ||
          current_char_offset > max_char_offset - partition_char_size) {
        throw std::overflow_error("Compressed-sliced string character offset exceeds offset type");
      }

      partition_char_offsets[chars_partition_offset_idx] = char_offset;
      char_offset += static_cast<offsets_type>(partition_char_size);
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
  };

  for (auto& [row_group, pruning_result] : *_pruning_results) {
    auto& column = get_column<string_compressed_sliced_column_interface>(row_group, _column_idx);
    fill_partition_offsets(column, pruning_result, partition_offset_idx);
    partition_offset_idx += pruning_result.partition_indexes().size();
  }

  // create a copy on device before adjust_offsets since we will do a binary search over this buffer
  rmm::device_uvector<cudf::size_type> d_partition_row_offsets_buffer(partition_row_offsets_buffer,
                                                                      concatenation_stream);
  auto d_partition_row_offsets = d_partition_row_offsets_buffer.data();

  adjust_offsets(reinterpret_cast<offsets_type*>(_offset_buffer.data()),
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

std::unique_ptr<cudf::table> output_table_builder::build(cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  // Select one builder per projected column. Compressed string row groups may use different offset
  // widths, so their output layout accounts for all selected row groups.
  _output_columns.clear();
  _output_columns.reserve(_column_indexes.size());

  for (size_t i = 0; i < _column_indexes.size(); ++i) {
    const auto column_idx = _column_indexes[i];
    const auto cudf_type  = _data_types[i];
    const auto layout     = select_output_column_layout(cudf_type, *_pruning_results, column_idx);
    _output_columns.push_back(
      make_output_column_builder(cudf_type, layout, _pruning_results, column_idx));
  }

  std::unique_lock<std::mutex> overlap_mtx_lock;
  auto copy_stream          = stream;
  auto decompression_stream = stream;
  auto column_build_stream  = stream;

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
    output_column->submit_materialization_requests(c_batch, d_batch, copy_stream, mr);
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
    copy_stream.sync();
  }

  if (has_compressed_buffers) {
    utility::nvtx_scoped_range nvtx_range("Decompressing columns");
    decompression_stream.sync();
    utility::semaphore_acquire_guard guard(_ctx_ref._task_manager_context->decompress_semaphore);
    d_batch.execute(decompression_stream);
    decompression_stream.sync();
  }

  auto cudf_column_view =
    _output_columns | std::views::transform([column_build_stream, mr](auto& output_column) {
      utility::nvtx_scoped_range nvtx_range("Creating cuDF column");
      return output_column->make_cudf_column(column_build_stream, mr);
    });
  std::vector<std::unique_ptr<cudf::column>> cudf_columns(cudf_column_view.begin(),
                                                          cudf_column_view.end());

  cudaEvent_t columns_ready_evt{};
  GQE_CUDA_TRY(cudaEventCreateWithFlags(&columns_ready_evt, cudaEventDisableTiming));
  GQE_CUDA_TRY(cudaEventRecord(columns_ready_evt, column_build_stream.get()));
  GQE_CUDA_TRY(cudaStreamWaitEvent(stream.get(), columns_ready_evt));

  if (_ctx_ref._query_context->parameters.use_overlap_mtx) {
    shared_stream.stream.synchronize();
    if (column_build_stream.get() != stream.get()) {
      // Columns built on the shared stream are complete; make them free on the producer stream.
      for (auto& column : cudf_columns) {
        column = cudf::rebind_stream(std::move(*column), stream);
      }
    }
    overlap_mtx_lock.unlock();
  }
  GQE_CUDA_TRY(cudaEventDestroy(columns_ready_evt));

  return std::make_unique<cudf::table>(std::move(cudf_columns));
}

namespace detail {

compression::configuration make_compression_configuration(query_context const& query_ctx)
{
  auto const& parameters = query_ctx.parameters;
  return {.primary_compression_format            = parameters.compression_format,
          .secondary_compression_format          = parameters.secondary_compression_format,
          .compression_chunk_size                = parameters.compression_chunk_size,
          .compression_ratio_threshold           = parameters.compression_ratio_threshold,
          .secondary_compression_ratio_threshold = parameters.secondary_compression_ratio_threshold,
          .secondary_compression_multiplier_threshold =
            parameters.secondary_compression_multiplier_threshold,
          .use_cpu_compression = parameters.use_cpu_compression,
          .compression_level   = parameters.compression_level,
          .decompress_backend  = parameters.decompression_backend};
}

/**
 * @brief Select the physical storage layout for a column.
 *
 * Uncompressed columns use contiguous storage. Compressed strings select their offset width based
 * on the size of the character data; all other compressed columns use compressed-sliced storage.
 *
 * @param column Column whose layout is being selected.
 * @param configured_format Configured primary compression format.
 * @param stream Stream used when inspecting string character data.
 * @return The storage layout to use when materializing the column.
 */
column_layout pick_column_layout(cudf::column_view const& column,
                                 compression_format configured_format,
                                 cuda::stream_ref stream)
{
  if (configured_format == compression_format::none) { return column_layout::contiguous; }
  if (column.type().id() != cudf::type_id::STRING) { return column_layout::compressed_sliced; }

  auto const chars_size = cudf::strings_column_view(column).chars_size(stream);
  return chars_size > std::numeric_limits<int32_t>::max()
           ? column_layout::large_string_compressed_sliced
           : column_layout::string_compressed_sliced;
}

/**
 * @brief Materialize and append one process-local row group.
 */
std::optional<table_statistics> write_row_group(
  in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  cuda::stream_ref stream)
{
  table_statistics row_group_statistics(input_table.num_rows(), data_types);
  // One row group per call. `append_table_statistics` sums this field.
  row_group_statistics.num_row_groups = 1;
  std::vector<std::unique_ptr<column_base>> columns(column_indexes.size());
  for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
    auto const& input_column   = input_table.column(column_idx);
    auto const destination_idx = column_indexes.at(column_idx);
    auto cudf_column           = cudf::column(input_column, stream, memory_resource);

    switch (
      pick_column_layout(input_column, compression_config.primary_compression_format, stream)) {
      case column_layout::contiguous:
        GQE_LOG_TRACE("Uncompressed column size {}", cudf_column.size());
        columns.at(destination_idx) =
          std::make_unique<contiguous_column>(std::move(cudf_column), stream, memory_resource);
        break;
      case column_layout::compressed_sliced:
        GQE_LOG_TRACE("Compressed sliced column size {}", cudf_column.size());
        columns.at(destination_idx) =
          std::make_unique<compressed_sliced_column>(std::move(cudf_column),
                                                     partition_size,
                                                     memory_kind,
                                                     compression_config,
                                                     stream,
                                                     memory_resource);
        break;
      case column_layout::string_compressed_sliced:
        columns.at(destination_idx) =
          std::make_unique<string_compressed_sliced_column<false>>(std::move(cudf_column),
                                                                   partition_size,
                                                                   memory_kind,
                                                                   compression_config,
                                                                   stream,
                                                                   memory_resource);
        break;
      case column_layout::large_string_compressed_sliced:
        columns.at(destination_idx) =
          std::make_unique<string_compressed_sliced_column<true>>(std::move(cudf_column),
                                                                  partition_size,
                                                                  memory_kind,
                                                                  compression_config,
                                                                  stream,
                                                                  memory_resource);
        break;
    }

    row_group_statistics.add_column_statistics(
      {.column_id         = static_cast<size_t>(destination_idx),
       .compression_stats = columns.at(destination_idx)->get_compression_stats()});
  }

  auto zone_map = partition_size > 0
                    ? std::make_unique<gqe::local_zone_map>(input_table, partition_size, stream)
                    : nullptr;
  row_group_appender(
    storage::row_group(input_table.num_rows(), std::move(columns), std::move(zone_map)));

  return row_group_statistics;
}

/**
 * @brief Materialize and append one shared-memory row group.
 *
 * @return Statistics for a published row group, or `std::nullopt` when this rank does not publish.
 */
std::optional<table_statistics> write_row_group(
  shared_in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  cuda::stream_ref stream)
{
  // TODO: Fix multi-rank write statistics before allowing every rank to publish.
  if (!gqe::utility::multi_process::nvshmem_rank_zero()) { return std::nullopt; }

  table_statistics row_group_statistics(input_table.num_rows(), data_types);
  // One row group per call. `append_table_statistics` sums this field.
  row_group_statistics.num_row_groups = 1;
  auto& segment                       = row_group_appender.segment();
  shared_memory_allocator alloc(segment.get_segment_manager());
  shared_row_group_storage shared_row_group(input_table.num_rows(), alloc);
  shared_row_group.columns.reserve(column_indexes.size());
  std::vector<std::optional<column_storage_handle>> column_handles(column_indexes.size());
  for (cudf::size_type column_idx = 0; column_idx < input_table.num_columns(); ++column_idx) {
    auto const& input_column   = input_table.column(column_idx);
    auto const destination_idx = column_indexes.at(column_idx);
    auto const layout =
      pick_column_layout(input_column, compression_config.primary_compression_format, stream);
    column_compression_statistics compression_statistics;

    switch (layout) {
      case column_layout::contiguous: {
        using storage_t = shared_contiguous_column_storage;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          input_column, stream, memory_resource, alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = shared_contiguous_column(segment, handle).get_compression_stats();
        break;
      }
      case column_layout::compressed_sliced: {
        using storage_t = shared_compressed_sliced_column_storage;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
      case column_layout::string_compressed_sliced: {
        using storage_t = shared_string_compressed_sliced_column_storage<false>;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
      case column_layout::large_string_compressed_sliced: {
        using storage_t = shared_string_compressed_sliced_column_storage<true>;
        auto* storage   = segment.construct<storage_t>(boost::interprocess::anonymous_instance)(
          cudf::column(input_column, stream, memory_resource),
          partition_size,
          memory_kind,
          compression_config,
          stream,
          memory_resource,
          alloc);
        auto const handle = segment.get_handle_from_address(storage);
        column_handles.at(destination_idx).emplace(layout, handle);
        compression_statistics = storage->get_compression_stats();
        break;
      }
    }

    row_group_statistics.add_column_statistics(
      {.column_id         = static_cast<size_t>(destination_idx),
       .compression_stats = std::move(compression_statistics)});
  }

  for (auto const& handle : column_handles) {
    if (!handle) { throw std::logic_error("Shared write did not populate every table column"); }
    shared_row_group.columns.push_back(*handle);
  }

  if (partition_size > 0) {
    shared_memory_allocator zone_map_alloc(segment.get_segment_manager());
    auto* zone_map_storage = segment.construct<gqe::shared_zone_map_storage>(
      boost::interprocess::anonymous_instance)(input_table,
                                               partition_size,
                                               stream,
                                               rmm::mr::get_current_device_resource_ref(),
                                               memory_resource,
                                               zone_map_alloc);
    shared_row_group.zone_map = segment.get_handle_from_address(zone_map_storage);
  }

  // Ensure all asynchronous copies into the shared segment have completed before publishing it.
  stream.sync();
  row_group_appender(std::move(shared_row_group));

  return row_group_statistics;
}

}  // namespace detail

in_memory_write_task::in_memory_write_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<task> input,
                                           rmm::device_async_resource_ref non_owned_memory_resource,
                                           row_group_appender_t row_group_appender,
                                           memory_kind::type memory_kind,
                                           std::vector<cudf::size_type> column_indexes,
                                           std::vector<cudf::data_type> data_types,
                                           table_statistics_manager* statistics)
  : write_task_base(ctx_ref, task_id, stage_id, input),
    _non_owned_memory_resource(non_owned_memory_resource),
    _row_group_appender(std::move(row_group_appender)),
    _memory_kind(memory_kind),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _statistics(statistics)
{
}

void in_memory_write_task::execute()
{
  GQE_LOG_TRACE("Write task execute");
  prepare_dependencies();
  utility::nvtx_scoped_range in_memory_write_task_range("in_memory_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto const input_table        = *dependent_tasks[0]->result();
  auto const partition_size     = get_query_context()->parameters.zone_map_partition_size;
  auto const compression_config = detail::make_compression_configuration(*get_query_context());
  if (compression_config.primary_compression_format != compression_format::none &&
      partition_size <= 0) {
    throw std::invalid_argument(
      "Compressed in-memory writes require zone_map_partition_size > 0 because "
      "compressed_sliced_column is the only supported compressed storage layout.");
  }

  auto const num_columns = _column_indexes.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data types must have the same length as the number of columns");
  }
  if (static_cast<std::size_t>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }
  for (std::size_t column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  auto const stream = cudf::get_default_stream();

  auto row_group_statistics = std::visit(
    [&](auto& row_group_appender) {
      return detail::write_row_group(row_group_appender,
                                     input_table,
                                     _non_owned_memory_resource,
                                     _memory_kind,
                                     _column_indexes,
                                     _data_types,
                                     compression_config,
                                     partition_size,
                                     stream);
    },
    _row_group_appender);
  if (row_group_statistics) { _statistics->append_table_statistics(*row_group_statistics); }

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

in_memory_writeable_view::in_memory_writeable_view(in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

in_memory_writeable_view::in_memory_writeable_view(shared_in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

std::unique_ptr<qep::fold_task> in_memory_writeable_view::get_fold_task(
  std::vector<std::string> column_names, std::vector<cudf::data_type> data_types)
{
  GQE_EXPECTS(column_names.size() == data_types.size(),
              "in_memory_writeable_view::get_fold_task: column_names and data_types must have the "
              "same length",
              std::invalid_argument);
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());

  auto* table = std::visit(
    [&](auto* concrete_table) -> in_memory_table_interface* {
      GQE_EXPECTS(column_names.size() == concrete_table->_column_types.size(),
                  "in_memory_writeable_view::get_fold_task: partial-table writes are not supported",
                  std::invalid_argument);

      for (std::size_t i = 0; i < data_types.size(); ++i) {
        auto const destination_idx = concrete_table->get_column_index(column_names[i]);
        GQE_EXPECTS(
          data_types[i] == concrete_table->_column_types.at(destination_idx),
          "in_memory_writeable_view::get_fold_task: input type does not match destination "
          "table column type",
          std::invalid_argument);
        column_indexes.push_back(destination_idx);
      }

      return concrete_table;
    },
    _non_owning_table);

  auto row_group_appender = std::visit(
    [](auto* concrete_table) -> chunked_in_memory_write_task::row_group_appender_t {
      return concrete_table->get_row_group_appender();
    },
    _non_owning_table);

  return std::make_unique<chunked_in_memory_write_task>(std::move(row_group_appender),
                                                        table->get_memory_kind(),
                                                        std::move(column_indexes),
                                                        std::move(data_types));
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
  assert(ctx_ref._task_manager_context != nullptr);

  auto* table =
    std::visit([](auto* concrete_table) -> in_memory_table_interface* { return concrete_table; },
               _non_owning_table);

  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(
    column_names.cbegin(),
    column_names.cend(),
    std::back_inserter(column_indexes),
    [table](std::string const& column_name) { return table->get_column_index(column_name); });

  std::vector<std::unique_ptr<write_task_base>> write_tasks;
  write_tasks.reserve(task_parameters.size());

  auto const memory_kind = table->get_memory_kind();
  auto memory_resource   = ctx_ref._task_manager_context->get_table_memory_resource(memory_kind);

  for (auto& task_parameter : task_parameters) {
    auto row_group_appender = std::visit(
      [](auto* concrete_table) -> in_memory_write_task::row_group_appender_t {
        return concrete_table->get_row_group_appender();
      },
      _non_owning_table);

    write_tasks.push_back(std::make_unique<in_memory_write_task>(ctx_ref,
                                                                 task_parameter.task_id,
                                                                 stage_id,
                                                                 std::move(task_parameter.input),
                                                                 memory_resource,
                                                                 std::move(row_group_appender),
                                                                 memory_kind,
                                                                 column_indexes,
                                                                 data_types,
                                                                 statistics));
  }

  return write_tasks;
}

}  // namespace storage

}  // namespace gqe
