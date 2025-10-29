/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <algorithm>
#include <cstddef>
#include <cudf_test/default_stream.hpp>
#include <deque>
#include <gqe/expression/json_formatter.hpp>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock
#include <numeric>
#include <shared_mutex>  // std::shared_lock
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

namespace storage {

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
    _data_size = sizeof(char) * string_device_view.chars_size(stream);
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

cudf::column_view contiguous_column::view() const { return _data.view(); }

cudf::mutable_column_view contiguous_column::mutable_view() { return _data.mutable_view(); }

std::unique_ptr<rmm::device_buffer> compressed_column::compress(rmm::device_buffer const* input,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr,
                                                                bool is_null_mask)
{
  std::unique_ptr<rmm::device_buffer> output;
  if (is_null_mask) {
    std::tie(output, _null_mask_compression_config) = _nvcomp_manager.do_compress(
      input, _null_mask_compression_ratio, _is_null_mask_compressed, stream, mr);
  } else {
    std::tie(output, _compression_config) =
      _nvcomp_manager.do_compress(input, _compression_ratio, _is_compressed, stream, mr);
  }
  return output;
}

compressed_column::compressed_column(cudf::column&& cudf_column,
                                     compression_format comp_format,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     nvcompType_t nvcomp_data_format,
                                     int compression_chunk_size,
                                     std::string column_name,
                                     cudf::data_type cudf_type)
  : column_base(),
    _comp_format(comp_format),
    _compression_ratio(0.0),
    _null_mask_compression_ratio(0.0),
    _is_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(
      comp_format, nvcomp_data_format, compression_chunk_size, stream, mr, column_name, cudf_type)
{
  _size       = cudf_column.size();
  _dtype      = cudf_column.type();
  _null_count = cudf_column.null_count();

  auto column_content = cudf_column.release();
  _compressed_data    = compress(column_content.data.get(), stream, mr, false);
  _compressed_size    = _compressed_data->size();
  if (_null_count > 0) {
    _compressed_null_mask = compress(column_content.null_mask.get(), stream, mr, true);
  }

  _compressed_children.reserve(column_content.children.size());

  for (auto& child : column_content.children) {
    auto dtype         = child->type().id();
    nvcomp_data_format = get_optimal_nvcomp_data_type(dtype);

    if ((comp_format == gqe::compression_format::best_compression_ratio) ||
        (comp_format == gqe::compression_format::best_decompression_speed)) {
      best_compression_config(
        dtype,
        comp_format,
        nvcomp_data_format,
        (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
    }

    _compressed_children.push_back(std::make_unique<compressed_column>(std::move(*child),
                                                                       comp_format,
                                                                       stream,
                                                                       mr,
                                                                       nvcomp_data_format,
                                                                       compression_chunk_size,
                                                                       column_name + "_child",
                                                                       cudf_type));
  }
}

shared_compressed_column_base::shared_compressed_column_base(
  gqe::storage::compressed_column&& compressed_column,
  boost::interprocess::managed_shared_memory& segment,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : _size(compressed_column._size),
    _compressed_size(compressed_column._compressed_data->size()),
    _compressed_null_mask_size(
      compressed_column._null_count > 0 ? compressed_column._compressed_null_mask->size() : 0),
    _dtype(compressed_column._dtype),
    _null_count(compressed_column._null_count),
    _comp_format(compressed_column._comp_format),
    _compression_ratio(compressed_column._compression_ratio),
    _null_mask_compression_ratio(
      compressed_column._null_count > 0 ? compressed_column._null_mask_compression_ratio : 0.0),
    _is_compressed(compressed_column._is_compressed),
    _is_null_mask_compressed(compressed_column._is_null_mask_compressed),
    _nvcomp_manager(compressed_column._nvcomp_manager.get_comp_format(),
                    compressed_column._nvcomp_manager.get_data_type(),
                    compressed_column._nvcomp_manager.get_compression_chunk_size(),
                    stream,
                    mr,
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
        _compressed_null_mask.get(), _compressed_null_mask_size, stream, mr);
    } else {
      decompressed_null_mask =
        std::make_unique<rmm::device_buffer>(static_cast<const void*>(_compressed_null_mask.get()),
                                             _compressed_null_mask_size,
                                             stream,
                                             mr);
    }
  } else {
    decompressed_null_mask = std::make_unique<rmm::device_buffer>();
  }

  return std::make_unique<cudf::column>(_dtype,
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
    _mr.deallocate(_compressed_null_mask.get(), _compressed_null_mask_size);
    _compressed_null_mask      = nullptr;
    _compressed_null_mask_size = 0;
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

int64_t compressed_column::size() const { return _size; }

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

      *_compression_config,
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
        *_null_mask_compression_config,
        stream,
        mr);
    } else {
      decompressed_null_mask =
        std::make_unique<rmm::device_buffer>(*_compressed_null_mask, stream, mr);
    }
  } else {
    decompressed_null_mask = std::make_unique<rmm::device_buffer>();
  }

  return std::make_unique<cudf::column>(_dtype,
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
                                 std::vector<cudf::data_type> const& column_types)
  : table(),
    _memory_kind(memory_kind),
    _column_types(std::move(column_types)),
    _row_groups(),
    _row_group_latch()
{
  // Populate column name-to-index map
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }

  // Create memory resource based on memory kind. The allocators must be thread-safe.
  std::visit(utility::overloaded{
               [this](memory_kind::system) {
                 _memory_resource = std::make_unique<memory_resource::system_memory_resource>();
               },
               [this](const memory_kind::numa& numa) {
                 _memory_resource = std::make_unique<memory_resource::numa_memory_resource>(
                   numa.numa_node_set, numa.page_kind);
               },
               [this](const memory_kind::pinned&) {
                 _memory_resource = std::make_unique<memory_resource::pinned_memory_resource>();
               },
               [this](const memory_kind::device&) {
                 // FIXME: specify device instead of allocating on default CUDA device
                 _memory_resource = std::make_unique<rmm::mr::cuda_memory_resource>();
               },
               [this](const memory_kind::managed&) {
                 _memory_resource = std::make_unique<rmm::mr::managed_memory_resource>();
               },
               [this](const memory_kind::numa_pinned& numa_pinned) {
                 _memory_resource = std::make_unique<memory_resource::numa_memory_resource>(
                   numa_pinned.numa_node_set, numa_pinned.page_kind, true);
               },
               [this](const memory_kind::boost_shared& boost_shared) {
                 _shared_memory_resource = boost_shared.mr;
               }},
             memory_kind);
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
  if (should_use_overlap_mtx()) {
    GQE_LOG_DEBUG("Using overlap mutex for in_memory_read_task");
    _decompression_stream =
      get_context_reference()._task_manager_context->copy_engine_stream.stream;
  } else {
    _decompression_stream = cudf::get_default_stream();
  }
}

bool in_memory_read_task::should_use_overlap_mtx() const
{
  return get_query_context()->parameters.use_overlap_mtx;
}

bool in_memory_read_task::can_prune_partitions() const
{
  if (!_partial_filter) { return false; }
  if (get_query_context()->parameters.zone_map_partition_size == 0) { return false; }
  return get_query_context()->parameters.use_partition_pruning;
}

std::vector<row_group_with_partitions> in_memory_read_task::evaluate_partial_filter()
{
  std::vector<row_group_with_partitions> row_groups_with_partitions;
  std::for_each(_row_groups.begin(), _row_groups.end(), [&](const auto* rg) {
    if (!rg->zone_map()) {
      throw std::logic_error("Row group should have a zone map but none exists");
    }
    const auto partitions =
      rg->zone_map()->evaluate(get_query_context()->parameters, *_partial_filter);
    const auto has_unpruned_partitions =
      std::find_if(partitions.begin(), partitions.end(), [](const auto& partition) {
        return !partition.pruned;
      }) != partitions.end();
    if (has_unpruned_partitions) {
      const auto consolidated_partitions = gqe::zone_map::consolidate_partitions(partitions);
      row_groups_with_partitions.push_back({rg, std::move(consolidated_partitions)});
    }
  });
  GQE_LOG_DEBUG("Row groups after pruning: {}", row_groups_with_partitions.size());
  return row_groups_with_partitions;
}

std::vector<row_group_with_partitions> in_memory_read_task::construct_fully_covering_partitions()
{
  GQE_LOG_DEBUG("No partial filter; processing all row groups");
  std::vector<row_group_with_partitions> row_groups_with_partitions;
  row_groups_with_partitions.reserve(_row_groups.size());
  std::transform(_row_groups.begin(),
                 _row_groups.end(),
                 std::back_inserter(row_groups_with_partitions),
                 [&](const auto* rg) {
                   std::vector<cudf::size_type> null_counts(rg->num_columns());

                   zone_map::partition partition{.pruned = false,
                                                 .start  = 0,
                                                 .end    = static_cast<cudf::size_type>(rg->size()),
                                                 .null_counts = null_counts};
                   std::vector<zone_map::partition> partitions = {std::move(partition)};
                   std::for_each(_column_indexes.begin(), _column_indexes.end(), [&](const auto i) {
                     const auto column_view =
                       optionally_decompress_column(rg->get_column(i), partitions);
                     partitions[0].null_counts[i] = column_view.null_count();
                   });
                   return row_group_with_partitions{rg, std::move(partitions)};
                 });
  return row_groups_with_partitions;
}

cudf::column_view in_memory_read_task::optionally_decompress_column(
  column_base& column, const std::vector<zone_map::partition>& partitions)
{
  switch (column.type()) {
    case in_memory_column_type::CONTIGUOUS: {
      return dynamic_cast<contiguous_column&>(column).view();
    }
    case in_memory_column_type::COMPRESSED: {
      auto stored_decompressed_column = _decompressed_columns.find(&column);
      if (stored_decompressed_column != _decompressed_columns.end()) {
        return stored_decompressed_column->second->view();
      } else {
        auto decompressed_column =
          dynamic_cast<compressed_column&>(column).decompress(_decompression_stream);
        const auto column_view = decompressed_column->view();
        _decompressed_columns.emplace(&column, std::move(decompressed_column));
        return column_view;
      }
    }
    case in_memory_column_type::COMPRESSED_SLICED: {
      // We'll get here if there is no partial filter
      auto stored_decompressed_column = _decompressed_columns.find(&column);
      if (stored_decompressed_column != _decompressed_columns.end()) {
        return stored_decompressed_column->second->view();
      }
      auto decompressed_column = dynamic_cast<compressed_sliced_column&>(column).decompress(
        _decompression_stream, partitions);
      const auto column_view = decompressed_column->view();
      _decompressed_columns.emplace(&column, std::move(decompressed_column));
      return column_view;
    }
    case in_memory_column_type::SHARED_CONTIGUOUS: {
      auto view = dynamic_cast<shared_contiguous_column&>(column).view();
      return view;
    }
    case in_memory_column_type::SHARED_COMPRESSED: {
      auto stored_decompressed_column = _decompressed_columns.find(&column);
      if (stored_decompressed_column != _decompressed_columns.end()) {
        return stored_decompressed_column->second->view();
      } else {
        auto decompressed_column =
          dynamic_cast<shared_compressed_column&>(column).decompress(_decompression_stream);
        const auto column_view = decompressed_column->view();
        _decompressed_columns.emplace(&column, std::move(decompressed_column));
        return column_view;
      }
    }
  }
  // Compiler complains about reaching end of non-void function, even though all cases in the
  // switch statement are specified and each case has a return statement.
  throw std::logic_error("Unknown column type");
}

std::vector<cudf::table_view> in_memory_read_task::slice_row_groups(
  const std::vector<row_group_with_partitions>& row_groups_with_partitions)
{
  if (row_groups_with_partitions.empty()) { return {}; }

  std::vector<cudf::table_view> sliced_row_groups;
  auto& first_row_group = row_groups_with_partitions.front().first;
  bool column_is_compressed_slice =
    first_row_group->get_column(0).type() == in_memory_column_type::COMPRESSED_SLICED;
  for (const auto& [row_group, partitions] : row_groups_with_partitions) {
    // we'll assume that all columns are the same format --
    // even if a column is uncompressed, it will still be a sliced_compressed_column if necessary

    if (column_is_compressed_slice) {
      std::vector<cudf::column_view> result_columns;
      result_columns.reserve(partitions.size());

      // If this is a compressed sliced column, we'll get one table view out of this row group
      // Otherwise we'll have a table view per partition
      for (auto const& idx : _column_indexes) {
        // We'll decompress the column and then slice it if necessary
        auto& column     = row_group->get_column(idx);
        auto column_type = column.type();
        if (column_type != in_memory_column_type::COMPRESSED_SLICED) {
          throw std::logic_error("If any column is compressed_sliced, all columns must be");
        }

        // We'll get a decompressed column that is fully reduced and materialized
        // The "cache" isn't necessary here, but we still use the map to enable persistent
        // storage of the decompressed column and to operate on column views.
        auto column_view = optionally_decompress_column(column, partitions);
        result_columns.push_back(column_view);
      }
      sliced_row_groups.push_back(cudf::table_view(std::move(result_columns)));
    } else {
      // We'll do the original logic
      std::vector<zone_map::partition> empty_partitions =
        {};  // Just to allow the argument to optionally decompress column
      for (const auto& partition : partitions) {
        if (!partition.pruned) {
          std::vector<cudf::column_view> result_columns;
          result_columns.reserve(partitions.size());
          for (auto const& idx : _column_indexes) {
            auto& column                = row_group->get_column(idx);
            const auto full_column_view = optionally_decompress_column(column, empty_partitions);
            const auto type             = full_column_view.type();
            const auto size             = partition.end - partition.start;
            const auto head             = full_column_view.head();
            const auto null_mask        = full_column_view.null_mask();
            const auto null_counts      = partition.null_counts[idx];
            const auto offset           = partition.start;
            const auto children = std::vector<cudf::column_view>(full_column_view.child_begin(),
                                                                 full_column_view.child_end());
            const cudf::column_view column_slice_view{
              type, size, head, null_mask, null_counts, offset, children};
            result_columns.push_back(column_slice_view);
          }
          sliced_row_groups.push_back(cudf::table_view(std::move(result_columns)));
        }
      }
    }
  }

  return sliced_row_groups;
}

bool in_memory_read_task::is_zero_copy_possible(
  const std::vector<cudf::table_view>& sliced_row_groups) const
{
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
  if (!_decompressed_columns.empty()) {
    GQE_LOG_DEBUG("Zero-copy read is disabled because at least one column was decompressed");
    return false;
  }
  if (sliced_row_groups.size() > 1) {
    GQE_LOG_DEBUG("Zero-copy read is not possible because there are multiple partitions");
    return false;
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

void in_memory_read_task::emit_single_partition(
  const std::vector<cudf::table_view>& sliced_row_groups)
{
  GQE_LOG_DEBUG("Performing zero-copy read");
  // Must have exactly one sliced partition, otherwise zero-copy is not possible.
  assert(sliced_row_groups.size() == 1);
  cudf::table_view result = sliced_row_groups.front();
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_reference, "
    "output_size={}.",
    task_id(),
    stage_id(),
    result.num_rows());
  emit_result(result);
}

void in_memory_read_task::emit_concatenated_partitions(
  const std::vector<cudf::table_view>& sliced_row_groups, cudaEvent_t& ce_evt)
{
  /*
   * If the column is not compressed we do a CE copy in concatenate, we want this copy
   * to be done on the shared CE stream to allow for pipelining.
   *
   * Otherwise, it is better to do this on the thread specific stream to avoid false
   * dependence. Note: The shared CE stream guarantees serial execution, if more than one
   * row group is compressed then the ce_evt will have overwrites. This is intended as we
   * only need to wait on the "last" scheduled copy on the shared CE stream.
   */
  std::unique_ptr<cudf::table> result;
  if (_decompressed_columns.empty() && should_use_overlap_mtx()) {
    result = cudf::concatenate(
      sliced_row_groups, _decompression_stream, rmm::mr::get_current_device_resource());
    GQE_CUDA_TRY(cudaEventRecord(ce_evt, _decompression_stream.value()));
    GQE_CUDA_TRY(cudaStreamWaitEvent(cudf::get_default_stream().value(), ce_evt));
  } else {
    GQE_LOG_DEBUG("Calling cudf::concatenate with {} partitions", sliced_row_groups.size());
    result = cudf::concatenate(sliced_row_groups);
    GQE_LOG_DEBUG("Called cudf::concatenate with {} partitions", sliced_row_groups.size());
  }

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

  // Lock on the shared copy engine stream.
  std::unique_lock<std::mutex> ce_lock;
  // Event used to synchronize the CUDF default stream and the shared copy engine stream.
  cudaEvent_t ce_evt;

  if (should_use_overlap_mtx()) {
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&ce_evt, cudaEventDisableTiming));
    ce_lock =
      std::unique_lock{get_context_reference()._task_manager_context->copy_engine_stream.mtx};
  }

  const std::vector<row_group_with_partitions> row_groups_with_partitions =
    can_prune_partitions() ? evaluate_partial_filter() : construct_fully_covering_partitions();
  const std::vector<cudf::table_view> sliced_row_groups =
    slice_row_groups(row_groups_with_partitions);

  if (!_decompressed_columns.empty() && should_use_overlap_mtx()) {
    // Wait on the CUDF default stream until data transfers of the shared copy engine are
    // finished.
    auto default_stream = cudf::get_default_stream().value();
    auto shared_ce_stream =
      get_context_reference()._task_manager_context->copy_engine_stream.stream.value();
    GQE_CUDA_TRY(cudaEventRecord(ce_evt, shared_ce_stream));
    GQE_CUDA_TRY(cudaStreamWaitEvent(default_stream, ce_evt));
  }

  auto unlock_shared_copy_engine = [&ce_lock, &ce_evt, this]() {
    // We sync while holding the lock to avoid aggressive memory allocation leading to OOM errors.
    // This is a hotfix so the behaviour is same as that for cuDF 24.12.
    get_context_reference()._task_manager_context->copy_engine_stream.stream.synchronize();
    ce_lock.unlock();
    GQE_CUDA_TRY(cudaEventDestroy(ce_evt));
  };
  // If there are multiple sliced row groups that have to be concatenated, the lock to the shared
  // copy engine has to be held until after the concatenation. Otherwise the lock can be relased
  // now.
  if (sliced_row_groups.empty()) {
    if (should_use_overlap_mtx()) { unlock_shared_copy_engine(); }
    emit_empty_table();
  } else if (is_zero_copy_possible(sliced_row_groups)) {
    if (should_use_overlap_mtx()) { unlock_shared_copy_engine(); }
    emit_single_partition(sliced_row_groups);
  } else {
    GQE_LOG_DEBUG("Concatenating {} partitions", sliced_row_groups.size());
    emit_concatenated_partitions(sliced_row_groups, ce_evt);
    GQE_LOG_DEBUG("Concatenated {} partitions", sliced_row_groups.size());
    if (should_use_overlap_mtx()) { unlock_shared_copy_engine(); }
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
  for (decltype(input_table.num_columns()) column_idx = 0; column_idx < input_table.num_columns();
       ++column_idx) {
    auto const& input_column = input_table.column(column_idx);
    auto cudf_column         = cudf::column(input_column, stream, _non_owned_memory_resource);

    auto comp_format = get_query_context()->parameters.in_memory_table_compression_format;
    auto const compression_chunk_size = get_query_context()->parameters.compression_chunk_size;

    auto dtype = cudf_column.type().id();

    nvcompType_t nvcomp_data_format = get_optimal_nvcomp_data_type(dtype);

    if ((comp_format == gqe::compression_format::best_compression_ratio) ||
        (comp_format == gqe::compression_format::best_decompression_speed)) {
      best_compression_config(
        dtype,
        comp_format,
        nvcomp_data_format,
        (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
    }

    if (comp_format == compression_format::none) {
      GQE_LOG_TRACE("Uncompressed column size {}", cudf_column.size());
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<contiguous_column>(std::move(cudf_column));
    } else {
      if (not partition_pruning_enabled) {
        GQE_LOG_TRACE("Compressed column size {}", cudf_column.size());
        new_columns[_column_indexes[column_idx]] =
          std::make_unique<compressed_column>(std::move(cudf_column),
                                              comp_format,
                                              stream,
                                              _non_owned_memory_resource,
                                              nvcomp_data_format,
                                              compression_chunk_size,
                                              _column_names[column_idx],
                                              cudf_column.type());
      } else if (dtype == cudf::type_id::STRING) {
        // Get a string view of the column
        cudf::strings_column_view strings_column_view(cudf_column);
        int64_t chars_size = strings_column_view.chars_size(stream);
        GQE_LOG_TRACE("String column size {}", chars_size);
        if (chars_size > std::numeric_limits<int32_t>::max()) {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<true>>(std::move(cudf_column),
                                                                    comp_format,
                                                                    stream,
                                                                    _non_owned_memory_resource,
                                                                    compression_chunk_size,
                                                                    partition_size,
                                                                    _column_names[column_idx]);
        } else {
          new_columns[_column_indexes[column_idx]] =
            std::make_unique<string_compressed_sliced_column<false>>(std::move(cudf_column),
                                                                     comp_format,
                                                                     stream,
                                                                     _non_owned_memory_resource,
                                                                     compression_chunk_size,
                                                                     partition_size,
                                                                     _column_names[column_idx]);
        }
      } else {
        GQE_LOG_TRACE("Compressed sliced column size {}", cudf_column.size());
        new_columns[_column_indexes[column_idx]] =
          std::make_unique<compressed_sliced_column>(std::move(cudf_column),
                                                     comp_format,
                                                     stream,
                                                     _non_owned_memory_resource,
                                                     nvcomp_data_format,
                                                     compression_chunk_size,
                                                     partition_size,
                                                     _column_names[column_idx],
                                                     cudf_column.type());
      }
    }
  }

  // Create zone map

  std::unique_ptr<gqe::zone_map> zone_map =
    partition_pruning_enabled ? std::make_unique<gqe::zone_map>(input_table, partition_size)
                              : nullptr;

  // Create row group from columns
  auto row_group = storage::row_group(std::move(new_columns), std::move(zone_map));

  // Append row group to table
  _appender(std::move(row_group));
  _statistics->add_rows(input_table.num_rows());

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

    std::string shared_column_name = [&] {
      std::ostringstream oss;
      oss << "shared_column_" << stage_id() << "_" << task_id() << "_" << column_idx << "_"
          << _column_names[column_idx];
      return oss.str();
    }();

    cudf::column_view input_column = [&]() {
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

        auto const chunk_size = get_query_context()->parameters.compression_chunk_size;

        auto dtype                      = input_column.type().id();
        nvcompType_t nvcomp_data_format = get_optimal_nvcomp_data_type(dtype);

        if ((comp_format == gqe::compression_format::best_compression_ratio) ||
            (comp_format == gqe::compression_format::best_decompression_speed)) {
          best_compression_config(
            dtype,
            comp_format,
            nvcomp_data_format,
            (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
        }

        // TODO: Not safe to pass the cudf::data_type to shared memory, need to investigate why
        // It's only used for logging.
        gqe::storage::compressed_column compressed_column(std::move(cudf_column),
                                                          comp_format,
                                                          stream,
                                                          rmm::mr::get_current_device_resource(),
                                                          nvcomp_data_format,
                                                          chunk_size,
                                                          _column_names[column_idx]);
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

    // We currently need to share boost_shared_memory_resource across tables
    // because of the issue: see https://gitlab-master.nvidia.com/haog/gqe-python/-/issues/10
    auto memory_resource = ctx_ref._query_context->parameters.use_in_memory_table_multigpu
                             ? _non_owning_table->_shared_memory_resource.get()
                             : _non_owning_table->_memory_resource.get();

    /// Create a new write task
    auto write_task = std::make_unique<in_memory_write_task>(ctx_ref,
                                                             task_parameter.task_id,
                                                             stage_id,
                                                             std::move(task_parameter.input),
                                                             memory_resource,
                                                             std::move(appender),
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
