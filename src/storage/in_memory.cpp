/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
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
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock
#include <numeric>
#include <shared_mutex>  // std::shared_lock
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {

namespace storage {

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

plain_buffer::plain_buffer(rmm::device_buffer const* input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  _buffer = std::make_unique<rmm::device_buffer>(*input, stream, mr);
}

std::unique_ptr<rmm::device_buffer> plain_buffer::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return std::make_unique<rmm::device_buffer>(*_buffer, stream, mr);
}

std::unique_ptr<rmm::device_buffer> compressed_column::compress(rmm::device_buffer const* input,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr,
                                                                bool is_null_mask)
{
  std::unique_ptr<rmm::device_buffer> output;
  if (is_null_mask) {
    output = _nvcomp_manager.do_compress(
      input, _null_mask_compression_ratio, _is_null_mask_compressed, stream, mr);
  } else {
    output = _nvcomp_manager.do_compress(input, _compression_ratio, _is_compressed, stream, mr);
  }
  return output;
}

compressed_column::compressed_column(cudf::column&& cudf_column,
                                     compression_format comp_format,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     nvcompType_t nvcomp_data_format,
                                     int chunk_size)
  : column_base(),
    _comp_format(comp_format),
    _compression_ratio(0.0),
    _null_mask_compression_ratio(0.0),
    _is_compressed(false),
    _is_null_mask_compressed(false),
    _nvcomp_manager(comp_format, nvcomp_data_format, chunk_size)
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

    _compressed_children.push_back(std::make_unique<compressed_column>(
      std::move(*child), comp_format, stream, mr, nvcomp_data_format, chunk_size));
  }
}

int64_t compressed_column::size() const { return _size; }

std::unique_ptr<cudf::column> compressed_column::decompress(rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> decompressed_children;
  decompressed_children.reserve(_compressed_children.size());

  for (auto const& compressed_child : _compressed_children) {
    decompressed_children.push_back(compressed_child->decompress(stream, mr));
  }

  std::unique_ptr<rmm::device_buffer> decompressed_data;

  if (_is_compressed) {
    decompressed_data = _nvcomp_manager.do_decompress(_compressed_data.get(), stream, mr);
  } else {
    decompressed_data = std::make_unique<rmm::device_buffer>(*_compressed_data, stream, mr);
  }

  std::unique_ptr<rmm::device_buffer> decompressed_null_mask;
  if (_null_count > 0) {
    if (_is_null_mask_compressed) {
      decompressed_null_mask =
        _nvcomp_manager.do_decompress(_compressed_null_mask.get(), stream, mr);
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

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns)
  : _columns(std::move(columns))
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

column_base& row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

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
  std::unique_ptr<rmm::mr::device_memory_resource> mr = std::visit(
    utility::overloaded{
      [](memory_kind::system) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::system_memory_resource>();
      },
      [](const memory_kind::numa& numa) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::numa_memory_resource>(numa.numa_node_set,
                                                                       numa.page_kind);
      },
      [](const memory_kind::pinned& numa) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::pinned_memory_resource>();
      },
      [](const memory_kind::device& device) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        // FIXME: specify device instead of allocating on default CUDA device
        return std::make_unique<rmm::mr::cuda_memory_resource>();
      },
      [](const memory_kind::managed) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::unique_ptr<rmm::mr::managed_memory_resource>();
      }},
    memory_kind);

  _memory_resource = std::move(mr);
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
}

void in_memory_read_task::execute_read_by_value()
{
  auto mr     = rmm::mr::get_current_device_resource();
  auto stream = cudf::get_default_stream();

  std::unique_ptr<cudf::table> result_table;
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(_column_indexes.size());
  bool use_overlap_mtx = get_query_context()->parameters.use_overlap_mtx;

  auto collect_src_views = [this](cudf::size_type idx, rmm::cuda_stream_view decompress_stream) {
    std::vector<cudf::column_view> src_views;
    src_views.reserve(_row_groups.size());
    std::vector<std::unique_ptr<cudf::column>>
      decompressed_columns;  // Needed for keeping the decompressed column alive
    for (auto const& row_group : this->_row_groups) {
      auto& current_column = row_group->get_column(idx);

      switch (current_column.type()) {
        case in_memory_column_type::CONTIGUOUS: {
          auto src_view = dynamic_cast<contiguous_column&>(current_column).view();
          src_views.push_back(src_view);
          break;
        }
        case in_memory_column_type::COMPRESSED: {
          auto decompressed_column =
            dynamic_cast<compressed_column&>(current_column).decompress(decompress_stream);
          src_views.push_back(decompressed_column->view());
          decompressed_columns.push_back(std::move(decompressed_column));
          break;
        }
      }
    }
    return std::make_tuple(src_views, std::move(decompressed_columns));
  };

  if (use_overlap_mtx) {
    GQE_LOG_TRACE("Using overlap mutex for in_memory_read_task");
    auto& shared_ce_stream = get_context_reference()._task_manager_context->copy_engine_stream;
    cudaEvent_t ce_evt;
    GQE_CUDA_TRY(cudaEventCreateWithFlags(&ce_evt, cudaEventDisableTiming));
    {
      const std::lock_guard lock{shared_ce_stream.mtx};
      // For each table column, concatenate the row groups into a single cudf::column
      for (auto const& idx : _column_indexes) {
        auto [src_views, decompressed_columns] = collect_src_views(idx, shared_ce_stream.stream);
        std::unique_ptr<cudf::column> result_column;

        /**
         * If column chunks are compressed we need to wait on memcopy +
         * decompression before we concatenate.
         */
        if (!decompressed_columns.empty()) {
          GQE_CUDA_TRY(cudaEventRecord(ce_evt, shared_ce_stream.stream.value()));
          GQE_CUDA_TRY(cudaStreamWaitEvent(stream.value(), ce_evt));
        }

        /**
         * If the column is not compressed we do a CE copy in concatenate, we want this copy
         * to be done on the shared CE stream to allow for pipelining.
         *
         * Otherwise, it is better to do this on the thread specific stream to avoid false
         * dependence. Note: The shared CE stream guarantees serial execution, if more than one
         * row group is compressed then the ce_evt will have overwrites. This is intended as we
         * only need to wait on the "last" scheduled copy on the shared CE stream.
         */
        if (decompressed_columns.empty()) {
          result_column = cudf::concatenate(src_views, shared_ce_stream.stream, mr);
          GQE_CUDA_TRY(cudaEventRecord(ce_evt, shared_ce_stream.stream.value()));
        } else {
          result_column = cudf::concatenate(src_views, stream, mr);
        }

        result_columns.emplace_back(std::move(result_column));
      }
    }

    /**
     * If all columns in all row groups are compressed this event will have no "extra" recorded
     * work and is a no-op.
     */
    GQE_CUDA_TRY(cudaStreamWaitEvent(stream.value(), ce_evt));
    GQE_CUDA_TRY(cudaEventDestroy(ce_evt));
  } else {
    // For each table column, concatenate the row groups into a single cudf::column
    for (auto const& idx : _column_indexes) {
      auto [src_views, decompressed_columns] = collect_src_views(idx, stream);

      std::unique_ptr<cudf::column> result_column = cudf::concatenate(src_views, stream, mr);
      result_columns.emplace_back(std::move(result_column));
    }
  }
  // Emit the result
  result_table = std::make_unique<cudf::table>(std::move(result_columns));
  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_value, "
    "output_size={}.",
    task_id(),
    stage_id(),
    result_table->num_rows());
  emit_result(std::move(result_table));
}

void in_memory_read_task::execute_read_by_reference()
{
  if (_row_groups.size() > 1) {
    throw std::logic_error(
      "Zero-copy read is only possible if columns have a contiguous "
      "memory layout. Cannot handle multiple row groups.");
  }

  std::vector<cudf::column_view> result_columns;
  result_columns.reserve(_column_indexes.size());

  for (auto const& idx : _column_indexes) {
    auto view = dynamic_cast<contiguous_column&>(_row_groups.front()->get_column(idx)).view();
    result_columns.push_back(view);
  }

  auto result = cudf::table_view(std::move(result_columns));

  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_reference, "
    "output_size={}.",
    task_id(),
    stage_id(),
    result.num_rows());
  emit_result(result);
}

void in_memory_read_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range in_memory_read_task_range("in_memory_read_task");

  // Check if zero-copy is legal.
  auto ctx_ref             = get_context_reference();
  auto& device_prop        = ctx_ref._task_manager_context->_device_properties;
  bool is_gpu_accessible   = memory_kind::is_gpu_accessible(device_prop, _memory_kind);
  bool is_single_row_group = _row_groups.size() <= 1;
  bool is_compressed =
    get_query_context()->parameters.in_memory_table_compression_format != compression_format::none;

  // Execute read.
  if (get_query_context()->parameters.read_zero_copy_enable) {
    if (_force_zero_copy_disable) {
      GQE_LOG_INFO("Disabling zero-copy read due to override by the executor for performance.");
      execute_read_by_value();
    } else if (!is_gpu_accessible) {
      GQE_LOG_WARN(
        "Disabling zero-copy read because the GPU cannot access pageable memory on this "
        "system.");
      execute_read_by_value();
    } else if (!is_single_row_group) {
      GQE_LOG_WARN("Disabling zero-copy because cannot handle more than one row group per task.");
      execute_read_by_value();
    } else if (is_compressed) {
      GQE_LOG_WARN("Disabling zero-copy because cannot handle compressed tables.");
      execute_read_by_value();
    } else {
      GQE_LOG_DEBUG("Performing zero-copy read.");
      execute_read_by_reference();
    }
  } else {
    execute_read_by_value();
  }

  remove_dependencies();
}

in_memory_write_task::in_memory_write_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> input,
  rmm::mr::device_memory_resource* non_owned_memory_resource,
  in_memory_table::row_group_appender appender,
  std::vector<cudf::size_type> column_indexes,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
  : write_task_base(ctx_ref, task_id, stage_id, input),
    _non_owned_memory_resource(non_owned_memory_resource),
    _appender(std::move(appender)),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _statistics(statistics)
{
}

void in_memory_write_task::execute()
{
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

  // FIXME: Maybe we can zero-copy construct (i.e., move to) the new columns if
  // the input_table columns have the same memory type as the result columns.
  // This would require us to destroy the result cache here, but that should be
  // ok because `write` is the root task.
  std::vector<std::unique_ptr<column_base>> new_columns(_column_indexes.size());
  for (decltype(input_table.num_columns()) column_idx = 0; column_idx < input_table.num_columns();
       ++column_idx) {
    auto const& input_column = input_table.column(column_idx);
    auto cudf_column         = cudf::column(input_column, stream, _non_owned_memory_resource);

    auto comp_format      = get_query_context()->parameters.in_memory_table_compression_format;
    auto const chunk_size = get_query_context()->parameters.compression_chunk_size;

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
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<contiguous_column>(std::move(cudf_column));
    } else {
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<compressed_column>(std::move(cudf_column),
                                            comp_format,
                                            stream,
                                            _non_owned_memory_resource,
                                            nvcomp_data_format,
                                            chunk_size);
    }
  }

  // Create row group from columns
  auto row_group = storage::row_group(std::move(new_columns));

  // Append row group to table
  _appender(std::move(row_group));
  _statistics->add_rows(input_table.num_rows());

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
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
                                                             std::move(task->partial_filter),
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
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

    // Create a new write task
    auto write_task =
      std::make_unique<in_memory_write_task>(ctx_ref,
                                             task_parameter.task_id,
                                             stage_id,
                                             std::move(task_parameter.input),
                                             _non_owning_table->_memory_resource.get(),
                                             std::move(appender),
                                             column_indexes,
                                             data_types,
                                             statistics);
    write_tasks.push_back(std::move(write_task));
  }

  return write_tasks;
}

}  // namespace storage

}  // namespace gqe
