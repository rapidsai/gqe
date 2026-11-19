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

#include <gqe/storage/in_memory_read_task.hpp>

#include "in_memory_common.hpp"

#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {

namespace storage {

namespace detail {

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

}  // namespace detail

in_memory_read_task::in_memory_read_task(context_reference ctx_ref,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::vector<const row_group_interface*> row_groups,
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

bool in_memory_read_task::is_zero_copy_possible(const pruning_results_t& pruning_results) const
{
  // The function emit_zero_copy_result assumes the positive outcome of the checks below.
  if (!get_query_context()->parameters.read_use_zero_copy) {
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
    auto layout = row_group->get_column(column_idx).layout();
    if (layout != column_layout::contiguous) {
      GQE_LOG_DEBUG(
        "Zero-copy read is not possible because at least one column is not contiguous: "
        "layout={}",
        layout);
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
    auto& column = row_group->get_column(column_idx);
    if (column.layout() != column_layout::contiguous) {
      throw std::logic_error("The column must be contiguous");
    }
    auto const full_column_view = dynamic_cast<contiguous_column_interface&>(column).view();
    const auto size             = partition.end - partition.start;
    const auto offset           = partition.start;
    const auto null_counts      = partition.null_counts[column_idx];
    const auto partitioned_column_view =
      detail::slice_column(full_column_view, size, offset, null_counts);
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
  auto result =
    builder.build(cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());
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

  auto pruning_results =
    detail::evaluate_in_memory_pruning(_row_groups,
                                       _column_indexes,
                                       _partial_filter,
                                       get_query_context()->parameters.use_partition_pruning,
                                       get_query_context()->parameters.zone_map_partition_size);

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

}  // namespace storage

}  // namespace gqe
