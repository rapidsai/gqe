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

#include <gqe/storage/in_memory_readable_view.hpp>

#include <gqe/expression/json_formatter.hpp>
#include <gqe/storage/chunked_in_memory_reader.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/in_memory_read_task.hpp>
#include <gqe/storage/in_memory_reference_iterate.hpp>
#include <gqe/storage/in_memory_resolve.hpp>
#include <gqe/storage/oneshot_in_memory_reader.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <numeric>

namespace gqe {

namespace storage {

namespace {

/// Transform a filter expression into an equivalent expression for the zone map.
/// Returns the transformed expression; or std::nullopt, if:
/// 1) the filter expression is nullptr
/// 2) the filter expression cannot be transformed
std::optional<arrow::compute::Expression> transform_partial_filter(gqe::expression* partial_filter)
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

/**
 * @brief Return the source metadata of a table whose memory keeps one residency.
 *
 * Chunked and deferred readers report one residency for all of their output. Managed memory may
 * migrate between host and device, so these readers reject it.
 *
 * @throws std::invalid_argument If `memory` is managed.
 */
qep::source_morsel_metadata fixed_residency_metadata(memory_kind::type const& memory)
{
  GQE_EXPECTS(
    !std::holds_alternative<memory_kind::managed>(memory),
    "in_memory_readable_view: chunked and deferred readers require fixed source residency",
    std::invalid_argument);
  return {.residency = std::holds_alternative<memory_kind::device>(memory)
                         ? qep::data_residency::device
                         : qep::data_residency::host};
}

}  // namespace

in_memory_readable_view::in_memory_readable_view(in_memory_table_interface* non_owning_table)
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
  std::shared_lock latch_guard(_non_owning_table->row_group_latch());
  auto const nrow_groups_unsigned = _non_owning_table->num_row_groups();

  assert(nrow_groups_unsigned <=
         static_cast<decltype(nrow_groups_unsigned)>(std::numeric_limits<int64_t>::max()));
  const auto nrow_groups = static_cast<int64_t>(nrow_groups_unsigned);

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
      std::vector<const row_group_interface*> row_groups_chunk;
      assert(end_offset >= begin_offset);
      row_groups_chunk.reserve(end_offset - begin_offset);
      for (auto row_group_idx = begin_offset; row_group_idx < end_offset; ++row_group_idx) {
        row_groups_chunk.push_back(
          _non_owning_table->row_group_at(static_cast<std::size_t>(row_group_idx)));
      }

      // Create a new read task
      auto read_task = std::make_unique<in_memory_read_task>(ctx_ref,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(row_groups_chunk),
                                                             column_indexes,
                                                             data_types,
                                                             _non_owning_table->get_memory_kind(),
                                                             zone_map_filter,
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
}

qep_read_pipeline in_memory_readable_view::get_read_pipeline(
  source_reader_type reader,
  std::string table_name,
  table_provider const* provider,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  std::optional<cudf::size_type> partition_column_idx,
  std::unique_ptr<gqe::expression> partial_filter)
{
  GQE_EXPECTS(data_types.size() == column_names.size(),
              "data_types must have the same length as the number of columns",
              std::length_error);
  GQE_EXPECTS(!partition_column_idx.has_value(),
              "in-memory QEP reads do not support partition-column pruning",
              std::logic_error);

  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](std::string const& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  std::vector<row_group_interface const*> row_groups;
  {
    std::shared_lock latch_guard(_non_owning_table->row_group_latch());
    auto const num_row_groups = _non_owning_table->num_row_groups();
    row_groups.reserve(num_row_groups);
    for (std::size_t row_group_idx = 0; row_group_idx < num_row_groups; ++row_group_idx) {
      row_groups.push_back(_non_owning_table->row_group_at(row_group_idx));
    }
  }

  switch (reader) {
    case source_reader_type::oneshot:
      return {.source  = std::make_unique<oneshot_in_memory_read_task>(std::move(row_groups),
                                                                      std::move(column_indexes),
                                                                      std::move(data_types),
                                                                      std::move(partial_filter)),
              .resolve = nullptr};
    case source_reader_type::chunked:
      return {.source = std::make_unique<chunked_in_memory_read_task>(
                std::move(row_groups),
                std::move(column_indexes),
                std::move(data_types),
                fixed_residency_metadata(_non_owning_table->get_memory_kind()),
                std::move(partial_filter)),
              .resolve = nullptr};
    case source_reader_type::deferred: {
      auto const metadata = fixed_residency_metadata(_non_owning_table->get_memory_kind());
      // Preserve catalog positions even after the iterator prunes or splits its row groups.
      std::vector<std::size_t> row_group_indexes(row_groups.size());
      std::iota(row_group_indexes.begin(), row_group_indexes.end(), std::size_t{0});
      auto resolve = std::make_unique<in_memory_resolve_task>(provider, column_indexes, data_types);
      auto source  = std::make_unique<in_memory_reference_iterate_task>(std::move(table_name),
                                                                       std::move(row_groups),
                                                                       std::move(row_group_indexes),
                                                                       std::move(column_indexes),
                                                                       std::move(data_types),
                                                                       metadata,
                                                                       std::move(partial_filter));
      return {.source = std::move(source), .resolve = std::move(resolve)};
    }
  }
  throw std::invalid_argument("in_memory_readable_view: invalid source reader");
}

}  // namespace storage
}  // namespace gqe
