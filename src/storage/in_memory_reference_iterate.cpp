/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/in_memory_reference_iterate.hpp>

#include "in_memory_common.hpp"
#include "source_work_split.hpp"

#include <gqe/expression/expression.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::storage {

namespace {

/**
 * @brief One mintable work item: the table-level row-group index and its pruning result.
 */
using reference_work_item = std::pair<std::size_t, pruning_result_t>;

struct reference_iterator_private : qep::task_private_state {
  std::atomic<std::size_t> next_index{0};
  std::atomic<bool> single_output_emitted{false};
  std::optional<cudf::size_type> count_only_rows;
  std::vector<reference_work_item> work_items;
};

}  // namespace

in_memory_reference_iterate_task::in_memory_reference_iterate_task(
  std::string table_name,
  std::vector<row_group_interface const*> row_groups,
  std::vector<std::size_t> row_group_indexes,
  std::vector<cudf::size_type> column_indexes,
  std::vector<cudf::data_type> output_data_types,
  qep::source_morsel_metadata source_metadata,
  std::unique_ptr<gqe::expression> partial_filter)
  : _table_name(std::move(table_name)),
    _row_groups(std::move(row_groups)),
    _row_group_indexes(std::move(row_group_indexes)),
    _column_indexes(std::move(column_indexes)),
    _output_data_types(std::move(output_data_types)),
    _source_metadata(source_metadata),
    _partial_filter(std::move(partial_filter))
{
  GQE_EXPECTS(!_table_name.empty(),
              "in_memory_reference_iterate_task: table_name must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::none_of(_row_groups, [](auto const* row_group) { return row_group == nullptr; }),
    "in_memory_reference_iterate_task: row_groups must not contain null",
    std::invalid_argument);
  GQE_EXPECTS(_row_groups.size() == _row_group_indexes.size(),
              "in_memory_reference_iterate_task: row_groups and row_group_indexes must have the "
              "same length",
              std::invalid_argument);
  GQE_EXPECTS(_column_indexes.size() == _output_data_types.size(),
              "in_memory_reference_iterate_task: column_indexes and output_data_types must have "
              "the same length",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::none_of(
      _output_data_types,
      [](auto type) { return type.id() != cudf::type_id::EMPTY && cudf::is_nested(type); }),
    "in_memory_reference_iterate_task: nested LIST and STRUCT columns are not supported because "
    "output_data_types does not describe their child schemas",
    std::invalid_argument);
  GQE_EXPECTS(!_column_indexes.empty() || _partial_filter == nullptr,
              "in_memory_reference_iterate_task: partial filters are not supported in count-only "
              "mode",
              std::invalid_argument);
}

in_memory_reference_iterate_task::~in_memory_reference_iterate_task() = default;

qep::state_container in_memory_reference_iterate_task::initialize(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"in_memory_reference_iterate_task::initialize"};

  GQE_EXPECTS(inputs.empty(),
              "in_memory_reference_iterate_task: source task expects no initialize inputs",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "in_memory_reference_iterate_task: query context must not be null",
              std::invalid_argument);

  for (auto const* row_group : _row_groups) {
    GQE_EXPECTS(
      row_group->size() >= 0 &&
        row_group->size() <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "in_memory_reference_iterate_task: row-group size exceeds cudf::size_type",
      std::overflow_error);
    for (std::size_t i = 0; i < _column_indexes.size(); ++i) {
      auto const column_index = _column_indexes[i];
      GQE_EXPECTS(
        column_index >= 0 && static_cast<int64_t>(column_index) < row_group->num_columns(),
        "in_memory_reference_iterate_task: column index is outside the row-group schema",
        std::out_of_range);

      auto const& column     = row_group->get_column(column_index);
      auto const column_type = detail::cudf_type_of(column);
      GQE_EXPECTS(column_type == _output_data_types[i],
                  "in_memory_reference_iterate_task: requested output type must match the stored "
                  "column type",
                  std::invalid_argument);

      auto const nulls_supported =
        column.layout() == column_layout::contiguous && column_type.id() == cudf::type_id::STRING;
      GQE_EXPECTS(column.null_count() == 0 || nulls_supported,
                  "in_memory_reference_iterate_task: projected column materialization does not "
                  "support null values",
                  std::invalid_argument);
    }
  }

  auto const partial_filter = _partial_filter
                                ? zone_map_expression_transformer::transform(*_partial_filter)
                                : std::optional<arrow::compute::Expression>{};

  auto pruning_results =
    detail::evaluate_in_memory_pruning(_row_groups,
                                       _column_indexes,
                                       partial_filter,
                                       ctx_ref._query_context->parameters.use_partition_pruning,
                                       ctx_ref._query_context->parameters.zone_map_partition_size);

  auto private_state = std::make_unique<reference_iterator_private>();
  if (_column_indexes.empty()) {
    private_state->count_only_rows = detail::checked_row_count(*pruning_results);
  } else {
    // Pruning drops fully pruned row groups but preserves the input order, so a single forward
    // cursor maps each surviving row-group pointer back to its table-level index.
    private_state->work_items.reserve(pruning_results->size());
    std::size_t input_cursor = 0;
    for (auto& [row_group, pruning_result] : *pruning_results) {
      while (input_cursor < _row_groups.size() && _row_groups[input_cursor] != row_group) {
        ++input_cursor;
      }
      GQE_EXPECTS(input_cursor < _row_groups.size(),
                  "in_memory_reference_iterate_task: pruning returned an unknown row group");
      private_state->work_items.emplace_back(_row_group_indexes[input_cursor],
                                             std::move(pruning_result));
      ++input_cursor;
    }
  }

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> in_memory_reference_iterate_task::next(
  qep::state_container_view iterator,
  context_reference /* ctx_ref */,
  query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"in_memory_reference_iterate_task::next"};

  auto& iter_data = qep::get_task_private_state<reference_iterator_private>(iterator, 0);

  // Three modes: emit the count once, emit the empty-input reference once, or claim one work item.
  if (iter_data.count_only_rows) {
    if (iter_data.single_output_emitted.exchange(true, std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return qep::make_row_count_container(*iter_data.count_only_rows);
  } else if (iter_data.work_items.empty()) {
    if (iter_data.single_output_emitted.exchange(true, std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return qep::state_container_builder()
      .add_state(qep::state_kind::reference{in_memory_reference::make_empty(_table_name)})
      .build();
  } else {
    auto const work_index = iter_data.next_index.fetch_add(1, std::memory_order_relaxed);
    if (work_index >= iter_data.work_items.size()) { return std::nullopt; }
    auto const& [row_group_index, pruning_result] = iter_data.work_items[work_index];
    auto ref = in_memory_reference::make(_table_name, row_group_index, pruning_result);
    return qep::state_container_builder()
      .add_state(qep::state_kind::reference{std::move(ref)})
      .build();
  }
}

std::size_t in_memory_reference_iterate_task::work_unit_count() const noexcept
{
  return _row_groups.size();
}

std::unique_ptr<qep::iterate_task> in_memory_reference_iterate_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<in_memory_reference_iterate_task>(
    _table_name,
    detail::select_source_work(_row_groups, range),
    detail::select_source_work(_row_group_indexes, range),
    _column_indexes,
    _output_data_types,
    _source_metadata,
    _partial_filter ? _partial_filter->clone() : nullptr);
}

std::unique_ptr<qep::task> in_memory_reference_iterate_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace gqe::storage
