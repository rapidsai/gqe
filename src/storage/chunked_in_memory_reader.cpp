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

#include <gqe/storage/chunked_in_memory_reader.hpp>

#include "in_memory_common.hpp"
#include "source_work_split.hpp"

#include <gqe/expression/expression.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
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
 * @brief Mutable state shared by concurrent calls to `chunked_in_memory_read_task::next`.
 */
struct chunked_in_memory_iterator_private : qep::task_private_state {
  std::atomic<std::size_t> next_index{0};
  std::atomic<bool> single_output_emitted{false};
  std::optional<cudf::size_type> count_only_rows;
  /// Row groups retained by pruning; each entry is claimed once using `next_index`. If
  /// GQE_USE_PARTITION_PRUNING=false, pruning_results simply returns the row groups themselves.
  std::unique_ptr<pruning_results_t> pruning_results;
};

}  // namespace

chunked_in_memory_read_task::chunked_in_memory_read_task(
  std::vector<row_group_interface const*> row_groups,
  std::vector<cudf::size_type> column_indexes,
  std::vector<cudf::data_type> output_data_types,
  qep::source_morsel_metadata source_metadata,
  std::unique_ptr<gqe::expression> partial_filter)
  : _row_groups(std::move(row_groups)),
    _column_indexes(std::move(column_indexes)),
    _source_metadata(source_metadata),
    _output_data_types(std::move(output_data_types)),
    _partial_filter(std::move(partial_filter))
{
  GQE_EXPECTS(
    std::ranges::none_of(_row_groups, [](auto const* row_group) { return row_group == nullptr; }),
    "chunked_in_memory_read_task: row_groups must not contain null",
    std::invalid_argument);
  GQE_EXPECTS(_column_indexes.size() == _output_data_types.size(),
              "chunked_in_memory_read_task: column_indexes and output_data_types must have the "
              "same length",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::none_of(
      _output_data_types,
      [](auto type) { return type.id() != cudf::type_id::EMPTY && cudf::is_nested(type); }),
    "chunked_in_memory_read_task: nested LIST and STRUCT columns are not supported because "
    "output_data_types does not describe their child schemas",
    std::invalid_argument);
  GQE_EXPECTS(!_column_indexes.empty() || _partial_filter == nullptr,
              "chunked_in_memory_read_task: partial filters are not supported in count-only mode",
              std::invalid_argument);
}

chunked_in_memory_read_task::~chunked_in_memory_read_task() = default;

qep::state_container chunked_in_memory_read_task::initialize(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"chunked_in_memory_read_task::initialize"};

  GQE_EXPECTS(inputs.empty(),
              "chunked_in_memory_read_task: source task expects no initialize inputs",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "chunked_in_memory_read_task: query context must not be null",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._task_manager_context != nullptr,
              "chunked_in_memory_read_task: task-manager context must not be null",
              std::invalid_argument);

  for (auto const* row_group : _row_groups) {
    GQE_EXPECTS(
      row_group->size() >= 0 &&
        row_group->size() <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "chunked_in_memory_read_task: row-group size exceeds cudf::size_type",
      std::overflow_error);
    for (std::size_t i = 0; i < _column_indexes.size(); ++i) {
      auto const column_index = _column_indexes[i];
      GQE_EXPECTS(
        column_index >= 0 && static_cast<int64_t>(column_index) < row_group->num_columns(),
        "chunked_in_memory_read_task: column index is outside the row-group schema",
        std::out_of_range);

      auto const& column     = row_group->get_column(column_index);
      auto const column_type = detail::cudf_type_of(column);
      GQE_EXPECTS(column_type == _output_data_types[i],
                  "chunked_in_memory_read_task: requested output type must match the stored column "
                  "type",
                  std::invalid_argument);

      auto const nulls_supported = column.layout() == column_layout::contiguous;
      GQE_EXPECTS(column.null_count() == 0 || nulls_supported,
                  "chunked_in_memory_read_task: projected column materialization does not support "
                  "null values",
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

  auto private_state = std::make_unique<chunked_in_memory_iterator_private>();
  if (_column_indexes.empty()) {
    private_state->count_only_rows = detail::checked_row_count(*pruning_results);
  } else {
    private_state->pruning_results = std::move(pruning_results);
  }

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> chunked_in_memory_read_task::next(
  qep::state_container_view iterator,
  context_reference ctx_ref,
  gqe::query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"chunked_in_memory_read_task::next"};

  GQE_EXPECTS(iterator.size() == 1,
              "chunked_in_memory_read_task: invalid iterator state",
              std::invalid_argument);
  auto* private_slot = std::get_if<qep::state_kind::task_private>(iterator.front().get());
  GQE_EXPECTS(private_slot != nullptr,
              "chunked_in_memory_read_task: invalid iterator state",
              std::invalid_argument);
  auto* iter_data = dynamic_cast<chunked_in_memory_iterator_private*>(private_slot->data.get());
  GQE_EXPECTS(iter_data != nullptr,
              "chunked_in_memory_read_task: invalid iterator private state",
              std::invalid_argument);

  if (iter_data->count_only_rows) {
    if (iter_data->single_output_emitted.exchange(true, std::memory_order_relaxed)) {
      return std::nullopt;
    }
    return qep::make_row_count_container(*iter_data->count_only_rows);
  }

  GQE_EXPECTS(iter_data->pruning_results != nullptr,
              "chunked_in_memory_read_task: pruning results must not be null",
              std::invalid_argument);

  auto const work_index = iter_data->next_index.fetch_add(1, std::memory_order_relaxed);
  if (work_index < iter_data->pruning_results->size()) {
    auto& [row_group, pruning_result] = iter_data->pruning_results->at(work_index);
    auto single_row_group             = std::make_unique<pruning_results_t>();
    single_row_group->emplace_back(row_group, std::move(pruning_result));
    output_table_builder builder{
      ctx_ref, _column_indexes, _output_data_types, std::move(single_row_group)};
    auto result = builder.build(stream, mr);
    return qep::state_container_builder().add_state(std::move(*result), stream_handle).build();
  }
  // if we reach here, we're done emitting work, and since we're at end of stream, emit nullopt
  if (!iter_data->pruning_results->empty()) { return std::nullopt; }
  // if we reach here, we never had any work, so emit a schema-correct empty container once
  if (!iter_data->single_output_emitted.exchange(true, std::memory_order_relaxed)) {
    return qep::make_empty_state_container(_output_data_types);
  }
  return std::nullopt;
}

std::size_t chunked_in_memory_read_task::work_unit_count() const noexcept
{
  return _row_groups.size();
}

std::unique_ptr<qep::iterate_task> chunked_in_memory_read_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<chunked_in_memory_read_task>(
    detail::select_source_work(_row_groups, range),
    _column_indexes,
    _output_data_types,
    _source_metadata,
    _partial_filter ? _partial_filter->clone() : nullptr);
}

std::unique_ptr<qep::task> chunked_in_memory_read_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace gqe::storage
