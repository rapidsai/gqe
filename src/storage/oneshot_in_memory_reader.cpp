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

#include <gqe/storage/oneshot_in_memory_reader.hpp>

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
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::storage {

namespace {

struct oneshot_in_memory_iterator_private : qep::task_private_state {
  std::mutex mutex;
  bool emitted                    = false;
  cudf::size_type count_only_rows = 0;
  std::vector<cudf::size_type> column_indexes;
  std::unique_ptr<pruning_results_t> pruning_results;
};

}  // namespace

cudf::data_type detail::cudf_type_of(column_base const& column)
{
  switch (column.layout()) {
    case column_layout::contiguous:
      return dynamic_cast<contiguous_column_interface const&>(column).view().type();
    case column_layout::compressed_sliced:
      return dynamic_cast<compressed_sliced_column_interface const&>(column)
        .data_buffer_view()
        .element_type();
    case column_layout::string_compressed_sliced:
    case column_layout::large_string_compressed_sliced:
      return cudf::data_type{cudf::type_id::STRING};
  }
  throw std::logic_error("unknown in-memory column type");
}

oneshot_in_memory_read_task::oneshot_in_memory_read_task(
  std::vector<row_group_interface const*> row_groups,
  std::vector<cudf::size_type> column_indexes,
  std::vector<cudf::data_type> output_data_types,
  std::unique_ptr<gqe::expression> partial_filter)
  : _row_groups(std::move(row_groups)),
    _column_indexes(std::move(column_indexes)),
    _output_data_types(std::move(output_data_types)),
    _partial_filter(std::move(partial_filter))
{
  GQE_EXPECTS(
    std::all_of(
      _row_groups.cbegin(), _row_groups.cend(), [](auto const* rg) { return rg != nullptr; }),
    "oneshot_in_memory_read_task: row groups must not contain null",
    std::invalid_argument);
  GQE_EXPECTS(_column_indexes.size() == _output_data_types.size(),
              "oneshot_in_memory_read_task: column_indexes and output_data_types must have the "
              "same length",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::none_of(
      _output_data_types,
      [](auto type) { return type.id() != cudf::type_id::EMPTY && cudf::is_nested(type); }),
    "oneshot_in_memory_read_task: nested LIST and STRUCT columns are not supported because "
    "output_data_types does not describe their child schemas",
    std::invalid_argument);
  GQE_EXPECTS(!_column_indexes.empty() || _partial_filter == nullptr,
              "oneshot_in_memory_read_task: partial filters are not supported in count-only mode",
              std::invalid_argument);
}

oneshot_in_memory_read_task::~oneshot_in_memory_read_task() = default;

qep::state_container oneshot_in_memory_read_task::initialize(
  qep::state_container_view inputs,
  context_reference ctx_ref,
  gqe::query_stream const& /* stream_handle */,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"oneshot_in_memory_read_task::initialize"};

  GQE_EXPECTS(inputs.empty(),
              "oneshot_in_memory_read_task: source task expects no initialize inputs",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "oneshot_in_memory_read_task: query context must not be null",
              std::invalid_argument);
  GQE_EXPECTS(ctx_ref._task_manager_context != nullptr,
              "oneshot_in_memory_read_task: task-manager context must not be null",
              std::invalid_argument);

  std::vector<std::optional<column_layout>> expected_layouts(_column_indexes.size());
  for (auto const* row_group : _row_groups) {
    for (std::size_t i = 0; i < _column_indexes.size(); ++i) {
      auto const column_index = _column_indexes[i];
      GQE_EXPECTS(
        column_index >= 0 && static_cast<int64_t>(column_index) < row_group->num_columns(),
        "oneshot_in_memory_read_task: column index is outside the row-group schema",
        std::out_of_range);
      auto const& column     = row_group->get_column(column_index);
      auto const column_type = detail::cudf_type_of(column);
      GQE_EXPECTS(column_type == _output_data_types[i],
                  "oneshot_in_memory_read_task: requested output type must match the stored column "
                  "type",
                  std::invalid_argument);
      auto const nulls_supported = column.layout() == column_layout::contiguous;
      GQE_EXPECTS(column.null_count() == 0 || nulls_supported,
                  "oneshot_in_memory_read_task: projected column materialization does not support "
                  "null values",
                  std::invalid_argument);
      if (expected_layouts[i]) {
        GQE_EXPECTS(
          column.layout() == *expected_layouts[i] ||
            (detail::is_compressed_string_layout(column.layout()) &&
             detail::is_compressed_string_layout(*expected_layouts[i])),
          "oneshot_in_memory_read_task: projected column storage layouts must match across row "
          "groups",
          std::invalid_argument);
      } else {
        expected_layouts[i] = column.layout();
      }
    }
  }

  auto partial_filter = _partial_filter
                          ? zone_map_expression_transformer::transform(*_partial_filter)
                          : std::optional<arrow::compute::Expression>{};

  auto pruning_results =
    detail::evaluate_in_memory_pruning(_row_groups,
                                       _column_indexes,
                                       partial_filter,
                                       ctx_ref._query_context->parameters.use_partition_pruning,
                                       ctx_ref._query_context->parameters.zone_map_partition_size);

  auto private_state = std::make_unique<oneshot_in_memory_iterator_private>();
  if (_column_indexes.empty()) {
    private_state->count_only_rows = detail::checked_row_count(*pruning_results);
  } else {
    private_state->pruning_results = std::move(pruning_results);
  }
  private_state->column_indexes = _column_indexes;

  return qep::state_container_builder()
    .add_state(qep::state_kind::task_private{std::move(private_state)})
    .build();
}

std::optional<qep::state_container> oneshot_in_memory_read_task::next(
  qep::state_container_view iterator,
  context_reference ctx_ref,
  gqe::query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"oneshot_in_memory_read_task::next"};

  GQE_EXPECTS(iterator.size() == 1,
              "oneshot_in_memory_read_task: invalid iterator state",
              std::invalid_argument);
  auto* private_slot = std::get_if<qep::state_kind::task_private>(iterator.front().get());
  GQE_EXPECTS(private_slot != nullptr,
              "oneshot_in_memory_read_task: invalid iterator state",
              std::invalid_argument);
  auto* iter_data = dynamic_cast<oneshot_in_memory_iterator_private*>(private_slot->data.get());
  GQE_EXPECTS(iter_data != nullptr,
              "oneshot_in_memory_read_task: invalid iterator private state",
              std::invalid_argument);

  {
    std::lock_guard const guard{iter_data->mutex};
    if (iter_data->emitted) { return std::nullopt; }
    iter_data->emitted = true;
  }

  if (iter_data->column_indexes.empty()) {
    return qep::make_row_count_container(iter_data->count_only_rows);
  }

  assert(iter_data->pruning_results != nullptr);
  if (iter_data->pruning_results->empty()) {
    return qep::make_empty_state_container(_output_data_types);
  }

  output_table_builder builder{
    ctx_ref, iter_data->column_indexes, _output_data_types, std::move(iter_data->pruning_results)};
  auto result = builder.build(stream, mr);
  return qep::state_container_builder().add_state(std::move(*result), stream_handle).build();
}

std::size_t oneshot_in_memory_read_task::work_unit_count() const noexcept
{
  return _row_groups.size();
}

std::unique_ptr<qep::iterate_task> oneshot_in_memory_read_task::split(
  qep::source_work_range range) const
{
  return std::make_unique<oneshot_in_memory_read_task>(
    detail::select_source_work(_row_groups, range),
    _column_indexes,
    _output_data_types,
    _partial_filter ? _partial_filter->clone() : nullptr);
}

std::unique_ptr<qep::task> oneshot_in_memory_read_task::clone() const
{
  return split({0, work_unit_count()});
}

}  // namespace gqe::storage
