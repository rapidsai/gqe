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

#include <gqe/qep/operator/conditional_semi_anti_join.hpp>

#include "../../executor/join_predicate.hpp"
#include "join_output.hpp"
#include "materialized_chunks.hpp"
#include "semi_anti_join.hpp"

#include <gqe/executor/eval.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gqe::qep {
namespace {

/** @brief Owns complete selectively materialized chunks until finalization. */
struct build_accumulator : task_private_state {
  std::mutex mutex;  ///< Protects atomic appends to `chunks`, after preparation outside the mutex.

  /** Read without locking only after the fold lifecycle ends. */
  std::vector<state_container> chunks;
};

/** @brief Immutable right columns and their original-to-compacted schema mapping. */
struct build_state : task_private_state {
  state_container columns;  ///< Owners survive all streamed probes and order frees after each use.

  detail::join_materialized_column_layout layout;
  expression_binding binding;  ///< Immutable mapping shared across probe batches.

  build_state(state_container input, detail::join_materialized_column_layout column_layout)
    : columns(std::move(input)), layout(std::move(column_layout)), binding(layout.logical_indices())
  {
  }
};

}  // namespace

conditional_semi_anti_join_build_task::conditional_semi_anti_join_build_task(
  cudf::size_type right_column_count, std::vector<cudf::size_type> right_predicate_columns)
  : _right_layout(right_column_count, {}, right_predicate_columns)
{
  GQE_EXPECTS(right_column_count > 0,
              "conditional semi/anti requires positive right width",
              std::invalid_argument);
  GQE_EXPECTS(!right_predicate_columns.empty(),
              "conditional semi/anti requires right predicate columns",
              std::invalid_argument);
}

std::unique_ptr<task> conditional_semi_anti_join_build_task::clone() const
{
  return std::make_unique<conditional_semi_anti_join_build_task>(
    _right_layout.logical_column_count(),
    std::vector<cudf::size_type>(_right_layout.logical_indices().begin(),
                                 _right_layout.logical_indices().end()));
}

state_container conditional_semi_anti_join_build_task::initialize(
  context_reference, query_stream const&, rmm::device_async_resource_ref) const
{
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<build_accumulator>()})
    .build();
}

void conditional_semi_anti_join_build_task::next(state_container_view inputs,
                                                 state_container_view accumulator,
                                                 context_reference,
                                                 query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr) const
{
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "conditional semi/anti build input must be a logical table");
  GQE_EXPECTS(logical->num_columns() == _right_layout.logical_column_count(),
              "conditional semi/anti build width differs");
  GQE_EXPECTS(accumulator.size() == 1, "conditional semi/anti build expects one accumulator");

  auto const& layout = _right_layout;
  detail::validate_semi_anti_columns(*logical, layout.logical_indices());

  auto materialized =
    materialize_compacted_columns(*logical, layout.logical_indices(), stream_handle, mr);
  auto chunk = std::move(materialized).release_column_states();

  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);
  {
    std::scoped_lock const guard(state.mutex);
    state.chunks.push_back(std::move(chunk));
  }
}

state_container conditional_semi_anti_join_build_task::finalize(
  state_container&& accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  GQE_EXPECTS(accumulator.size() == 1, "conditional semi/anti build expects one accumulator");
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);

  state_container columns;
  auto concatenated = detail::concatenate_materialized_chunks(state.chunks, stream_handle, mr);
  if (concatenated.table) {
    columns =
      state_container_builder().add_state(std::move(*concatenated.table), stream_handle).build();
  }

  return state_container_builder()
    .add_state(
      state_kind::task_private{std::make_unique<build_state>(std::move(columns), _right_layout)})
    .build();
}

conditional_semi_anti_join_probe_task::conditional_semi_anti_join_probe_task(
  cudf::size_type left_column_count,
  cudf::size_type right_column_count,
  std::unique_ptr<expression> residual,
  join_type_type join_type,
  std::vector<cudf::size_type> projection_indices,
  std::vector<cudf::data_type> output_data_types)
  : _left_column_count(left_column_count),
    _right_column_count(right_column_count),
    _residual(std::move(residual)),
    _join_type(join_type),
    _projection_indices(std::move(projection_indices)),
    _output_data_types(std::move(output_data_types)),
    _left_layout(0, {}, {})
{
  GQE_EXPECTS(
    _left_column_count > 0 && _right_column_count > 0 &&
      _left_column_count <= std::numeric_limits<cudf::size_type>::max() - _right_column_count,
    "conditional semi/anti widths must be positive and fit cudf::size_type",
    std::invalid_argument);

  GQE_EXPECTS(_residual != nullptr && gqe::detail::is_join_predicate_supported(*_residual),
              "conditional semi/anti requires a supported residual predicate",
              std::invalid_argument);
  GQE_EXPECTS(_join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti,
              "conditional semi/anti supports only left semi and left anti",
              std::invalid_argument);

  GQE_EXPECTS(_projection_indices.size() == _output_data_types.size() &&
                std::ranges::all_of(_output_data_types, detail::is_supported_semi_anti_column),
              "conditional semi/anti output metadata is unsupported",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_projection_indices,
                                  [&](auto i) { return i >= 0 && i < _left_column_count; }),
              "conditional semi/anti projects only logical-left columns",
              std::invalid_argument);

  auto operands = gqe::detail::try_collect_join_predicate_columns(
    *_residual, _left_column_count, _right_column_count);
  GQE_EXPECTS(operands.has_value(),
              "conditional semi/anti predicate column out of range",
              std::invalid_argument);
  auto const& left_predicate_columns = operands->left;
  _right_predicate_columns           = std::move(operands->right);

  GQE_EXPECTS(!left_predicate_columns.empty() && !_right_predicate_columns.empty(),
              "conditional semi/anti predicate must reference both inputs",
              std::invalid_argument);

  _left_layout =
    detail::join_materialized_column_layout(_left_column_count, {}, left_predicate_columns);
  _left_binding = expression_binding(_left_layout.logical_indices());
}

std::unique_ptr<task> conditional_semi_anti_join_probe_task::clone() const
{
  return std::make_unique<conditional_semi_anti_join_probe_task>(_left_column_count,
                                                                 _right_column_count,
                                                                 _residual->clone(),
                                                                 _join_type,
                                                                 _projection_indices,
                                                                 _output_data_types);
}

state_container conditional_semi_anti_join_probe_task::initialize(
  state_container_view inputs,
  context_reference,
  query_stream const&,
  rmm::device_async_resource_ref) const
{
  GQE_EXPECTS(inputs.size() == 1, "conditional semi/anti probe expects one build state");
  auto const& build = get_task_private_state<build_state>(inputs, 0);
  GQE_EXPECTS(build.layout.logical_column_count() == _right_column_count,
              "conditional semi/anti build configuration differs");

  // No-chunk builds select rows without evaluating a predicate.
  if (!build.columns.empty()) {
    auto const columns = build.layout.logical_indices();
    for (auto index : _right_predicate_columns) {
      GQE_EXPECTS(std::ranges::find(columns, index) != columns.end(),
                  "conditional semi/anti build is missing a predicate operand",
                  std::invalid_argument);
    }
  }

  return make_mutable_state_copy(inputs);
}

std::optional<state_container> conditional_semi_anti_join_probe_task::next(
  state_container_view inputs,
  state_container_view accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream  = stream_handle.ref();
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "conditional semi/anti probe input must be a logical table");
  GQE_EXPECTS(logical->num_columns() == _left_column_count,
              "conditional semi/anti probe width differs");
  GQE_EXPECTS(accumulator.size() == 1, "conditional semi/anti probe expects one build state");
  auto const& build = get_task_private_state<build_state>(accumulator, 0);

  // Shared output helpers use [build, probe] coordinates. An empty build layout makes
  // every projection address the logical-left probe input.
  detail::join_materialized_column_layout const empty_layout(0, {}, {});
  detail::validate_join_projection_types(
    _projection_indices, _output_data_types, {}, empty_layout, *logical);

  auto const& left_layout = _left_layout;
  detail::validate_semi_anti_columns(*logical, left_layout.logical_indices());

  auto const right = to_table_view(build.columns);
  GQE_EXPECTS(build.columns.empty() || right.has_value(),
              "conditional semi/anti build state must hold a plain table");

  position_list_handle positions;
  if (!right.has_value() || right->num_rows() == 0) {
    if (_join_type == join_type_type::left_semi) {
      return _projection_indices.empty() ? make_row_count_container(0)
                                         : make_empty_state_container(_output_data_types);
    }

    // Anti selects every valid left row without gathering predicate operands.
    auto selection =
      materialize_compacted_columns(*logical, {}, !_projection_indices.empty(), stream_handle, mr);
    if (_projection_indices.empty()) {
      order_deallocation_after_stream_use(inputs, stream_handle);
      return make_row_count_container(selection.row_count());
    }
    positions = selection.row_positions();
    if (positions == nullptr) {
      positions = make_position_list(
        cudf::sequence(selection.row_count(),
                       cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
                       stream,
                       mr),
        stream_handle);
    }
  } else {
    auto left = materialize_compacted_columns(
      *logical, left_layout.logical_indices(), !_projection_indices.empty(), stream_handle, mr);
    auto const left_columns = left.table_view();

    // Bind compacted columns without changing the predicate's logical input coordinates.
    expression_input const left_input(left_columns, _left_binding, left_columns.num_rows());
    expression_input const right_input(*right, build.binding, right->num_rows());
    auto const predicate = gqe::detail::make_join_predicate(
      *_residual, _left_column_count, left_input, right_input, stream, mr);

    auto matches =
      _join_type == join_type_type::left_semi
        ? cudf::conditional_left_semi_join(left_columns, *right, predicate.root(), {}, stream, mr)
        : cudf::conditional_left_anti_join(left_columns, *right, predicate.root(), {}, stream, mr);
    GQE_EXPECTS(matches != nullptr, "conditional semi/anti returned null positions");
    order_deallocation_after_stream_use(build.columns, stream_handle);
    order_deallocation_after_stream_use(left.states(), stream_handle);
    if (_projection_indices.empty()) {
      order_deallocation_after_stream_use(inputs, stream_handle);
      return make_row_count_container(matches->size());
    }

    positions = make_position_list(
      std::make_unique<cudf::column>(std::move(*matches), rmm::device_buffer(), 0), stream_handle);
    // Restore logical row positions after mask compaction before selecting deferred payload.
    if (auto const& rows = left.row_positions(); rows != nullptr) {
      positions = compose_position_lists(rows, positions, stream_handle, mr);
    }
  }

  auto output = detail::make_join_indirect_output(_projection_indices,
                                                  {},
                                                  empty_layout,
                                                  *logical,
                                                  positions,
                                                  positions,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream_handle,
                                                  mr);
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace gqe::qep
