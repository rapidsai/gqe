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

#include <gqe/qep/operator/conditional_inner_join.hpp>

#include "../../executor/join_predicate.hpp"
#include "../../optimizer/utility.hpp"
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
#include <cudf/join/conditional_join.hpp>
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

/** @brief Immutable retained columns and their original-to-compacted schema mapping. */
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

conditional_inner_join_build_task::conditional_inner_join_build_task(
  cudf::size_type build_column_count,
  std::vector<cudf::size_type> build_predicate_columns,
  std::vector<cudf::size_type> build_projected_columns)
  : _build_layout(build_column_count, build_predicate_columns, build_projected_columns)
{
  GQE_EXPECTS(build_column_count > 0,
              "conditional inner requires positive build width",
              std::invalid_argument);
  GQE_EXPECTS(!build_predicate_columns.empty(),
              "conditional inner requires build predicate columns",
              std::invalid_argument);
}

std::unique_ptr<task> conditional_inner_join_build_task::clone() const
{
  // Build execution only needs the retained column order, not the reason each column was selected.
  return std::make_unique<conditional_inner_join_build_task>(
    _build_layout.logical_column_count(),
    std::vector<cudf::size_type>(_build_layout.logical_indices().begin(),
                                 _build_layout.logical_indices().end()),
    std::vector<cudf::size_type>{});
}

state_container conditional_inner_join_build_task::initialize(context_reference,
                                                              query_stream const&,
                                                              rmm::device_async_resource_ref) const
{
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<build_accumulator>()})
    .build();
}

void conditional_inner_join_build_task::next(state_container_view inputs,
                                             state_container_view accumulator,
                                             context_reference,
                                             query_stream const& stream_handle,
                                             rmm::device_async_resource_ref mr) const
{
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "conditional inner build input must be a logical table");
  GQE_EXPECTS(logical->num_columns() == _build_layout.logical_column_count(),
              "conditional inner build width differs");
  GQE_EXPECTS(accumulator.size() == 1, "conditional inner build expects one accumulator");

  auto const& layout = _build_layout;
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

state_container conditional_inner_join_build_task::finalize(state_container&& accumulator,
                                                            context_reference,
                                                            query_stream const& stream_handle,
                                                            rmm::device_async_resource_ref mr) const
{
  GQE_EXPECTS(accumulator.size() == 1, "conditional inner build expects one accumulator");
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);

  state_container columns;
  auto concatenated = detail::concatenate_materialized_chunks(state.chunks, stream_handle, mr);
  if (concatenated.table) {
    columns =
      state_container_builder().add_state(std::move(*concatenated.table), stream_handle).build();
  }

  return state_container_builder()
    .add_state(
      state_kind::task_private{std::make_unique<build_state>(std::move(columns), _build_layout)})
    .build();
}

conditional_inner_join_probe_task::conditional_inner_join_probe_task(
  cudf::size_type left_column_count,
  cudf::size_type right_column_count,
  std::unique_ptr<expression> residual,
  bool is_build_left,
  std::vector<cudf::size_type> projection_indices,
  std::vector<cudf::data_type> output_data_types)
  : _left_column_count(left_column_count),
    _right_column_count(right_column_count),
    _residual(std::move(residual)),
    _is_build_left(is_build_left),
    _projection_indices(std::move(projection_indices)),
    _output_data_types(std::move(output_data_types)),
    _probe_layout(0, {}, {})
{
  GQE_EXPECTS(
    _left_column_count > 0 && _right_column_count > 0 &&
      _left_column_count <= std::numeric_limits<cudf::size_type>::max() - _right_column_count,
    "conditional inner widths must be positive and fit cudf::size_type",
    std::invalid_argument);
  GQE_EXPECTS(_residual != nullptr && gqe::detail::is_join_predicate_supported(*_residual),
              "conditional inner requires a supported predicate",
              std::invalid_argument);

  auto const total = _left_column_count + _right_column_count;
  GQE_EXPECTS(
    _projection_indices.size() == _output_data_types.size() &&
      std::ranges::all_of(_output_data_types, detail::is_supported_semi_anti_column) &&
      std::ranges::all_of(_projection_indices, [total](auto i) { return i >= 0 && i < total; }),
    "conditional inner output metadata is unsupported",
    std::invalid_argument);

  auto operands = gqe::detail::try_collect_join_predicate_columns(
    *_residual, _left_column_count, _right_column_count);
  GQE_EXPECTS(
    operands.has_value(), "conditional inner predicate column out of range", std::invalid_argument);
  GQE_EXPECTS(!operands->left.empty() && !operands->right.empty(),
              "conditional inner predicate must reference both inputs",
              std::invalid_argument);
  auto const& probe_predicate_columns = _is_build_left ? operands->right : operands->left;
  _probe_layout                       = detail::join_materialized_column_layout(
    _is_build_left ? _right_column_count : _left_column_count, {}, probe_predicate_columns);
  _probe_binding = expression_binding(_probe_layout.logical_indices());

  // Output helpers use build-first execution coordinates.
  if (!_is_build_left) {
    optimizer::utility::swap_projection_indices_inplace(
      _projection_indices, _left_column_count, _right_column_count);
  }
}

std::unique_ptr<task> conditional_inner_join_probe_task::clone() const
{
  // The constructor accepts logical coordinates, so undo the stored build-first ordering.
  auto projection = _projection_indices;
  if (!_is_build_left) {
    optimizer::utility::swap_projection_indices_inplace(
      projection, _right_column_count, _left_column_count);
  }

  return std::make_unique<conditional_inner_join_probe_task>(_left_column_count,
                                                             _right_column_count,
                                                             _residual->clone(),
                                                             _is_build_left,
                                                             std::move(projection),
                                                             _output_data_types);
}

state_container conditional_inner_join_probe_task::initialize(state_container_view inputs,
                                                              context_reference,
                                                              query_stream const&,
                                                              rmm::device_async_resource_ref) const
{
  GQE_EXPECTS(inputs.size() == 1, "conditional inner probe expects one build state");
  auto const& build = get_task_private_state<build_state>(inputs, 0);
  GQE_EXPECTS(build.layout.logical_column_count() ==
                (_is_build_left ? _left_column_count : _right_column_count),
              "conditional inner build configuration differs");
  return make_mutable_state_copy(inputs);
}

std::optional<state_container> conditional_inner_join_probe_task::next(
  state_container_view inputs,
  state_container_view accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream  = stream_handle.ref();
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "conditional inner probe input must be a logical table");
  auto const probe_width = _is_build_left ? _right_column_count : _left_column_count;
  GQE_EXPECTS(logical->num_columns() == probe_width, "conditional inner probe width differs");
  GQE_EXPECTS(accumulator.size() == 1, "conditional inner probe expects one build state");
  auto const& build = get_task_private_state<build_state>(accumulator, 0);
  if (build.columns.empty()) {
    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  auto const build_columns = to_table_view(build.columns);
  GQE_EXPECTS(build_columns.has_value(), "conditional inner build must hold a plain table");
  detail::validate_join_projection_types(
    _projection_indices, _output_data_types, *build_columns, build.layout, *logical);

  auto const& probe_layout = _probe_layout;
  detail::validate_semi_anti_columns(*logical, probe_layout.logical_indices());

  // A schema-bearing empty build cannot produce pairs. Validate metadata above without
  // gathering probe operands or constructing a predicate that will never be evaluated.
  if (build_columns->num_rows() == 0) {
    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  auto const has_probe_projection = std::ranges::any_of(
    _projection_indices, [&](auto index) { return index >= build.layout.logical_column_count(); });

  auto probe = materialize_compacted_columns(
    *logical, probe_layout.logical_indices(), has_probe_projection, stream_handle, mr);

  auto const probe_columns  = probe.table_view();
  auto const left           = _is_build_left ? *build_columns : probe_columns;
  auto const right          = _is_build_left ? probe_columns : *build_columns;
  auto const& left_binding  = _is_build_left ? build.binding : _probe_binding;
  auto const& right_binding = _is_build_left ? _probe_binding : build.binding;

  // Bind compacted columns while keeping the predicate in logical left/right coordinates.
  expression_input const left_input(left, left_binding, left.num_rows());
  expression_input const right_input(right, right_binding, right.num_rows());
  auto const predicate = gqe::detail::make_join_predicate(
    *_residual, _left_column_count, left_input, right_input, stream, mr);

  auto [left_matches, right_matches] =
    cudf::conditional_inner_join(left, right, predicate.root(), {}, stream, mr);
  GQE_EXPECTS(left_matches != nullptr && right_matches != nullptr,
              "conditional inner returned null positions");
  order_deallocation_after_stream_use(build.columns, stream_handle);
  order_deallocation_after_stream_use(probe.states(), stream_handle);

  GQE_EXPECTS(left_matches->size() == right_matches->size(),
              "conditional inner match-vector sizes differ");
  GQE_EXPECTS(
    left_matches->size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "conditional inner result exceeds cudf::size_type",
    std::overflow_error);

  if (_projection_indices.empty()) {
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(static_cast<cudf::size_type>(left_matches->size()));
  }

  // cuDF returns logical left/right matches. Output helpers consume build/probe positions.
  auto build_matches   = _is_build_left ? std::move(left_matches) : std::move(right_matches);
  auto probe_matches   = _is_build_left ? std::move(right_matches) : std::move(left_matches);
  auto build_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*build_matches), rmm::device_buffer(), 0),
    stream_handle);
  auto probe_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*probe_matches), rmm::device_buffer(), 0),
    stream_handle);

  if (auto const& rows = probe.row_positions(); rows != nullptr) {
    // Restore original probe rows before selecting deferred payload columns.
    probe_positions = compose_position_lists(rows, probe_positions, stream_handle, mr);
  }

  auto output = detail::make_join_indirect_output(_projection_indices,
                                                  build.columns,
                                                  build.layout,
                                                  *logical,
                                                  build_positions,
                                                  probe_positions,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream_handle,
                                                  mr);

  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace gqe::qep
