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

#include <gqe/qep/operator/filter_join_indices.hpp>

#include "../../executor/join_predicate.hpp"

#include <gqe/executor/eval.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/utility.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe::qep {

namespace {

/** @brief One logical join side's gather source and borrowed predicate-column views. */
struct logical_side_source {
  indirect_gather_source gather_source;
  std::vector<cudf::column_view> conditional_columns;
};

[[nodiscard]] cudf::device_span<cudf::size_type const> position_span(
  position_list_handle const& positions)
{
  auto const view = positions->column().view();
  return cudf::device_span<cudf::size_type const>{view.data<cudf::size_type>(),
                                                  static_cast<std::size_t>(view.size())};
}

[[nodiscard]] position_list_handle make_filtered_positions(
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> positions,
  query_stream const& stream_handle)
{
  GQE_EXPECTS(positions != nullptr, "filter_join_indices_task: cuDF returned null positions");
  return make_position_list(
    std::make_unique<cudf::column>(std::move(*positions), rmm::device_buffer{}, 0), stream_handle);
}

}  // namespace

filter_join_indices_task::filter_join_indices_task(
  join_type_type join_type,
  std::unique_ptr<expression> residual,
  cudf::size_type left_column_count,
  cudf::size_type right_column_count,
  mixed_join_candidate_layout layout,
  std::vector<cudf::data_type> candidate_data_types,
  std::vector<cudf::data_type> output_data_types)
  : _join_type{join_type},
    _residual{std::move(residual)},
    _left_column_count(left_column_count),
    _right_column_count(right_column_count),
    _layout(std::move(layout)),
    _candidate_data_types{std::move(candidate_data_types)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(_join_type == join_type_type::inner || _join_type == join_type_type::left,
              "filter_join_indices_task: only inner and left joins are supported",
              std::invalid_argument);
  GQE_EXPECTS(_residual != nullptr,
              "filter_join_indices_task: residual must not be null",
              std::invalid_argument);
  GQE_EXPECTS(
    _left_column_count >= 0 && _right_column_count >= 0 &&
      _left_column_count <= std::numeric_limits<cudf::size_type>::max() - _right_column_count,
    "filter_join_indices_task: logical widths must be non-negative and fit cudf::size_type",
    std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::all_of(
      _layout.logical_candidate_indices,
      [&](auto index) { return index >= 0 && index < _left_column_count + _right_column_count; }),
    "filter_join_indices_task: logical candidate index is out of range",
    std::invalid_argument);
  GQE_EXPECTS(
    !_layout.logical_candidate_indices.empty() &&
      _layout.logical_candidate_indices.size() == _candidate_data_types.size(),
    "filter_join_indices_task: candidate index and type counts must match and be non-zero",
    std::invalid_argument);
  GQE_EXPECTS(_layout.output_indices.size() == _output_data_types.size(),
              "filter_join_indices_task: output index and type counts must match",
              std::invalid_argument);
  auto const index_is_valid = [&](auto candidate_index) {
    return candidate_index >= 0 &&
           static_cast<std::size_t>(candidate_index) < _layout.logical_candidate_indices.size();
  };
  GQE_EXPECTS(std::ranges::all_of(_layout.left_conditional_indices, index_is_valid) &&
                std::ranges::all_of(_layout.right_conditional_indices, index_is_valid) &&
                std::ranges::all_of(_layout.output_indices, index_is_valid),
              "filter_join_indices_task: candidate column index is out of range",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::all_of(
      _layout.left_conditional_indices,
      [&](auto candidate_index) {
        return _layout.logical_candidate_indices[static_cast<std::size_t>(candidate_index)] <
               _left_column_count;
      }) &&
      std::ranges::all_of(
        _layout.right_conditional_indices,
        [&](auto candidate_index) {
          return _layout.logical_candidate_indices[static_cast<std::size_t>(candidate_index)] >=
                 _left_column_count;
        }),
    "filter_join_indices_task: conditional column index has the wrong logical side",
    std::invalid_argument);
  expression const* roots[] = {_residual.get()};
  for (auto index : collect_column_references(roots)) {
    auto const& conditional_indices = index < _left_column_count
                                        ? _layout.left_conditional_indices
                                        : _layout.right_conditional_indices;
    GQE_EXPECTS(std::ranges::any_of(conditional_indices,
                                    [&](auto candidate_index) {
                                      return _layout.logical_candidate_indices[candidate_index] ==
                                             index;
                                    }),
                "filter_join_indices_task: residual operand is absent from conditional columns",
                std::invalid_argument);
  }
  GQE_EXPECTS(
    std::ranges::equal(_output_data_types,
                       _layout.output_indices | std::views::transform([&](auto candidate_index) {
                         return _candidate_data_types[static_cast<std::size_t>(candidate_index)];
                       })),
    "filter_join_indices_task: output column types do not match the output schema",
    std::invalid_argument);

  // Materialization and source selection use the same fixed order as these bindings.
  auto make_binding = [&](std::span<cudf::size_type const> indices,
                          cudf::size_type logical_offset) {
    std::vector<cudf::size_type> logical_indices;
    logical_indices.reserve(indices.size());
    for (auto index : indices) {
      logical_indices.push_back(_layout.logical_candidate_indices[index] - logical_offset);
    }
    return expression_binding(logical_indices);
  };
  _conditional_indices = _layout.left_conditional_indices;
  _conditional_indices.insert(_conditional_indices.end(),
                              _layout.right_conditional_indices.begin(),
                              _layout.right_conditional_indices.end());
  _left_binding     = make_binding(_layout.left_conditional_indices, 0);
  _right_binding    = make_binding(_layout.right_conditional_indices, _left_column_count);
  _combined_binding = make_binding(_conditional_indices, 0);
}

std::unique_ptr<task> filter_join_indices_task::clone() const
{
  return std::make_unique<filter_join_indices_task>(_join_type,
                                                    _residual->clone(),
                                                    _left_column_count,
                                                    _right_column_count,
                                                    _layout,
                                                    _candidate_data_types,
                                                    _output_data_types);
}

std::optional<state_container> filter_join_indices_task::next(
  state_container_view inputs,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::filter_join_indices_task::next"};

  if (auto const plain = to_table_view(inputs); plain.has_value()) {
    GQE_EXPECTS(_join_type == join_type_type::inner,
                "filter_join_indices_task: only an empty inner candidate table may be plain");
    GQE_EXPECTS(plain->num_rows() == 0,
                "filter_join_indices_task: a plain candidate table must be empty");
    GQE_EXPECTS(plain->num_columns() == static_cast<cudf::size_type>(_candidate_data_types.size()),
                "filter_join_indices_task: empty candidate schema has the wrong column count");

    for (cudf::size_type i = 0; i < plain->num_columns(); ++i) {
      GQE_EXPECTS(plain->column(i).type() == _candidate_data_types[static_cast<std::size_t>(i)],
                  "filter_join_indices_task: empty candidate schema has the wrong type");
    }

    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  auto const indirect = indirect_table_view::try_from(inputs);
  GQE_EXPECTS(indirect.has_value(),
              "filter_join_indices_task: input must be an indirect candidate table");
  GQE_EXPECTS(indirect->mask == std::nullopt,
              "filter_join_indices_task: candidate table must not have a mask");
  GQE_EXPECTS(indirect->columns.size() == _layout.logical_candidate_indices.size() &&
                indirect->indirect_columns.size() == indirect->columns.size(),
              "filter_join_indices_task: every candidate column must be indirect");

  // The current join can inherit several sources from an earlier join feeding its probe input.
  // Evaluate through existing positions only when each side shares one compatible source.
  indirect_gather_source left_source;
  indirect_gather_source right_source;
  bool has_single_source_per_side = true;
  for (std::size_t candidate_index = 0; candidate_index < _layout.logical_candidate_indices.size();
       ++candidate_index) {
    auto const& candidate =
      *std::get<state_kind::indirect_column>(*indirect->columns[candidate_index]).state();
    auto& source = _layout.logical_candidate_indices[candidate_index] < _left_column_count
                     ? left_source
                     : right_source;
    if (!source) {
      source = candidate.gather_source();
    } else if (source->relation_source() != candidate.gather_source()->relation_source() ||
               source->position_list() != candidate.gather_source()->position_list() ||
               source->bounds_policy() != candidate.gather_source()->bounds_policy()) {
      has_single_source_per_side = false;
    }
  }

  GQE_EXPECTS(left_source != nullptr && right_source != nullptr,
              "filter_join_indices_task: each logical side needs a candidate source");

  auto const has_nullifying_inner_source =
    _join_type == join_type_type::inner &&
    (left_source->bounds_policy() == cudf::out_of_bounds_policy::NULLIFY ||
     right_source->bounds_policy() == cudf::out_of_bounds_policy::NULLIFY);
  // Gather only predicate operands when source positions cannot represent their values.
  // Projected payloads remain indirect in both paths.
  bool const needs_gathered_predicate_operands =
    !has_single_source_per_side || has_nullifying_inner_source;
  if (needs_gathered_predicate_operands) {
    return filter_with_gathered_predicate_operands(inputs, *indirect, stream_handle, mr);
  } else {
    return filter_with_source_positions(
      inputs, *indirect, left_source, right_source, stream_handle, mr);
  }
}

state_container filter_join_indices_task::filter_with_gathered_predicate_operands(
  state_container_view inputs,
  indirect_table_view const& candidates,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(_join_type == join_type_type::inner,
              "filter_join_indices_task: multiple source groups require inner join semantics");

  auto const logical     = logical_table_view::try_from(inputs);
  auto predicate_columns = materialize_columns(*logical, _conditional_indices, stream_handle, mr);
  auto const predicate_table = predicate_columns.table_view();

  // Bind the combined residual schema to one table, retaining the join AST's cast semantics.
  expression_input const input(predicate_table, _combined_binding, predicate_table.num_rows());
  auto predicate = gqe::detail::make_join_predicate(*_residual,
                                                    _left_column_count + _right_column_count,
                                                    input,
                                                    expression_input(cudf::table_view{}),
                                                    stream,
                                                    mr);
  auto mask      = cudf::compute_column(predicate_table, predicate.root(), stream, mr);

  cudf::numeric_scalar<cudf::size_type> zero{0, true, stream, mr};
  auto positions =
    make_position_list(cudf::sequence(logical->row_count(), zero, stream, mr), stream_handle);
  auto selection = compact_position_list(positions, mask->view(), stream_handle, mr);

  if (_layout.output_indices.empty()) {
    order_deallocation_after_stream_use(predicate_columns.states(), stream_handle);
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(selection->column().size());
  } else {
    std::unordered_map<indirect_gather_source_state const*, indirect_gather_source> sources;
    sources.reserve(_layout.output_indices.size());

    state_container output;
    output.reserve(_layout.output_indices.size());

    for (auto candidate_index : _layout.output_indices) {
      auto const& candidate =
        *std::get<state_kind::indirect_column>(*candidates.columns[candidate_index]).state();
      auto const& source  = candidate.gather_source();
      auto [it, inserted] = sources.try_emplace(source.get());
      if (inserted) {
        it->second = indirect_gather_source_state::make(
          source->relation_source(),
          compose_position_lists(source->position_list(), selection, stream_handle, mr),
          source->bounds_policy());
      }
      output.push_back(indirect_column_state::make(it->second, candidate.source_column_index()));
    }

    order_deallocation_after_stream_use(predicate_columns.states(), stream_handle);
    order_deallocation_after_stream_use(inputs, stream_handle);
    return output;
  }
}

state_container filter_join_indices_task::filter_with_source_positions(
  state_container_view inputs,
  indirect_table_view const& candidates,
  indirect_gather_source const& left_source,
  indirect_gather_source const& right_source,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream         = stream_handle.ref();
  auto const left_relation  = left_source->relation_source()->contiguous_table_view();
  auto const right_relation = right_source->relation_source()->contiguous_table_view();
  for (std::size_t candidate_index = 0; candidate_index < _layout.logical_candidate_indices.size();
       ++candidate_index) {
    auto const& candidate =
      *std::get<state_kind::indirect_column>(*candidates.columns[candidate_index]).state();
    auto const& relation = _layout.logical_candidate_indices[candidate_index] < _left_column_count
                             ? left_relation
                             : right_relation;
    GQE_EXPECTS(
      relation.column(candidate.source_column_index()).type() ==
        _candidate_data_types[candidate_index],
      "filter_join_indices_task: candidate column type does not match the candidate schema");
  }

  auto resolve_side = [&](indirect_gather_source const& source,
                          cudf::table_view const& relation,
                          std::span<cudf::size_type const> conditional_indices) {
    logical_side_source result{source, {}};
    result.conditional_columns.reserve(conditional_indices.size());
    for (auto const candidate_index : conditional_indices) {
      auto const& candidate = *std::get<state_kind::indirect_column>(
                                 *candidates.columns[static_cast<std::size_t>(candidate_index)])
                                 .state();
      result.conditional_columns.push_back(relation.column(candidate.source_column_index()));
    }
    return result;
  };

  auto const left  = resolve_side(left_source, left_relation, _layout.left_conditional_indices);
  auto const right = resolve_side(right_source, right_relation, _layout.right_conditional_indices);
  GQE_EXPECTS(left.gather_source.get() != right.gather_source.get(),
              "filter_join_indices_task: logical sides must have distinct gather sources");
  GQE_EXPECTS(left.gather_source->bounds_policy() == cudf::out_of_bounds_policy::DONT_CHECK,
              "filter_join_indices_task: logical-left candidates have the wrong bounds policy");
  GQE_EXPECTS(right.gather_source->bounds_policy() == (_join_type == join_type_type::left
                                                         ? cudf::out_of_bounds_policy::NULLIFY
                                                         : cudf::out_of_bounds_policy::DONT_CHECK),
              "filter_join_indices_task: logical-right candidates have the wrong bounds policy");

  auto const left_positions  = left.gather_source->position_list();
  auto const right_positions = right.gather_source->position_list();
  GQE_EXPECTS(left_positions->column().size() == right_positions->column().size(),
              "filter_join_indices_task: candidate position-list sizes differ");

  cudf::table_view const left_table(left.conditional_columns);
  cudf::table_view const right_table(right.conditional_columns);
  expression_input const left_input(left_table, _left_binding, left_table.num_rows());
  expression_input const right_input(right_table, _right_binding, right_table.num_rows());
  auto const predicate = gqe::detail::make_join_predicate(
    *_residual, _left_column_count, left_input, right_input, stream, mr);
  auto [filtered_left, filtered_right] = cudf::filter_join_indices(
    left_table,
    right_table,
    position_span(left_positions),
    position_span(right_positions),
    predicate.root(),
    _join_type == join_type_type::left ? cudf::join_kind::LEFT_JOIN : cudf::join_kind::INNER_JOIN,
    std::nullopt,
    stream,
    mr);

  auto filtered_left_positions  = make_filtered_positions(std::move(filtered_left), stream_handle);
  auto filtered_right_positions = make_filtered_positions(std::move(filtered_right), stream_handle);
  GQE_EXPECTS(filtered_left_positions->column().size() == filtered_right_positions->column().size(),
              "filter_join_indices_task: filtered position-list sizes differ");

  if (_layout.output_indices.empty()) {
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(filtered_left_positions->column().size());
  } else {
    auto const rebuild_source = [](logical_side_source const& source,
                                   position_list_handle positions) {
      return indirect_gather_source_state::make(source.gather_source->relation_source(),
                                                std::move(positions),
                                                source.gather_source->bounds_policy());
    };
    auto left_output_source  = rebuild_source(left, std::move(filtered_left_positions));
    auto right_output_source = rebuild_source(right, std::move(filtered_right_positions));

    state_container output;
    output.reserve(_layout.output_indices.size());
    for (auto const candidate_index : _layout.output_indices) {
      auto const index = static_cast<std::size_t>(candidate_index);
      auto const& candidate =
        *std::get<state_kind::indirect_column>(*candidates.columns[index]).state();
      auto source = _layout.logical_candidate_indices[index] < _left_column_count
                      ? left_output_source
                      : right_output_source;
      output.push_back(
        indirect_column_state::make(std::move(source), candidate.source_column_index()));
    }

    order_deallocation_after_stream_use(inputs, stream_handle);
    return output;
  }
}

}  // namespace gqe::qep
