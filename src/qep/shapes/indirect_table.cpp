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

#include <gqe/qep/shapes/indirect_table.hpp>

#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/types.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Source pair retained while rebuilding one chunk-specific gather source.
 *
 * `original` retains the position list read by the asynchronous copy. `sliced` owns the copied
 * chunk position list used by the rebuilt indirect-column descriptors.
 */
struct sliced_gather_source {
  indirect_gather_source original;
  indirect_gather_source sliced;
};

}  // namespace

indirect_table_source_state::indirect_table_source_state(
  detail::passkey<indirect_table_source_state>, state_container columns)
  : _columns{std::move(columns)}
{
  GQE_EXPECTS(!_columns.empty(),
              "indirect_table_source_state: source must contain at least one column",
              std::invalid_argument);
  GQE_EXPECTS(
    _columns.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "indirect_table_source_state: source column count exceeds cudf::size_type",
    std::invalid_argument);
  _num_columns = static_cast<cudf::size_type>(_columns.size());

  std::optional<cudf::size_type> row_count;
  for (auto const& column : _columns) {
    // A null buried in this container is malformed state (logic_error from the recognizer);
    // direct null handle arguments to sibling constructors are invalid_argument.
    auto const view = to_column_view(column);
    GQE_EXPECTS(view.has_value(),
                "indirect_table_source_state: state must be a cudf_column or cudf_column_view",
                std::invalid_argument);
    if (!row_count.has_value()) {
      row_count = view->size();
    } else {
      GQE_EXPECTS(*row_count == view->size(),
                  "indirect_table_source_state: source column row counts do not match",
                  std::invalid_argument);
    }
    detail::append_unique_deallocation_streams(_deallocation_streams,
                                               column.deallocation_streams());
  }
  _row_count = *row_count;
}

cudf::table_view indirect_table_source_state::contiguous_table_view() const
{
  // The constructor validates this shape; guard against contract-violating mutation of the
  // retained states.
  auto table = to_table_view(_columns);
  GQE_EXPECTS(table.has_value(),
              "indirect_table_source_state::contiguous_table_view: failed to produce a concrete "
              "table");
  return *table;
}

cudf::column_view indirect_table_source_state::column(cudf::size_type source_column_index) const
{
  GQE_EXPECTS(source_column_index >= 0 && source_column_index < _num_columns,
              "indirect_table_source_state::column: source column index is out of range",
              std::out_of_range);
  auto const result = to_column_view(_columns[static_cast<std::size_t>(source_column_index)]);
  GQE_EXPECTS(result.has_value(),
              "indirect_table_source_state::column: retained source column is no longer direct");
  return *result;
}

indirect_table_source indirect_table_source_state::make(state_container columns)
{
  return std::make_shared<indirect_table_source_state const>(
    detail::passkey<indirect_table_source_state>{}, std::move(columns));
}

indirect_gather_source_state::indirect_gather_source_state(
  detail::passkey<indirect_gather_source_state>,
  indirect_table_source relation,
  position_list_handle positions,
  cudf::out_of_bounds_policy policy)
  : _relation{std::move(relation)}, _positions{std::move(positions)}, _policy{policy}
{
  GQE_EXPECTS(_relation != nullptr,
              "indirect_gather_source_state: relation must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_positions != nullptr,
              "indirect_gather_source_state: positions must not be null",
              std::invalid_argument);
  detail::append_unique_deallocation_streams(_deallocation_streams,
                                             _relation->deallocation_streams());
  detail::append_unique_deallocation_streams(_deallocation_streams,
                                             _positions->deallocation_streams());
}

indirect_gather_source indirect_gather_source_state::make(indirect_table_source relation,
                                                          position_list_handle positions,
                                                          cudf::out_of_bounds_policy policy)
{
  return std::make_shared<indirect_gather_source_state const>(
    detail::passkey<indirect_gather_source_state>{},
    std::move(relation),
    std::move(positions),
    policy);
}

indirect_column_state::indirect_column_state(detail::passkey<indirect_column_state>,
                                             indirect_gather_source gather_source,
                                             cudf::size_type source_column_index)
  : _gather_source{std::move(gather_source)}, _source_column_index{source_column_index}
{
  GQE_EXPECTS(_gather_source != nullptr,
              "indirect_column_state: gather source must not be null",
              std::invalid_argument);
  GQE_EXPECTS(_source_column_index >= 0 &&
                _source_column_index < _gather_source->relation_source()->num_columns(),
              "indirect_column_state: source column index is out of range",
              std::invalid_argument);
}

shared_state indirect_column_state::make(indirect_gather_source gather_source,
                                         cudf::size_type source_column_index)
{
  auto state = std::make_shared<indirect_column_state const>(
    detail::passkey<indirect_column_state>{}, std::move(gather_source), source_column_index);
  return make_shared_state(state_kind::indirect_column{std::move(state)});
}

std::optional<indirect_table_view> indirect_table_view::try_from(state_container_view container)
{
  auto const logical = logical_table_view::try_from(container);
  if (!logical.has_value()) { return std::nullopt; }
  bool const claims_shape = std::ranges::any_of(logical->column_states(), [](auto const& slot) {
    return std::holds_alternative<state_kind::indirect_column>(*slot);
  });
  if (!claims_shape) { return std::nullopt; }

  indirect_table_view result;
  result.row_count = logical->row_count();
  result.mask      = logical->valid_mask();
  result.mask_state =
    logical->valid_mask_state() != nullptr ? *logical->valid_mask_state() : shared_state{};
  result.columns.assign(logical->column_states().begin(), logical->column_states().end());
  result.indirect_columns.reserve(result.columns.size());

  for (std::size_t output_index = 0; output_index < result.columns.size(); ++output_index) {
    if (auto const* indirect =
          std::get_if<state_kind::indirect_column>(result.columns[output_index].get())) {
      auto const& indirect_state     = indirect->state();
      auto const& gather_source      = indirect_state->gather_source();
      auto const source_column_index = indirect_state->source_column_index();
      result.indirect_columns.push_back(
        indirect_column_view{output_index, gather_source, source_column_index});
    }
  }
  return result;
}

state_container slice_indirect_table(indirect_table_view const& input,
                                     cudf::size_type begin,
                                     cudf::size_type end,
                                     query_stream const& stream_handle,
                                     rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(begin >= 0 && begin <= end && end <= input.row_count,
              "slice_indirect_table: range is outside the indirect table",
              std::invalid_argument);

  state_container output;
  output.reserve(input.columns.size() + (input.mask.has_value() ? 1 : 0));
  if (begin == 0 && end == input.row_count) {
    if (input.mask.has_value()) {
      GQE_EXPECTS(input.mask_state != nullptr,
                  "slice_indirect_table: validated mask has no source state");
      output.push_back(input.mask_state);
    }
    output.insert(output.end(), input.columns.begin(), input.columns.end());
    return output;
  }

  if (input.mask.has_value()) {
    GQE_EXPECTS(input.mask_state != nullptr,
                "slice_indirect_table: validated mask has no source state");
    auto const mask = cudf::slice(*input.mask, {begin, end}, stream).front();
    output.push_back(make_anchored_state(input.mask_state, state_kind::valid_mask_view{mask}));
  }

  std::vector<sliced_gather_source> sliced_sources;
  sliced_sources.reserve(input.indirect_columns.size());
  std::unordered_map<indirect_gather_source_state const*, std::size_t> sliced_source_indices;
  sliced_source_indices.reserve(input.indirect_columns.size());
  auto indirect_column = input.indirect_columns.begin();
  for (std::size_t output_index = 0; output_index < input.columns.size(); ++output_index) {
    if (indirect_column == input.indirect_columns.end() ||
        indirect_column->output_index != output_index) {
      auto const direct = to_column_view(input.columns[output_index]);
      GQE_EXPECTS(direct.has_value(),
                  "slice_indirect_table: validated direct column is no longer concrete");
      auto const sliced = cudf::slice(*direct, {begin, end}, stream).front();
      output.push_back(
        make_anchored_state(input.columns[output_index], state_kind::cudf_column_view{sliced}));
      continue;
    }

    auto const source = indirect_column->gather_source;
    auto rebuilt      = sliced_source_indices.find(source.get());
    if (rebuilt == sliced_source_indices.end()) {
      auto const positions =
        cudf::slice(source->position_list()->column().view(), {begin, end}, stream).front();
      auto owned_positions = std::make_unique<cudf::column>(positions, stream, mr);
      auto sliced_source   = indirect_gather_source_state::make(
        source->relation_source(),
        make_position_list(std::move(owned_positions), stream_handle),
        source->bounds_policy());
      sliced_sources.push_back(sliced_gather_source{source, std::move(sliced_source)});
      rebuilt = sliced_source_indices.emplace(source.get(), sliced_sources.size() - 1).first;
    }

    auto state = indirect_column_state::make(sliced_sources[rebuilt->second].sliced,
                                             indirect_column->source_column_index);
    output.push_back(make_anchored_state(input.columns[output_index], std::move(state)));
    ++indirect_column;
  }

  GQE_EXPECTS(indirect_column == input.indirect_columns.end(),
              "slice_indirect_table: failed to rebuild every indirect column");
  return output;
}

}  // namespace qep
}  // namespace gqe
