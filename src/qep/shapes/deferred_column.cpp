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

#include <gqe/qep/shapes/deferred_column.hpp>

#include <gqe/utility/error.hpp>

#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

// -----------------------------------------------------------------------------
// deferred_table_source_state
// -----------------------------------------------------------------------------

deferred_table_source_state::deferred_table_source_state(cudf::size_type num_rows,
                                                         std::vector<cudf::data_type> column_types)
  : _num_rows{num_rows}, _column_types{std::move(column_types)}
{
  GQE_EXPECTS(num_rows >= 0,
              "deferred_table_source_state: num_rows must not be negative",
              std::invalid_argument);
  GQE_EXPECTS(!_column_types.empty(),
              "deferred_table_source_state: column_types must not be empty",
              std::invalid_argument);
}

cudf::data_type deferred_table_source_state::column_type(cudf::size_type source_column_index) const
{
  GQE_EXPECTS(source_column_index >= 0 && source_column_index < num_columns(),
              "deferred_table_source_state: source column index is out of range",
              std::out_of_range);
  return _column_types[static_cast<std::size_t>(source_column_index)];
}

std::vector<shared_state> deferred_table_source_state::materialize_columns(
  std::span<cudf::size_type const> source_column_indices,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  for (auto const index : source_column_indices) {
    GQE_EXPECTS(index >= 0 && index < num_columns(),
                "deferred_table_source_state: source column index is out of range",
                std::out_of_range);
  }

  // Collect the distinct requested columns in increasing order: mark them in a presence bitmap
  // and scan it, which dedupes and sorts in one pass over the column count.
  std::vector<bool> requested(static_cast<std::size_t>(num_columns()), false);
  for (auto const index : source_column_indices) {
    requested[static_cast<std::size_t>(index)] = true;
  }
  std::vector<cudf::size_type> distinct;
  for (cudf::size_type i = 0; i < num_columns(); ++i) {
    if (requested[static_cast<std::size_t>(i)]) { distinct.push_back(i); }
  }
  if (distinct.empty()) { return {}; }

  auto columns = materialize_columns_impl(distinct, stream_handle, mr);
  GQE_EXPECTS(columns.size() == distinct.size(),
              "deferred_table_source_state: materialization produced the wrong column count");

  std::vector<shared_state> by_column(static_cast<std::size_t>(num_columns()));
  for (std::size_t i = 0; i < distinct.size(); ++i) {
    GQE_EXPECTS(columns[i] != nullptr,
                "deferred_table_source_state: materialization produced a null column");
    GQE_EXPECTS(columns[i]->type() == column_type(distinct[i]) && columns[i]->size() == num_rows(),
                "deferred_table_source_state: materialized column does not match the source's "
                "declared type or row count");
    // Per the materialize_columns_impl contract, the decoded buffers were allocated from `mr` on
    // `stream`, so `stream` is their deallocation stream.
    by_column[static_cast<std::size_t>(distinct[i])] =
      make_shared_state(state_kind::cudf_column{std::move(columns[i])}, stream_handle);
  }

  std::vector<shared_state> result;
  result.reserve(source_column_indices.size());
  for (auto const index : source_column_indices) {
    result.push_back(by_column[static_cast<std::size_t>(index)]);
  }
  return result;
}

// -----------------------------------------------------------------------------
// deferred_column_state
// -----------------------------------------------------------------------------

deferred_column_state::deferred_column_state(std::shared_ptr<deferred_table_source_state> source,
                                             cudf::size_type source_column_index)
  : _source{std::move(source)}, _source_column_index{source_column_index}
{
  GQE_EXPECTS(
    _source != nullptr, "deferred_column_state: source must not be null", std::invalid_argument);
  GQE_EXPECTS(source_column_index >= 0 && source_column_index < _source->num_columns(),
              "deferred_column_state: source column index is out of range",
              std::out_of_range);
}

// -----------------------------------------------------------------------------
// shape helpers
// -----------------------------------------------------------------------------

state_container make_deferred_column_container(std::shared_ptr<deferred_table_source_state> source)
{
  GQE_EXPECTS(source != nullptr,
              "make_deferred_column_container: source must not be null",
              std::invalid_argument);
  state_container_builder builder;
  auto const num_columns = source->num_columns();
  for (cudf::size_type i = 0; i < num_columns; ++i) {
    builder.add_state(
      state_kind::deferred_column{std::make_shared<deferred_column_state const>(source, i)});
  }
  return builder.build();
}

state_container materialize_deferred_columns(state_container_view container,
                                             std::span<std::size_t const> slot_indexes,
                                             query_stream const& stream_handle,
                                             rmm::device_async_resource_ref mr)
{
  // A resolve task emits one source per chunk, so the selected deferred slots share one source and
  // materialize in one batched decode. Containers that combine several deferred sources are the
  // concern of column-level materialization over a logical table view, not of this helper.
  std::shared_ptr<deferred_table_source_state> source;
  std::vector<std::size_t> deferred_slots;
  std::vector<cudf::size_type> source_column_indexes;

  std::vector<bool> selected(container.size(), false);
  for (auto const slot : slot_indexes) {
    GQE_EXPECTS(slot < container.size(),
                "materialize_deferred_columns: slot index is out of range",
                std::out_of_range);
    if (selected[slot] || container[slot] == nullptr) { continue; }
    selected[slot] = true;

    auto const* deferred = std::get_if<state_kind::deferred_column>(container[slot].get());
    if (deferred == nullptr) { continue; }

    if (source == nullptr) { source = deferred->state()->source(); }
    GQE_EXPECTS(deferred->state()->source().get() == source.get(),
                "materialize_deferred_columns: deferred columns from different sources in one "
                "container are not supported");
    deferred_slots.push_back(slot);
    source_column_indexes.push_back(deferred->state()->source_column_index());
  }

  if (source == nullptr) { return make_mutable_state_copy(container); }

  state_container result(container.begin(), container.end());
  auto states = source->materialize_columns(source_column_indexes, stream_handle, mr);
  for (std::size_t i = 0; i < deferred_slots.size(); ++i) {
    result[deferred_slots[i]] = std::move(states[i]);
  }
  return result;
}

state_container materialize_deferred_columns(state_container_view container,
                                             query_stream const& stream_handle,
                                             rmm::device_async_resource_ref mr)
{
  std::vector<std::size_t> all_slots(container.size());
  std::iota(all_slots.begin(), all_slots.end(), std::size_t{0});
  return materialize_deferred_columns(container, all_slots, stream_handle, mr);
}

}  // namespace qep
}  // namespace gqe
