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

#include <gqe/qep/column_materialization.hpp>

#include <gqe/qep/indirect_gather.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cstddef>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/** @brief Supported physical representation of one logical column. */
enum class physical_column_representation { direct, indirect };

/** @brief One requested logical column and its physical representation. */
struct column_request {
  cudf::size_type logical_index;
  physical_column_representation representation;
};

/**
 * @brief Validate and classify columns in request order.
 *
 * Validation completes before any GPU work. Repeated indices remain separate requests.
 */
[[nodiscard]] std::vector<column_request> make_request_plan(
  logical_table_view input, std::span<cudf::size_type const> logical_column_indices)
{
  std::vector<column_request> requests;
  requests.reserve(logical_column_indices.size());
  for (auto const logical_index : logical_column_indices) {
    GQE_EXPECTS(logical_index >= 0 && logical_index < input.num_columns(),
                "materialize_columns: logical column index is out of range",
                std::out_of_range);
    auto const& state = input.column_state(logical_index);
    physical_column_representation representation;
    if (to_column_view(state).has_value()) {
      representation = physical_column_representation::direct;
    } else {
      GQE_EXPECTS(try_get_indirect_column(state) != nullptr,
                  "materialize_columns: logical column has an unsupported representation");
      representation = physical_column_representation::indirect;
    }
    requests.push_back(column_request{logical_index, representation});
  }
  return requests;
}

/** @brief Requested indirect columns that share one gather-source identity. */
struct gather_group {
  indirect_gather_source source;
  std::vector<std::size_t> request_indices;
  std::vector<cudf::size_type> source_column_indices;
};

/** @brief Partition indirect requests by gather-source pointer identity. */
[[nodiscard]] std::vector<gather_group> make_gather_groups(logical_table_view input,
                                                           std::span<column_request const> requests)
{
  std::vector<gather_group> groups;
  std::unordered_map<indirect_gather_source_state const*, std::size_t> group_by_source;
  group_by_source.reserve(requests.size());

  for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
    auto const& request = requests[request_index];
    switch (request.representation) {
      case physical_column_representation::direct: break;
      case physical_column_representation::indirect: {
        auto const* indirect = try_get_indirect_column(input.column_state(request.logical_index));
        GQE_EXPECTS(indirect != nullptr,
                    "materialize_columns: indirect request has no indirect descriptor");
        auto const& source         = indirect->gather_source();
        auto [entry, was_inserted] = group_by_source.emplace(source.get(), groups.size());
        if (was_inserted) { groups.push_back(gather_group{source, {}, {}}); }
        auto& group = groups[entry->second];
        group.request_indices.push_back(request_index);
        group.source_column_indices.push_back(indirect->source_column_index());
        break;
      }
    }
  }
  return groups;
}

/** @brief Build `[0, input.num_columns())` for full materialization. */
[[nodiscard]] std::vector<cudf::size_type> all_logical_indices(logical_table_view input)
{
  std::vector<cudf::size_type> indices(static_cast<std::size_t>(input.num_columns()));
  std::iota(indices.begin(), indices.end(), cudf::size_type{0});
  return indices;
}

}  // namespace

column_materialization::column_materialization(std::vector<cudf::size_type> logical_column_indices,
                                               state_container states,
                                               cudf::size_type row_count)
  : _logical_column_indices{std::move(logical_column_indices)},
    _states{std::move(states)},
    _row_count{row_count}
{
  GQE_EXPECTS(_logical_column_indices.size() == _states.size(),
              "column_materialization: logical-index and state counts do not match");
  for (auto const& state : _states) {
    auto const column = to_column_view(state);
    GQE_EXPECTS(column.has_value(),
                "column_materialization: result state is not a direct cuDF column");
    GQE_EXPECTS(column->size() == _row_count,
                "column_materialization: result column row count does not match result geometry");
  }
}

cudf::table_view column_materialization::table_view() const
{
  if (_states.empty()) { return cudf::table_view{}; }

  auto const table = to_table_view(_states);
  GQE_EXPECTS(table.has_value(),
              "column_materialization::table_view: result state is not a direct cuDF column");
  return *table;
}

state_container_view column_materialization::states() const { return _states; }

state_container column_materialization::release_column_states() &&
{
  GQE_EXPECTS(!_states.empty(),
              "column_materialization::release_column_states: a zero-column result cannot be "
              "published");
  return std::move(_states);
}

column_materialization materialize_columns(logical_table_view input,
                                           std::span<cudf::size_type const> logical_column_indices,
                                           query_stream const& stream_handle,
                                           rmm::device_async_resource_ref mr)
{
  auto const requests = make_request_plan(input, logical_column_indices);
  state_container states(requests.size());
  state_container asynchronous_dependencies;

  for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
    auto const& request = requests[request_index];
    switch (request.representation) {
      case physical_column_representation::direct:
        states[request_index] = input.column_state(request.logical_index);
        break;
      case physical_column_representation::indirect: break;
    }
  }

  for (auto const& group : make_gather_groups(input, requests)) {
    for (auto const request_index : group.request_indices) {
      asynchronous_dependencies.push_back(
        input.column_state(requests[request_index].logical_index));
    }
    auto gathered =
      materialize_indirect_columns(
        group.source, group.source_column_indices, group.source->position_list(), stream_handle, mr)
        .release();
    GQE_EXPECTS(gathered.size() == group.request_indices.size(),
                "materialize_columns: gathered group has unexpected column count");
    for (std::size_t i = 0; i < gathered.size(); ++i) {
      auto const request_index = group.request_indices[i];
      states[request_index] =
        make_shared_state(state_kind::cudf_column{std::move(gathered[i])}, stream_handle);
    }
  }
  order_deallocation_after_stream_use(asynchronous_dependencies, stream_handle);

  return column_materialization(
    std::vector(logical_column_indices.begin(), logical_column_indices.end()),
    std::move(states),
    input.row_count());
}

column_materialization materialize_all_columns(logical_table_view input,
                                               query_stream const& stream_handle,
                                               rmm::device_async_resource_ref mr)
{
  auto const indices = all_logical_indices(input);
  return materialize_columns(input, indices, stream_handle, mr);
}

column_materialization materialize_compacted_columns(
  logical_table_view input,
  std::span<cudf::size_type const> logical_column_indices,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  return materialize_compacted_columns(input, logical_column_indices, false, stream_handle, mr);
}

column_materialization materialize_compacted_columns(
  logical_table_view input,
  std::span<cudf::size_type const> logical_column_indices,
  bool retain_row_positions,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  auto const mask   = input.valid_mask();
  if (!mask.has_value()) {
    return materialize_columns(input, logical_column_indices, stream_handle, mr);
  }

  auto const requests = make_request_plan(input, logical_column_indices);

  auto const* mask_state = input.valid_mask_state();
  GQE_EXPECTS(mask_state != nullptr,
              "materialize_compacted_columns: validated mask has no source state");

  state_container asynchronous_dependencies{*mask_state};
  asynchronous_dependencies.reserve(requests.size() + 1);
  for (auto const& request : requests) {
    asynchronous_dependencies.push_back(input.column_state(request.logical_index));
  }

  auto const groups = make_gather_groups(input, requests);
  std::unordered_map<position_list_state const*, position_list_handle> compacted_position_lists;

  state_container states(requests.size());
  std::vector<std::size_t> direct_request_indices;
  std::vector<cudf::column_view> compaction_columns;
  direct_request_indices.reserve(requests.size());
  compaction_columns.reserve(requests.size());
  for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
    auto const& request = requests[request_index];
    switch (request.representation) {
      case physical_column_representation::direct: {
        direct_request_indices.push_back(request_index);
        auto const direct = to_column_view(input.column_state(request.logical_index));
        GQE_EXPECTS(direct.has_value(),
                    "materialize_compacted_columns: validated direct column is no longer "
                    "concrete");
        compaction_columns.push_back(*direct);
        break;
      }
      case physical_column_representation::indirect: break;
    }
  }

  // Pack direct columns first, distinct indirect position lists next, and optional logical row
  // IDs last. The unpacking below uses this order to recover each output from one mask operation.
  // Different gather-source groups can share a position list, so deduplicate position lists by
  // pointer identity independently of the groups to avoid compacting shared position lists twice.
  std::vector<position_list_state const*> position_keys;
  position_keys.reserve(groups.size());
  compacted_position_lists.reserve(groups.size());
  for (auto const& group : groups) {
    auto const& original = group.source->position_list();
    if (compacted_position_lists.emplace(original.get(), nullptr).second) {
      position_keys.push_back(original.get());
      compaction_columns.push_back(original->column().view());
    }
  }
  std::unique_ptr<cudf::column> row_ids;
  position_list_handle row_positions;
  if (retain_row_positions) {
    cudf::numeric_scalar<cudf::size_type> zero{0, true, stream, mr};
    row_ids = cudf::sequence(input.row_count(), zero, stream, mr);
    compaction_columns.push_back(row_ids->view());
  }

  std::optional<cudf::size_type> compacted_row_count;
  auto const record_row_count = [&](cudf::size_type row_count) {
    GQE_EXPECTS(!compacted_row_count.has_value() || *compacted_row_count == row_count,
                "materialize_compacted_columns: compacted column row counts do not match");
    compacted_row_count = row_count;
  };
  if (!compaction_columns.empty()) {
    auto compacted =
      cudf::apply_retention_mask(cudf::table_view{compaction_columns}, *mask, stream, mr)
        ->release();
    GQE_EXPECTS(compacted.size() == compaction_columns.size(),
                "materialize_compacted_columns: compacted table has unexpected column "
                "count");
    record_row_count(compacted.front()->size());
    for (std::size_t i = 0; i < direct_request_indices.size(); ++i) {
      auto const request_index = direct_request_indices[i];
      states[request_index] =
        make_shared_state(state_kind::cudf_column{std::move(compacted[i])}, stream_handle);
    }
    for (std::size_t i = 0; i < position_keys.size(); ++i) {
      compacted_position_lists.at(position_keys[i]) =
        make_position_list(std::move(compacted[direct_request_indices.size() + i]), stream_handle);
    }
    if (retain_row_positions) {
      row_positions = make_position_list(std::move(compacted.back()), stream_handle);
    }
  }

  for (auto const& group : groups) {
    auto const compacted = compacted_position_lists.find(group.source->position_list().get());
    GQE_EXPECTS(compacted != compacted_position_lists.end() && compacted->second != nullptr,
                "materialize_compacted_columns: missing compacted position list");
    auto const& positions = compacted->second;
    auto gathered         = materialize_indirect_columns(
                      group.source, group.source_column_indices, positions, stream_handle, mr)
                      .release();
    GQE_EXPECTS(gathered.size() == group.request_indices.size(),
                "materialize_compacted_columns: gathered group has unexpected column count");
    record_row_count(positions->column().size());
    for (std::size_t i = 0; i < gathered.size(); ++i) {
      auto const request_index = group.request_indices[i];
      states[request_index] =
        make_shared_state(state_kind::cudf_column{std::move(gathered[i])}, stream_handle);
    }
  }

  if (!compacted_row_count.has_value()) {
    auto total = cudf::reduce(*mask,
                              *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                              cudf::data_type{cudf::type_id::INT32},
                              stream,
                              mr);
    // SUM returns a null scalar for empty or all-null masks; the surviving row count is zero.
    record_row_count(
      total->is_valid(stream)
        ? static_cast<cudf::numeric_scalar<cudf::size_type> const&>(*total).value(stream)
        : 0);
  }

  order_deallocation_after_stream_use(asynchronous_dependencies, stream_handle);

  GQE_EXPECTS(compacted_row_count.has_value(),
              "materialize_compacted_columns: failed to determine compacted row count");
  auto result = column_materialization(
    std::vector(logical_column_indices.begin(), logical_column_indices.end()),
    std::move(states),
    *compacted_row_count);
  result._row_positions = std::move(row_positions);
  return result;
}

column_materialization materialize_all_compacted_columns(logical_table_view input,
                                                         query_stream const& stream_handle,
                                                         rmm::device_async_resource_ref mr)
{
  auto const indices = all_logical_indices(input);
  return materialize_compacted_columns(input, indices, stream_handle, mr);
}

}  // namespace qep
}  // namespace gqe
