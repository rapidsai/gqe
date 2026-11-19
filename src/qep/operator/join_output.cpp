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

#include "join_output.hpp"

#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe::qep::detail {

join_projection_source resolve_join_projection(cudf::size_type projection_index,
                                               cudf::size_type build_logical_column_count,
                                               cudf::size_type probe_logical_column_count)
{
  GQE_EXPECTS(build_logical_column_count >= 0 && probe_logical_column_count >= 0,
              "resolve_join_projection: logical column counts must be non-negative");
  GQE_EXPECTS(projection_index >= 0,
              "resolve_join_projection: projection index must be non-negative");
  auto const combined_width = static_cast<std::size_t>(build_logical_column_count) +
                              static_cast<std::size_t>(probe_logical_column_count);
  GQE_EXPECTS(static_cast<std::size_t>(projection_index) < combined_width,
              "resolve_join_projection: projection index is out of range");
  bool const is_build = projection_index < build_logical_column_count;
  return join_projection_source{
    .is_build = is_build,
    .logical_column_index =
      is_build ? projection_index : projection_index - build_logical_column_count};
}

std::vector<cudf::size_type> join_materialized_payload_indices(
  std::span<cudf::size_type const> projection_indices,
  cudf::size_type materialized_logical_column_count,
  cudf::size_type streaming_logical_column_count)
{
  GQE_EXPECTS(materialized_logical_column_count >= 0 && streaming_logical_column_count >= 0,
              "join_materialized_payload_indices: logical column counts must be non-negative",
              std::invalid_argument);
  std::vector<bool> was_projected(static_cast<std::size_t>(materialized_logical_column_count));
  std::vector<cudf::size_type> result;
  result.reserve(projection_indices.size());
  for (auto const projection : projection_indices) {
    auto const source = resolve_join_projection(
      projection, materialized_logical_column_count, streaming_logical_column_count);
    if (!source.is_build) { continue; }
    auto const logical_index = static_cast<std::size_t>(source.logical_column_index);
    if (was_projected[logical_index]) { continue; }
    was_projected[logical_index] = true;
    result.push_back(source.logical_column_index);
  }
  return result;
}

void validate_join_projection_types(std::span<cudf::size_type const> projection_indices,
                                    std::span<cudf::data_type const> output_data_types,
                                    cudf::table_view build_columns,
                                    join_materialized_column_layout const& build_layout,
                                    logical_table_view probe)
{
  GQE_EXPECTS(projection_indices.size() == output_data_types.size(),
              "validate_join_projection_types: projection and output type counts differ",
              std::invalid_argument);
  for (std::size_t i = 0; i < projection_indices.size(); ++i) {
    auto const source = resolve_join_projection(
      projection_indices[i], build_layout.logical_column_count(), probe.num_columns());
    auto const actual_type =
      source.is_build
        ? build_columns.column(build_layout.materialized_index(source.logical_column_index)).type()
        : probe.column_type(source.logical_column_index);
    GQE_EXPECTS(actual_type == output_data_types[i],
                "validate_join_projection_types: projected output type differs from configured "
                "type",
                std::invalid_argument);
  }
}

state_container make_join_indirect_output(std::span<cudf::size_type const> projection_indices,
                                          state_container_view build_columns,
                                          join_materialized_column_layout const& build_layout,
                                          logical_table_view probe,
                                          position_list_handle const& build_positions,
                                          position_list_handle const& probe_positions,
                                          cudf::out_of_bounds_policy build_policy,
                                          query_stream const& stream_handle,
                                          rmm::device_async_resource_ref mr)
{
  GQE_EXPECTS(build_positions != nullptr && probe_positions != nullptr,
              "make_join_indirect_output: position lists must not be null");
  GQE_EXPECTS(build_positions->column().size() == probe_positions->column().size(),
              "make_join_indirect_output: position-list sizes differ");

  std::vector<cudf::size_type> build_source_indices(build_columns.size(), cudf::size_type{-1});
  std::vector<cudf::size_type> direct_probe_source_indices(
    static_cast<std::size_t>(probe.num_columns()), cudf::size_type{-1});
  state_container build_source_states;
  state_container direct_probe_source_states;

  // Select every projected direct build or probe column once. Indirect probe columns keep their
  // existing gather sources and are handled when projection order is restored below.
  for (auto const projection_index : projection_indices) {
    auto const source = resolve_join_projection(
      projection_index, build_layout.logical_column_count(), probe.num_columns());
    if (source.is_build) {
      auto const materialized_index =
        static_cast<std::size_t>(build_layout.materialized_index(source.logical_column_index));
      GQE_EXPECTS(materialized_index < build_columns.size(),
                  "make_join_indirect_output: materialized build index is out of range");
      if (build_source_indices[materialized_index] >= 0) { continue; }
      build_source_indices[materialized_index] =
        static_cast<cudf::size_type>(build_source_states.size());
      build_source_states.push_back(build_columns[materialized_index]);
      continue;
    }

    if (try_get_indirect_column(probe.column_state(source.logical_column_index)) != nullptr) {
      continue;
    }
    auto const logical_index = static_cast<std::size_t>(source.logical_column_index);
    if (direct_probe_source_indices[logical_index] >= 0) { continue; }
    direct_probe_source_indices[logical_index] =
      static_cast<cudf::size_type>(direct_probe_source_states.size());
    direct_probe_source_states.push_back(probe.column_state(source.logical_column_index));
  }

  indirect_gather_source build_gather_source;
  if (!build_source_states.empty()) {
    auto relation = indirect_table_source_state::make(std::move(build_source_states));
    build_gather_source =
      indirect_gather_source_state::make(std::move(relation), build_positions, build_policy);
  }

  indirect_gather_source direct_probe_gather_source;
  if (!direct_probe_source_states.empty()) {
    auto relation = indirect_table_source_state::make(std::move(direct_probe_source_states));
    direct_probe_gather_source = indirect_gather_source_state::make(
      std::move(relation), probe_positions, cudf::out_of_bounds_policy::DONT_CHECK);
  }

  // Restore projection order and share each position-list composition across columns that
  // reference the same original indirect gather source.
  std::unordered_map<indirect_gather_source_state const*, indirect_gather_source>
    composed_probe_sources;
  composed_probe_sources.reserve(projection_indices.size());
  state_container result;
  result.reserve(projection_indices.size());
  for (auto const projection_index : projection_indices) {
    auto const source = resolve_join_projection(
      projection_index, build_layout.logical_column_count(), probe.num_columns());
    if (source.is_build) {
      auto const materialized_index =
        static_cast<std::size_t>(build_layout.materialized_index(source.logical_column_index));
      result.push_back(
        indirect_column_state::make(build_gather_source, build_source_indices[materialized_index]));
      continue;
    }

    auto const* indirect = try_get_indirect_column(probe.column_state(source.logical_column_index));
    if (indirect == nullptr) {
      auto const logical_index = static_cast<std::size_t>(source.logical_column_index);
      result.push_back(indirect_column_state::make(direct_probe_gather_source,
                                                   direct_probe_source_indices[logical_index]));
      continue;
    }

    auto const& original_source = indirect->gather_source();
    auto [entry, was_inserted] =
      composed_probe_sources.try_emplace(original_source.get(), indirect_gather_source{});
    if (was_inserted) {
      auto composed_positions = compose_position_lists(
        original_source->position_list(), probe_positions, stream_handle, mr);
      entry->second = indirect_gather_source_state::make(original_source->relation_source(),
                                                         std::move(composed_positions),
                                                         original_source->bounds_policy());
    }
    result.push_back(indirect_column_state::make(entry->second, indirect->source_column_index()));
  }
  return result;
}

}  // namespace gqe::qep::detail
