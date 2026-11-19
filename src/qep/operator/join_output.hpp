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

#pragma once

#include <gqe/qep/detail/join_materialized_column_layout.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <span>
#include <vector>

namespace gqe::qep::detail {

/** @brief Identifies one projected column from the execution-order build or probe schema. */
struct join_projection_source {
  bool is_build;  ///< Whether the projection addresses the physical build input.

  cudf::size_type logical_column_index;  ///< Side-local logical column index.
};

/** @brief Resolve one execution-order projection against the original physical input widths. */
[[nodiscard]] join_projection_source resolve_join_projection(
  cudf::size_type projection_index,
  cudf::size_type build_logical_column_count,
  cudf::size_type probe_logical_column_count);

/** @brief Derive unique materialized-side payload indices in first-projection order. */
[[nodiscard]] std::vector<cudf::size_type> join_materialized_payload_indices(
  std::span<cudf::size_type const> projection_indices,
  cudf::size_type materialized_logical_column_count,
  cudf::size_type streaming_logical_column_count);

/**
 * @brief Validate projected join indices and their configured output types.
 *
 * Projection indices address the execution-order `[build_columns..., probe_columns...]` schema.
 *
 * @param[in] projection_indices Execution-order projection indices.
 * @param[in] output_data_types Expected projected types in emission order.
 * @param[in] build_columns Selectively materialized physical build input.
 * @param[in] build_layout Original build width and logical-to-materialized mapping.
 * @param[in] probe Logical probe input. Projected types are read without materializing payloads.
 *
 * @throws std::logic_error If a projection index is out of range.
 * @throws std::invalid_argument If projection metadata counts or types differ.
 */
void validate_join_projection_types(std::span<cudf::size_type const> projection_indices,
                                    std::span<cudf::data_type const> output_data_types,
                                    cudf::table_view build_columns,
                                    join_materialized_column_layout const& build_layout,
                                    logical_table_view probe);

/**
 * @brief Publish join projections while preserving original probe payload sources.
 *
 * Probe positions address the original logical row geometry, before mask compaction. Direct
 * columns share a new gather source. Indirect columns retain their original sources and bounds
 * policies with position lists composed once per distinct gather source. For M matches and S
 * distinct indirect probe sources, composition takes O(M * S) device work and storage.
 *
 * @param[in] projection_indices Projection into the logical build-plus-probe schema.
 * @param[in] build_columns Materialized build columns.
 * @param[in] build_layout Logical-to-materialized build mapping.
 * @param[in] probe Original logical probe relation.
 * @param[in] build_positions Match positions in the materialized build relation.
 * @param[in] probe_positions Match positions in the original probe relation.
 * @param[in] build_policy Bounds policy, including nullification for unmatched left-join rows.
 * @param[in] stream CUDA stream for all device work.
 * @param[in] mr Memory resource for all device allocations.
 * @return Indirect columns retaining their sources in projection order.
 * @throws std::logic_error If position lists are null or have different sizes.
 */
[[nodiscard]] state_container make_join_indirect_output(
  std::span<cudf::size_type const> projection_indices,
  state_container_view build_columns,
  join_materialized_column_layout const& build_layout,
  logical_table_view probe,
  position_list_handle const& build_positions,
  position_list_handle const& probe_positions,
  cudf::out_of_bounds_policy build_policy,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr);

}  // namespace gqe::qep::detail
