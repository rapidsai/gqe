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

#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/query_context.hpp>

#include <cuda/stream>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <span>

namespace gqe {
namespace qep {

/**
 * @brief Compose an existing position list with a second selecting position list.
 *
 * For original positions `P` and selecting positions `M`, the result contains `P[M]`.
 * Selecting positions must be valid indices into `P`; composition uses `DONT_CHECK` because join
 * match positions satisfy that invariant.
 *
 * The two input handles remain locally owned until the gather has been enqueued. Their deallocation
 * streams are then ordered after that gather, so the returned list owns independent storage and
 * does not retain either input.
 *
 * Device work is O(N) and returned storage is O(N) for N selecting positions.
 *
 * @param[in] positions Original position list `P`.
 * @param[in] selecting_positions Selecting position list `M`.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Memory resource for the returned position-list storage.
 *
 * @return Independently owning position list containing `P[M]`.
 *
 * @throws std::invalid_argument If either handle is null.
 */
[[nodiscard]] position_list_handle compose_position_lists(
  position_list_handle const& positions,
  position_list_handle const& selecting_positions,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compact a position list using a valid-row mask.
 *
 * Retains, in order, each position whose corresponding mask entry is valid and `true`.
 *
 * @param[in] positions Position list to compact.
 * @param[in] valid_mask BOOL8 mask with the same row count as `positions`.
 * @param[in] stream_handle CUDA stream lifetime capability for all device work.
 * @param[in] mr Memory resource for the returned position-list storage.
 *
 * @return A position list containing the entries selected by `valid_mask`.
 *
 * @throws std::invalid_argument If `positions` is null.
 * @throws std::logic_error If `valid_mask` is not BOOL8 or has a different row count.
 */
[[nodiscard]] position_list_handle compact_position_list(
  position_list_handle const& positions,
  cudf::column_view valid_mask,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize selected columns from one indirect gather source.
 *
 * `positions` supplies the gather map. It may be the gather source's original position list or a
 * compacted position list produced by `compact_position_list`. Columns are returned in
 * `source_column_indices` order; repeated indices produce repeated output columns.
 *
 * @param[in] source Relation source and bounds policy.
 * @param[in] source_column_indices Columns to gather, in output order.
 * @param[in] positions Position list to use as the gather map.
 * @param[in] stream_handle CUDA stream lifetime capability for all device work.
 * @param[in] mr Memory resource for returned columns.
 *
 * @return Materialized columns.
 *
 * @throws std::invalid_argument If a handle is null, no column is requested, or a source-column
 *         index is out of range.
 */
[[nodiscard]] cudf::table materialize_indirect_columns(
  indirect_gather_source const& source,
  std::span<cudf::size_type const> source_column_indices,
  position_list_handle const& positions,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace qep
}  // namespace gqe
