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

#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cuda/stream>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief A non-owning view into a masked-table state container.
 *
 * The masked-table convention is a `state_container` with shape:
 *
 *     [(valid_mask | valid_mask_view), (cudf_column | cudf_column_view)+]
 *
 * Use the `try_from` named constructor to unpack and validate the shape in one step:
 *
 *     if (auto masked = masked_table_view::try_from(inputs); masked) {
 *       // mask-aware path
 *     } else {
 *       // regular table path
 *     }
 */
struct masked_table_view {
  /**
   * @brief Named constructor: try to unpack a state container as a masked table.
   *
   * @param[in] container The state container to inspect.
   *
   * @return The unpacked view, or `std::nullopt` if the container does not
   *         match the masked-table convention. Unsupported data-state kinds are rejected
   *         before validating column storage or row counts.
   * @throws std::logic_error If a matching shape has missing column storage, no data
   *         columns, or inconsistent row counts.
   */
  [[nodiscard]] static std::optional<masked_table_view> try_from(state_container_view container);

  cudf::size_type row_count;  ///< Number of rows in the table.
  cudf::column_view mask;     ///< Boolean mask. A row survives a subsequent gather iff its entry
                              ///< is valid and `true`; a null entry is equivalent to `false`.
  cudf::table_view columns;   ///< The data columns the mask applies to.
};

/**
 * @brief Convert a BOOL8 valid mask to an owned, non-nullable BOOL8 column.
 *
 * The output contains `true` exactly where `valid_mask` is valid and `true`; false and null input
 * entries both become false. This is the representation required by consumers that read BOOL8
 * values directly without consulting the column null mask.
 *
 * @param[in] valid_mask BOOL8 column to normalize.
 * @param[in] stream_handle CUDA stream lifetime capability.
 * @param[in] mr Memory resource used for the output and temporary bitmask.
 * @return An owning, non-nullable BOOL8 column.
 */
[[nodiscard]] std::unique_ptr<cudf::column> make_non_nullable_valid_mask(
  cudf::column_view valid_mask,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Per-column null masks produced by AND-merging a BOOL8 valid_mask into each column's
 *        existing null mask.
 *
 * For each row where `valid_mask` is `false` or null, the corresponding bit in every result
 * column's null mask is clear. Compute kernels that propagate input nulls (notably the cuDF
 * AST evaluator) then treat masked-out rows as null and short-circuit per-row.
 *
 * # Lifetime
 *
 * `mask_bitmask` and `per_column_null_masks` are the storage backing the column views in
 * `columns`. They must outlive any compute that reads `columns`.
 */
struct null_mask_injection {
  rmm::device_buffer mask_bitmask;                        ///< BOOL8 → bitmask of `valid_mask`.
  std::vector<rmm::device_buffer> per_column_null_masks;  ///< Per-column merged null masks.
  std::vector<cudf::column_view> columns;                 ///< Mask-injected column views.

  /**
   * @brief Return a `cudf::table_view` over `columns`.
   */
  [[nodiscard]] cudf::table_view as_table_view() const;
};

/**
 * @brief AND-merge a BOOL8 valid_mask into every column's null mask.
 *
 * Used to make the cuDF AST evaluator (and other null-propagating compute) treat masked-out
 * rows as null. The caller must keep the returned struct alive across any downstream compute
 * that reads the result column views.
 *
 * @note Prefer adding native valid_mask support to new operators over routing through this
 *       function. Encoding the mask as nulls is a compatibility shim for compute that only
 *       understands cuDF null masks (notably the AST evaluator): it allocates a fresh null mask
 *       per column, conflates "masked out" with "null", and forces null-propagating semantics on
 *       operators that could otherwise consume the BOOL8 mask directly. New code that can read the
 *       valid_mask itself should do so and skip this conversion.
 *
 * @param[in] valid_mask BOOL8 column where `true` rows are valid.
 * @param[in] columns Columns to inject into. Must have the same row count as `valid_mask`.
 * @param[in] stream_handle CUDA stream lifetime capability.
 * @param[in] mr Memory resource.
 *
 * @return Mask-injected column views and the buffers that store their null masks.
 */
[[nodiscard]] null_mask_injection inject_valid_mask_as_nulls(
  cudf::column_view valid_mask,
  cudf::table_view columns,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief AND-merge a BOOL8 valid mask with the null masks of selected columns.
 *
 * The returned table has the same columns in the same order. Selected columns use the logical AND
 * of their existing null mask and `valid_mask`; unselected columns are unchanged. Duplicate
 * indices are ignored.
 *
 * @param[in] valid_mask BOOL8 column where `true` rows are valid.
 * @param[in] columns Input columns.
 * @param[in] column_indices Indices of columns whose null masks include `valid_mask`.
 * @param[in] stream_handle CUDA stream lifetime capability.
 * @param[in] mr Memory resource used for the generated null masks.
 * @return Column views and the buffers that store their generated null masks.
 */
[[nodiscard]] null_mask_injection inject_valid_mask_as_nulls(
  cudf::column_view valid_mask,
  cudf::table_view columns,
  std::span<cudf::size_type const> column_indices,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace qep
}  // namespace gqe
