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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief A non-owning view into a masked-table state container.
 *
 * The masked-table convention is a `state_container` with shape:
 *
 *     [valid_mask, (cudf_column | cudf_column_view)+]
 *
 * Use the `try_from` named constructor to unpack and validate the shape in
 * one step:
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
   *         match the masked-table convention.
   */
  [[nodiscard]] static std::optional<masked_table_view> try_from(state_container_view container);

  cudf::size_type row_count;  ///< Number of rows in the table.
  cudf::column_view mask;     ///< Boolean mask. `true` rows survive a subsequent gather.
  cudf::table_view columns;   ///< The data columns the mask applies to.
};

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
 * @param[in] stream CUDA stream.
 * @param[in] mr Memory resource.
 *
 * @return Mask-injected column views and their lifetime anchors.
 */
[[nodiscard]] null_mask_injection inject_valid_mask_as_nulls(
  cudf::column_view valid_mask,
  cudf::table_view columns,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace qep
}  // namespace gqe
