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

#include <gqe/qep/task.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Reduction aggregate task (no group-by).
 *
 * Reduces all input rows to a single-row result. Values are referenced by column index.
 *
 * # Design
 *
 * The QEP-builder must insert a `project_task` upstream for non-trivial expressions. This is an
 * explicit choice to DRY expression evaluation. As of 2026-05-13, neither cuDF nor GQE provide a
 * reduce function that kernel-fuses expression evaluation.
 *
 * NOTE: A future kernel-fused expression evaluation SHALL NOT add expressions to this task.
 * Instead, create a new aggregate task and expose it in the physical plan.
 *
 * # Input Shape
 *
 * The task accepts every column-bearing `logical_table_view` shape and `[row_count{N}]` for pure
 * `COUNT_ALL`. It materializes only unique non-`COUNT_ALL` value columns. Masked `COUNT_ALL`
 * compacts an empty column request to count surviving rows without gathering indirect data.
 * Masked and unmasked chunks may be combined in the same reduction.
 *
 * # Output shape
 *
 * `[cudf_column (agg 0), cudf_column (agg 1), ...]` -- single-row.
 *
 * # Limitations
 *
 * Masked paths with value aggregates synchronize via `inject_valid_mask_as_nulls` (see
 * `qep/shapes/masked_table.hpp`). TODO: Replace with a non-blocking implementation.
 */
class reduce_aggregate_task : public fold_task {
 public:
  /**
   * @brief Construct a reduction aggregate task.
   *
   * @param[in] value_indices List of `(aggregation kind, column index)` pairs. The column
   *   index selects the value column from the input table; `COUNT_ALL` reads no column and
   *   its index must be `std::nullopt`.
   * @param[in] output_data_types Output column types in emission order. Used
   *   to construct nulls for non-COUNT reductions over an empty input.
   */
  reduce_aggregate_task(
    std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> value_indices,
    std::vector<cudf::data_type> output_data_types);

  /**
   * @copydoc gqe::qep::fold_task::initialize
   */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::next
   */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::finalize
   */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> _value_indices;
  std::vector<cudf::data_type> _output_data_types;
  std::vector<cudf::size_type> _materialized_indices;  ///< Unique logical value columns.

  std::vector<std::optional<cudf::size_type>>
    _materialized_positions;  ///< Column index in the materialized table for each aggregate, absent
                              ///< for COUNT_ALL.
};

}  // namespace qep
}  // namespace gqe
