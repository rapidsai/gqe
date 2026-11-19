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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Sort a stream of regular tables into one materialized table.
 *
 * Each accumulator defines one independent sort operation.
 *
 * # Input shape
 *
 * Each call to `next` accepts one regular table with the shape
 * `[cudf_column | cudf_column_view, ...]`. Each column must contain the same number of rows. The
 * column count and data types must match `output_data_types`.
 *
 * # Ordering
 *
 * The columns in `key_column_indices` form a lexicographic key. The first index has the highest
 * priority. For each key, the value at the same position in `column_orders` specifies the sort
 * direction. The value at the same position in `null_precedences` specifies the position of null
 * values.
 *
 * # Output shape
 *
 * `finalize` returns one regular table with the shape `[cudf_column, ...]`. The result owns all its
 * columns. The result contains all rows from the input tables added to the accumulator. The rows
 * are in the specified lexicographic order. The column order and data types match
 * `output_data_types`.
 *
 * If the accumulator has no rows, `finalize` returns empty columns with the data types in
 * `output_data_types`.
 */
class merge_sort_task : public fold_task {
 public:
  /**
   * @brief Construct a merge-sort task.
   *
   * @param[in] key_column_indices Input column indices in lexicographic priority order. Repeated
   *   indices are allowed.
   * @param[in] column_orders Sort direction for each key column.
   * @param[in] null_precedences Null precedence for each key column.
   * @param[in] output_data_types Input and output data types in column order.
   *
   * @throws std::invalid_argument If `output_data_types` is empty.
   * @throws std::invalid_argument If `key_column_indices` is empty.
   * @throws std::invalid_argument If `key_column_indices` and `column_orders` have different sizes.
   * @throws std::invalid_argument If `key_column_indices` and `null_precedences` have different
   *   sizes.
   * @throws std::invalid_argument If a key index is outside the declared schema.
   */
  merge_sort_task(std::vector<cudf::size_type> key_column_indices,
                  std::vector<cudf::order> column_orders,
                  std::vector<cudf::null_order> null_precedences,
                  std::vector<cudf::data_type> output_data_types);

  /** @copydoc fold_task::initialize */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc fold_task::next */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc fold_task::finalize */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _key_column_indices;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_precedences;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
