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
 * @brief Per-chunk perfect-hash aggregate (the first apply in apply-concat-apply).
 *
 * This stateless transform accepts every column-bearing `logical_table_view` shape and emits an
 * owning, unmasked regular table shaped as `[keys..., partial aggregates...]`. It materializes the
 * unique union of grouping keys and aggregate values, then remaps configured logical indices to
 * materialized table indices. False or null QEP mask entries exclude their rows.
 *
 * Group keys and aggregate values must use the fixed-width types supported by the perfect hash. The
 * current perfect-hash implementation does not inspect input column null masks, so key columns and
 * aggregate value columns must contain no nulls anywhere, including rows excluded by the QEP valid
 * mask. `COUNT_ALL` is the exception: it ignores value nullness but still requires an in-range
 * value index.
 *
 * `COUNT_VALID` remains supported because existing aggregate plans use it directly and as the
 * partial count for `MEAN`. Since its value column must contain no nulls, it currently produces the
 * same result as `COUNT_ALL`.
 *
 * `next` is safe to call concurrently: the task stores immutable configuration only, and each call
 * operates solely on its input and call-local output. A downstream `perfect_hash_aggregate_task`,
 * configured to aggregate the partial columns, supplies the concat and second apply stages.
 *
 * @note The current perfect-hash APIs accept a CUDA stream but no memory resource. cuDF allocations
 * receive the supplied resource; the perfect hash retains its existing internal allocation
 * behavior.
 */
class perfect_hash_partial_aggregate_task : public stateless_transform_task {
 public:
  /**
   * @brief Construct a partial perfect-hash aggregate task.
   *
   * @param[in] key_indices Data-column indices of the grouping keys.
   * @param[in] value_indices `(aggregation kind, data-column index)` pairs.
   * @param[in] output_data_types Key types followed by aggregate result types.
   *
   * Configuration entries are positional and need not be unique. Repeated key indices or
   * aggregate specifications produce repeated output columns in the supplied order.
   */
  perfect_hash_partial_aggregate_task(
    std::vector<cudf::size_type> key_indices,
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
    std::vector<cudf::data_type> output_data_types);

  /**
   * @copydoc gqe::qep::stateless_transform_task::next
   */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _value_indices;
  std::vector<cudf::data_type> _output_data_types;
  std::vector<cudf::size_type> _materialized_indices;
  std::vector<cudf::size_type> _materialized_key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _materialized_value_indices;
};

/**
 * @brief Folding perfect-hash grouped aggregate (concat and second apply).
 *
 * `next` intentionally accepts only direct-column regular-table or masked-table chunks and retains
 * their shared ownership in task-private state; indirect input is rejected because physical
 * lowering feeds this task only direct partial output. Concurrent calls are mutex-protected.
 * `finalize` concatenates those chunks and emits one owning, unmasked regular table shaped as
 * `[keys..., aggregates...]`. All chunks must have identical masked/plain shape and column schema.
 * The grouping, null, and mask constraints match `perfect_hash_partial_aggregate_task`.
 *
 * This task is a pipeline breaker. For apply-concat-apply, configure its keys to reference the
 * partial output keys and its values to reference the partial aggregates. `SUM`, `PRODUCT`, and
 * `MIN` combine with themselves; `COUNT_VALID` and `COUNT_ALL` partials combine with `SUM`.
 */
class perfect_hash_aggregate_task : public fold_task {
 public:
  /**
   * @brief Construct a folding perfect-hash aggregate task.
   *
   * @param[in] key_indices Data-column indices of the grouping keys.
   * @param[in] value_indices `(aggregation kind, data-column index)` pairs.
   * @param[in] output_data_types Key types followed by aggregate result types.
   *
   * Configuration entries are positional and need not be unique. Repeated key indices or
   * aggregate specifications produce repeated output columns in the supplied order.
   */
  perfect_hash_aggregate_task(
    std::vector<cudf::size_type> key_indices,
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
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
  std::vector<cudf::size_type> _key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _value_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
