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
 * @brief Per-chunk general hash aggregate (the first apply in apply-concat-apply).
 *
 * This stateless transform accepts every column-bearing `logical_table_view` shape and always emits
 * an owning, unmasked regular table shaped as `[keys..., partial aggregates...]`. It materializes
 * the unique union of grouping keys and aggregate values, then remaps configured logical indices to
 * materialized table indices. False and null entries in a QEP valid mask both exclude the
 * corresponding row. Input column nulls retain cuDF group-by semantics.
 *
 * `next` is safe to call concurrently from different host threads and CUDA streams. The task stores
 * immutable configuration only. The group-by aggregation may synchronize the supplied stream to
 * size internal storage and select its execution path. All GPU work uses that stream, and the
 * supplied memory resource is propagated to allocation APIs.
 */
class hash_partial_aggregate_task : public stateless_transform_task {
 public:
  /**
   * @brief Construct a partial general hash aggregate task.
   *
   * @param[in] key_indices Data-column indices of the grouping keys.
   * @param[in] value_indices `(aggregation kind, data-column index)` pairs.
   *
   * Configuration entries are positional and need not be unique. Repeated key indices or
   * aggregate specifications produce repeated output columns in the supplied order.
   *
   * Supported aggregation kinds are `SUM`, `MIN`, `MAX`, `COUNT_VALID`, and `COUNT_ALL`.
   *
   * @throws std::invalid_argument If there are no grouping keys or an aggregation kind is not
   * supported.
   */
  hash_partial_aggregate_task(
    std::vector<cudf::size_type> key_indices,
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices);

  /** @copydoc gqe::qep::stateless_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc gqe::qep::task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _value_indices;
  std::vector<cudf::size_type> _materialized_indices;
  std::vector<cudf::size_type> _materialized_key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _materialized_value_indices;
};

/**
 * @brief Folding general hash aggregate (concat and second apply).
 *
 * `next` accepts only regular-table partials and retains their shared states in task-private
 * accumulator state. When a partial contains borrowed column states, the backing column owners
 * must remain alive until `finalize` returns. Concurrent calls are mutex-protected, and the task
 * object itself remains stateless. `finalize` concatenates the retained partial tables and emits
 * one owning regular table shaped as `[keys..., final aggregates...]`. The executor owns
 * cross-stream ordering. `finalize` may synchronize its supplied stream while the group-by
 * aggregation sizes internal storage and selects its execution path.
 *
 * Configure keys and values to address the partial output. Counts must be configured with `SUM` in
 * this combine stage. `final_output_types` supplies the schema when no partial chunks arrive and is
 * also checked against non-empty inputs.
 */
class hash_aggregate_task : public fold_task {
 public:
  /**
   * @brief Construct a folding general hash aggregate task.
   *
   * @param[in] key_indices Partial-table indices of the grouping keys.
   * @param[in] value_indices `(combine aggregation kind, partial-table index)` pairs.
   * @param[in] final_output_types Key types followed by final aggregate result types.
   *
   * Supported combine aggregation kinds are `SUM`, `MIN`, and `MAX`. The number of final output
   * types must equal the number of keys plus the number of aggregates.
   *
   * @throws std::invalid_argument If there are no grouping keys, an aggregation kind is not
   * supported, or the final output type count does not match the configured output columns.
   */
  hash_aggregate_task(
    std::vector<cudf::size_type> key_indices,
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
    std::vector<cudf::data_type> final_output_types);

  /** @copydoc gqe::qep::fold_task::initialize */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc gqe::qep::fold_task::next */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc gqe::qep::fold_task::finalize */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc gqe::qep::task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _key_indices;
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> _value_indices;
  std::vector<cudf::data_type> _final_output_types;
};

}  // namespace qep
}  // namespace gqe
