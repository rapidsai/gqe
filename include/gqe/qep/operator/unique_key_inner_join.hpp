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
#include <gqe/qep/task.hpp>
#include <gqe/query_context.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Materialize unique build keys and projected payloads, optionally caching the hash map.
 *
 * Physical-to-QEP lowering selects this task for eligible inner joins when `join_use_unique_keys`
 * is enabled. Build keys must satisfy both requirements:
 *
 * - Types are integer, floating-point, Boolean, or fixed-point decimal.
 * - Keys are unique across all retained chunks under the configured null equality.
 *
 * The constructor validates type support. Uniqueness is a planner guarantee and is not checked
 * by scanning data.
 *
 * # Input and state shapes
 *
 * Each `next` input is a column-bearing `logical_table_view`. False and null mask entries are
 * discarded while equality keys and projected payloads are materialized.
 *
 * `initialize` returns a `[task_private]` accumulator. `finalize` returns `[task_private]` owning
 * the retained build columns and optional cached join for `unique_key_inner_join_probe_task`.
 *
 * # Ownership, streams, and memory resources
 *
 * Borrowed input storage must outlive all work reading it. Concurrent `next` calls are supported.
 * The executor must complete the fold lifecycle before probing and order build work before uses
 * on other streams. Direct callers have the same ordering responsibilities.
 *
 * Compaction uses each `next` call's stream and resource. Concatenation and the cached hash map
 * use those supplied to `finalize`. cuDF allocates device-side column-view metadata through its
 * own resource. These metadata allocations do not use `mr`.
 */
class unique_key_inner_join_build_task : public fold_task {
 public:
  /**
   * @brief Construct a unique-key inner-join build task.
   *
   * @param[in] build_logical_column_count Original build-side logical width.
   * @param[in] build_key_indices Build-side equality-key logical column indices.
   * @param[in] projected_payload_indices Side-local projected payload indices in first-projection
   * order. Repetitions reuse one materialized column.
   * @param[in] null_equality Whether null keys compare equal.
   * @param[in] is_hash_map_cache_enabled Whether to build one reusable hash map during finalize.
   * @param[in] build_key_types Equality-key types in key-index order. Each type must be supported
   * by `gqe::unique_key_join_supported`. Actual input types are checked by `next`.
   *
   * @throws std::invalid_argument If keys or layout indices are invalid, the type count differs
   * from the key count, or a key type is unsupported.
   */
  unique_key_inner_join_build_task(cudf::size_type build_logical_column_count,
                                   std::vector<cudf::size_type> build_key_indices,
                                   std::vector<cudf::size_type> projected_payload_indices,
                                   cudf::null_equality null_equality,
                                   bool is_hash_map_cache_enabled,
                                   std::vector<cudf::data_type> build_key_types);

  /** @copydoc fold_task::initialize */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc fold_task::next
   *
   * @throws std::logic_error If a unique-key input's key types differ from the constructor's
   * metadata. This invariant is checked before materializing or retaining the input.
   */
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
  std::vector<cudf::size_type> _build_key_indices;
  detail::join_materialized_column_layout _column_layout;
  cudf::null_equality _null_equality;
  bool _is_hash_map_cache_enabled;
  std::vector<cudf::data_type> _build_key_types;
};

/**
 * @brief Probe a unique-key inner join and publish indirect projected columns.
 *
 * Either logical input may be the build input. Projection indices address the execution-order
 * `[build_columns..., probe_columns...]` schema using the original logical column widths.
 *
 * # Input and output shapes
 *
 * `initialize` shares the `[task_private]` state from `unique_key_inner_join_build_task::finalize`
 * and rejects states from other task types. Each `next` input is a column-bearing
 * `logical_table_view` with key types matching the build keys. False and null mask entries are
 * discarded before probing.
 *
 * - A non-empty projection emits `[indirect_column, ...]` in projection order without gathering
 *   probe payloads. If the build fold received no chunks, it emits schema-correct empty columns.
 * - An empty projection emits `[row_count]`.
 *
 * # Ownership, streams, and memory resources
 *
 * Output retains its projected sources, which may also own unprojected columns. Concurrent probes
 * may share finalized build state on separate streams, subject to the build task's ordering
 * contract. QEP owns stream ordering between output and downstream consumers.
 *
 * Each `next` call uses its supplied stream and resource for probe allocations and, when caching
 * is disabled, its local hash map. The build task documents the device-view metadata exception.
 * The unique-key primitive reads the match count on the host before returning, synchronizing the
 * supplied stream through the probe. The task adds no further synchronization.
 */
class unique_key_inner_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Construct a unique-key inner-join probe task.
   *
   * @param[in] probe_key_indices Probe-side equality-key column indices.
   * @param[in] projection_indices Output column indices in the execution-order schema.
   * @param[in] output_data_types Expected projected output types in emission order.
   *
   * @throws std::invalid_argument If keys or projections are invalid, output metadata is
   * inconsistent, or an output type is nested.
   */
  unique_key_inner_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                                   std::vector<cudf::size_type> projection_indices,
                                   std::vector<cudf::data_type> output_data_types);

  /** @copydoc stateful_transform_task::initialize */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc stateful_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _probe_key_indices;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
