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
#include <gqe/query_context.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe::qep {

/**
 * @brief Build reusable equality-key state for left-semi and left-anti joins.
 *
 * This task consumes the physical build input. Lowering assigns the logical right input to it
 * for left-semi and left-anti joins. Each build chunk materializes and compacts only equality
 * keys. Finalization concatenates the retained keys and constructs one `cudf::filtered_join`
 * that every probe chunk reuses. The same finalized state supports either probe mode.
 *
 * # Input shape
 *
 * Each `next` input must be a logical relation accepted by `logical_table_view`. Borrowed
 * selected-key storage must survive its asynchronous use. Nested equality keys (`LIST` and
 * `STRUCT`) are rejected before materialization.
 *
 * # State and output shapes
 *
 * The accumulator and finalized output each contain one `[task_private]` state. Finalized state
 * owns the build keys and filtered join consumed by `equality_semi_anti_join_probe_task`.
 *
 * - No build chunks produce state without a filtered join or key-type metadata.
 * - A zero-row chunk preserves the key schema and constructs an empty filtered join.
 *
 * ## Thread and stream safety
 *
 * Concurrent `next` calls append complete chunks atomically under the accumulator mutex.
 * Finalized state is immutable and supports concurrent probes on separate streams. The caller
 * must order build work before finalization and finalization before probing across streams.
 * Tasks do not synchronize streams themselves.
 *
 * ## Streams and memory resources
 *
 * Each `next` materialization uses that call's stream and memory resource. Concatenation uses the
 * stream and memory resource passed to `finalize`. Retained input deallocation is ordered after
 * concatenation before the input states are released. The filtered join's hash table keeps the
 * `finalize` resource, which must outlive the build state. cuDF allocates key metadata and build
 * scratch from the current device resource, not `mr`.
 */
class equality_semi_anti_join_build_task : public fold_task {
 public:
  /**
   * @brief Construct an equality semi/anti build task.
   *
   * @param[in] build_key_indices Build-side logical equality-key column indices.
   * @param[in] null_equality Whether null keys compare equal.
   *
   * @throws std::invalid_argument If `build_key_indices` is empty or contains a negative index.
   */
  equality_semi_anti_join_build_task(std::vector<cudf::size_type> build_key_indices,
                                     cudf::null_equality null_equality);

  /** @copydoc fold_task::initialize */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc fold_task::next
   *
   * @throws std::logic_error If input shape, key indices, or accumulator state are invalid.
   * @throws std::invalid_argument If a selected build key has a nested type.
   */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc fold_task::finalize
   *
   * @throws std::logic_error If accumulator or build-schema invariants are violated.
   */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _build_key_indices;
  cudf::null_equality _null_equality;
};

/**
 * @brief Probe a reusable equality filter and publish selected probe rows.
 *
 * The task supports only `left_semi` and `left_anti`. It consumes the physical probe input, which
 * lowering assigns to the logical left input for these join types. Projection indices address the
 * probe-side logical schema. Only equality keys are materialized and compacted before probing.
 * Selected positions are mapped back to the original logical rows before projecting deferred
 * payload columns.
 *
 * # Input and output shapes
 *
 * `initialize` accepts the `[task_private]` state emitted by build finalization. Each `next` input
 * must be a logical relation accepted by `logical_table_view`. Nested equality keys (`LIST` and
 * `STRUCT`) are rejected before materialization, even when the build has no chunks.
 *
 * - A non-empty projection returns indirect columns in projection order. Direct payload columns
 *   share a gather source. Existing indirect payloads retain their sources and bounds policies,
 *   with position lists composed once per distinct gather source.
 * - An empty projection returns `[row_count]`.
 * - An empty build selects no rows in semi mode and every valid probe row in anti mode.
 *
 * ## Streams and memory resources
 *
 * Concurrent probes may use separate host threads and streams after build finalization is ready.
 * The caller must order input readiness on each supplied stream. All probe materialization,
 * match positions, and position-list composition use that call's stream and memory resource, except
 * cuDF probe scratch, which uses the current device resource. Shared input allocations retain their
 * original resource. Deallocation is ordered after asynchronous reads. The task does not
 * synchronize its stream.
 *
 * # Ownership
 *
 * Outputs retain projected source states. Borrowed storage must survive downstream use, including
 * for masked inputs.
 */
class equality_semi_anti_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Construct an equality semi/anti probe task.
   *
   * @param[in] probe_key_indices Probe-side logical equality-key column indices.
   * @param[in] join_type Join semantics. Only `left_semi` and `left_anti` are supported.
   * @param[in] projection_indices Probe-side logical output column indices.
   * @param[in] output_data_types Expected projected output types in emission order.
   *
   * @throws std::invalid_argument If key indices, join type, or projection metadata are invalid.
   * Supported output types are fixed-width and `STRING`.
   */
  equality_semi_anti_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                                     join_type_type join_type,
                                     std::vector<cudf::size_type> projection_indices,
                                     std::vector<cudf::data_type> output_data_types);

  /**
   * @copydoc stateful_transform_task::initialize
   *
   * @throws std::logic_error If the input is not finalized equality semi/anti build state.
   */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc stateful_transform_task::next
   *
   * @throws std::logic_error If input state or column indices are invalid, or keys are
   * incompatible.
   * @throws std::invalid_argument If projected types differ from their metadata or keys are nested.
   */
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
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace gqe::qep
