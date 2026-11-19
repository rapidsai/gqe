/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Whether a cuDF type ID is supported as a perfect-hash join key.
 *
 * The supported IDs mirror the type switch used by the perfect hash's erased buffer accessor:
 * signed and unsigned integers, `FLOAT32`, `FLOAT64`, and `TIMESTAMP_DAYS`.
 *
 * @param[in] type Type ID to inspect.
 * @return `true` if `type` is supported as a perfect-hash-join key.
 */
[[nodiscard]] bool is_supported_perfect_hash_join_key_type(cudf::type_id type) noexcept;

/** @brief Selects which QEP input supplies the perfect hash's unique build keys. */
enum class perfect_hash_join_build_input {
  materialized,  ///< Build from the shared input emitted by the build fold.
  streaming      ///< Build from each partition supplied to the probe task's `next` call.
};

/**
 * @brief Accumulate and concatenate the broadcast input for a QEP perfect hash join.
 *
 * This task materializes the physical relation selected by the broadcast policy. That relation may
 * be either the perfect hash's unique build input or its probe input. The companion probe task
 * selects the perfect hash's orientation independently when it processes each streaming partition.
 * Equality keys and projected payloads are selectively materialized before concatenation.
 *
 * # Input shape
 *
 * Each `next` input is a validated column-bearing logical relation, with or without a valid mask.
 * `next` compacts masked input, so the materialized relation holds only selected rows and every
 * probe reads only those rows. A false or null mask entry rejects its row. A `cudf_column_view`
 * remains borrowed while retained by the accumulator. Its external storage owner must remain alive
 * until the work enqueued by `finalize` completes on the supplied stream.
 *
 * # State and output shapes
 *
 * `initialize` returns `[task_private]`. `finalize` returns `[task_private]`; that slot contains
 * the materialized state consumed by `perfect_hash_join_probe_task::initialize`. The state retains
 * the original logical width, the logical-to-materialized layout, and concatenated direct columns.
 * It has no valid mask.
 *
 * ## Thread, stream, and memory-resource safety
 *
 * Concurrent `next` calls each append one complete compacted chunk atomically under the
 * accumulator mutex. The fold lifecycle calls `finalize` after all `next` calls return. Each call
 * enqueues all work on its supplied stream and uses its supplied memory resource for every
 * allocation. After asynchronous concatenation is enqueued, input deallocation streams are ordered
 * behind it and the input states are released. QEP supplies cross-stream readiness before a probe
 * consumes the immutable result.
 *
 */
class perfect_hash_join_build_task : public fold_task {
 public:
  /**
   * @brief Construct a perfect-hash join build task.
   *
   * @param[in] build_logical_column_count Original materialized-side logical width.
   * @param[in] build_key_indices Materialized-side equality-key logical column indices.
   * @param[in] projected_payload_indices Side-local projected payload indices in first-projection
   * order. Repetitions reuse one materialized column.
   *
   * @throws std::invalid_argument If `build_key_indices` is empty or contains a negative
   * index.
   */
  perfect_hash_join_build_task(cudf::size_type build_logical_column_count,
                               std::vector<cudf::size_type> build_key_indices,
                               std::vector<cudf::size_type> projected_payload_indices);

  /**
   * @copydoc fold_task::initialize
   */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc fold_task::next
   */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc fold_task::finalize
   */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _build_key_indices;
  detail::join_materialized_column_layout _column_layout;
};

/**
 * @brief Join streaming partitions with a materialized broadcast side using perfect hashing.
 *
 * The materialized input comes from `perfect_hash_join_build_task`, while each `next` input is one
 * streaming partition. `build_input` selects which input is the unique side passed first to
 * the perfect hash. Projection indices always address the physical
 * `[materialized_columns..., streaming_columns...]` schema, regardless of which
 * input is selected as the perfect hash's build input. An empty projection produces count-only
 * state. Direct streaming payloads retain their source states, and indirect payloads compose their
 * existing position lists with the streaming match positions.
 *
 * # Input shapes
 *
 * `initialize` accepts the `[task_private]` materialized state emitted by
 * `perfect_hash_join_build_task::finalize` and returns `[task_private]`. Each `next` input is
 * any validated column-bearing logical-relation shape.
 *
 * # Output shape
 *
 * A non-empty projection normally returns `[indirect_column, ...]` in projection order. If the
 * fold received no build chunks, `next` returns schema-correct empty `cudf_column` states instead.
 * An empty projection returns `[row_count]`.
 *
 * @pre The keys selected by `build_input` are unique across that perfect-hash build input.
 * @pre The selected build and probe key columns have matching types. Supported key type IDs are
 * `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`, `FLOAT32`, `FLOAT64`,
 * and `TIMESTAMP_DAYS`.
 * @pre The selected key columns contain no nulls, including rows excluded by a `valid_mask`.
 *
 * ## Ownership, streams, and memory resources
 *
 * Probe keys are selectively materialized on the supplied stream and memory resource. Match lists
 * independently own their storage; temporary build/probe keys and masks are released after their
 * deallocation streams are ordered behind the join. A projected direct probe column is retained by
 * exact shared-state identity. A projected indirect probe column reuses its original relation
 * source and bounds policy, and allocates only the composed `P[M]` position list on the supplied
 * memory resource. Columns sharing one original gather source share one composed list per `next`
 * call. The task does not synchronize.
 */
class perfect_hash_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Construct a perfect-hash join probe task.
   *
   * @param[in] probe_key_indices Equality-key column indices in the streaming input.
   * @param[in] build_input Input whose equality keys are unique and form the perfect hash's build
   * side.
   * @param[in] projection_indices Output column indices in the
   *   `[materialized_columns..., streaming_columns...]` schema.
   * @param[in] output_data_types Expected projected output types, in emission order. Each type must
   *   be fixed-width or `STRING`; child-schema-dependent types are unsupported because
   *   `cudf::data_type` does not carry their child schema.
   *
   * @throws std::invalid_argument If `probe_key_indices` is empty, a key or projection index
   *   is negative, projection and output type counts differ, or an output type is neither
   *   fixed-width nor `STRING`.
   */
  perfect_hash_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                               perfect_hash_join_build_input build_input,
                               std::vector<cudf::size_type> projection_indices,
                               std::vector<cudf::data_type> output_data_types);

  /**
   * @copydoc stateful_transform_task::initialize
   */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc stateful_transform_task::next
   */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::size_type> _probe_key_indices;
  perfect_hash_join_build_input _build_input;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
