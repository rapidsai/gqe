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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Materialize the build input and optionally cache its hash map.
 *
 * The build input is the physical right input to `cudf::hash_join`. Each input chunk is a logical
 * relation. False and null mask entries are discarded while only equality keys and projected
 * payload columns are compacted. The finalized state retains the original logical width and a
 * materialized relation ordered as distinct requested key columns followed by first-projection
 * payloads.
 *
 * # Input shape
 *
 * Each `next` input is a validated column-bearing `logical_table_view` shape. Input chunks may mix
 * direct, indirect, masked, and masked-indirect representations. Borrowed column storage must
 * remain alive until all work reading it has completed on the supplied stream.
 *
 * # State and output shapes
 *
 * `initialize` returns `[task_private]`. Concurrent `next` calls append complete chunks to that
 * accumulator. `finalize` returns `[task_private]`, which owns the concatenated payload, selected
 * build-key views, and optional cached `cudf::hash_join` consumed by
 * `hash_join_probe_task::initialize`.
 *
 * ## Thread and stream safety
 *
 * `next` may be called concurrently from host threads. Its accumulator mutex protects only the
 * retained chunk sequence and one complete append is the atomic state transition. The QEP fold
 * lifecycle calls `finalize` only after all `next` calls have returned, so finalization reads the
 * accumulator without locking.
 *
 * A cached hash join supports concurrent probe calls on separate streams after finalization is
 * ready. QEP supplies the cross-stream producer-to-consumer ordering. Direct callers must likewise
 * order the finalization stream before any different probe stream. The cached hash join is declared
 * after its viewed payload so destruction releases the hash join first.
 *
 * ## Memory resources
 *
 * Each `next` compaction uses that call's stream and memory resource. Final concatenation and the
 * cached hash map use the stream and memory resource supplied to `finalize`. Once concatenation has
 * been enqueued, input deallocation streams are ordered after it and the input states are released.
 * The cached hash join retains its memory resource, so that resource must outlive the finalized
 * build state, as QEP's executor resource does. cuDF allocates key metadata and build scratch from
 * the current device resource, not `mr`.
 *
 * For `R` retained rows, `K` keys, and `D` values across all retained payload columns, finalization
 * takes `O(D + R * K)` expected work and retains `O(D + R)` storage, including cuDF's hash map.
 */
class hash_join_build_task : public fold_task {
 public:
  /**
   * @brief Construct a hash-join build task.
   *
   * @param[in] build_logical_column_count Original build-side logical width.
   * @param[in] build_key_indices Build-side equality-key logical column indices.
   * @param[in] projected_payload_indices Side-local projected payload indices in first-projection
   * order. Repetitions reuse one materialized column.
   * @param[in] null_equality Whether null keys compare equal.
   * @param[in] is_hash_map_cache_enabled Whether to build one reusable hash map during finalize.
   *
   * @throws std::invalid_argument If `build_key_indices` is empty or contains a negative index.
   */
  hash_join_build_task(cudf::size_type build_logical_column_count,
                       std::vector<cudf::size_type> build_key_indices,
                       std::vector<cudf::size_type> projected_payload_indices,
                       cudf::null_equality null_equality,
                       bool is_hash_map_cache_enabled);

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
  std::vector<cudf::size_type> _build_key_indices;
  detail::join_materialized_column_layout _column_layout;
  cudf::null_equality _null_equality;
  bool _is_hash_map_cache_enabled;
};

/**
 * @brief Probe a materialized cuDF hash join and publish indirect projected columns.
 *
 * Projection indices address the execution-order `[build_columns..., probe_columns...]` schema.
 * `inner` joins may use either logical input as the build input. A `left` join requires the
 * logical right input as build because cuDF preserves the physical probe input. A logical right
 * join is represented by swapping logical inputs before lowering it as a left join.
 *
 * Probe chunks compact only equality keys before hashing. Match positions are mapped back to
 * original logical rows, and existing probe payload position lists are composed without gathering
 * payload values. Emitted indirect columns retain their original probe sources. Inner
 * outputs use unchecked build and probe gathers. Left outputs nullify out-of-bounds build positions
 * and use unchecked probe positions. An empty projection emits `row_count`.
 *
 * Full joins are unsupported because probing each streamed chunk independently would emit
 * unmatched build rows more than once. The current transform lifecycle has no finalization output
 * for emitting those rows exactly once.
 *
 * # Input shapes
 *
 * `initialize` accepts the `[task_private]` state emitted by `hash_join_build_task::finalize` and
 * returns a shared `[task_private]` state. Each `next` input is
 * any validated column-bearing logical-relation shape.
 *
 * # Output shape
 *
 * A non-empty projection returns `[indirect_column, ...]` in projection order. Projected source
 * columns share position lists by source, and repeated projections share their source state. An
 * empty projection returns `[row_count]`. An inner join whose build fold received no chunks returns
 * schema-correct empty columns. A left join requires a schema-bearing build chunk, including when
 * that chunk contains zero rows, so unmatched build projections have known types.
 *
 * ## Ownership, streams, and memory resources
 *
 * Indirect output retains projected direct columns and the original sources of indirect columns.
 * A retained source can also own unprojected columns. All probe compaction, position-list,
 * join-output, and uncached hash-map allocations use the `mr` supplied to `next`. Match lists
 * independently own their storage, while temporary keys and masks are released after their
 * deallocation streams are ordered behind the join. QEP owns stream ordering between the emitted
 * indirect state and downstream gather. Uncached hash joins allocate key metadata and build scratch
 * from the current device resource, not `mr`.
 *
 * Cached state is immutable during probing and may be shared by concurrent calls on different
 * streams. When caching is disabled, each call constructs a stream-local hash join over the shared
 * immutable build keys. For `P` probe rows, `K` fixed-width keys, `M` matches, and `S` distinct
 * projected indirect probe sources, `next` takes `O(P * K + M * (S + 1))` expected work and
 * storage. Variable-width keys add their materialization and hashing costs. Duplicate keys can
 * produce `M = P * B` matches for `B` build rows.
 */
class hash_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Construct a hash-join probe task.
   *
   * @param[in] probe_key_indices Probe-side equality-key column indices.
   * @param[in] join_type Join semantics. Only `inner` and `left` are supported.
   * @param[in] projection_indices Output column indices in the execution-order schema.
   * @param[in] output_data_types Expected projected output types in emission order.
   *
   * @throws std::invalid_argument If keys or projections are invalid, output metadata is
   * inconsistent, an output type is nested, or `join_type` is unsupported.
   */
  hash_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                       join_type_type join_type,
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
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
