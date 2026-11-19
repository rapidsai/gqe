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

#include <gqe/expression/expression.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/qep/traits/static_iterate_split_trait.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Retain the preserved left input and prepare columns for probing.
 *
 * False and null mask entries are excluded during hash insertion and output selection. Only keys
 * and configured residual columns are materialized and concatenated. Original payload chunks
 * retain their direct or indirect representation. Numeric equality keys, duplicate keys, and both
 * null-equality policies are supported. The retained build state is immutable.
 *
 * # Input shape
 *
 * Each `next` input is a column-bearing `logical_table_view`, with an optional nullable valid mask.
 * Input chunks may mix masked and unmasked forms.
 * Borrowed column and mask storage must remain alive through probing, scanning, and downstream
 * materialization of the retained payload.
 *
 * # State and output shapes
 *
 * `initialize` returns a `[task_private]` chunk accumulator. `finalize` returns a
 * `[task_private]` state retaining original payload chunks and selected materialized columns.
 * The probe task constructs its mutable hash index from these columns during initialization.
 *
 * ## Thread and stream safety
 *
 * Concurrent `next` calls append complete chunks under a mutex. The fold lifecycle orders
 * those calls before `finalize`, which reads the accumulator without locking. The executor
 * orders producers before consumers, and tasks reverse-order retained storage after device use.
 *
 * ## Memory resources
 *
 * GQE-owned allocations use the supplied `mr` and `stream`.
 */
class mark_join_build_task : public fold_task {
 public:
  /**
   * @brief Configure the preserved-side build.
   *
   * @param[in] build_key_indices Nonempty, nonnegative left equality-key indices.
   * @param[in] null_equality Whether null build_key_indices compare equal.
   * @param[in] predicate_column_indices Left-local columns used by the residual join predicate.
   *
   * @throws std::invalid_argument If build_key_indices are empty or contain a negative index.
   */
  mark_join_build_task(std::vector<cudf::size_type> build_key_indices,
                       cudf::null_equality null_equality,
                       std::vector<cudf::size_type> predicate_column_indices = {});

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
  cudf::null_equality _null_equality;
  std::vector<cudf::size_type> _predicate_column_indices;
};

/**
 * @brief Probe the left hash index and atomically mark matching left rows.
 *
 * Optional residual predicates use cuDF AST evaluation, including arithmetic, literals, casts,
 * and disjunctions. A null predicate result does not match. A relation mask excludes false and null
 * rows before probing, independently of the equality policy for null keys.
 *
 * # Input and output shapes
 *
 * `initialize` consumes one build `[task_private]` state and returns an independent
 * `[task_private]` probe accumulator. Each `next` input is a column-bearing `logical_table_view`,
 * including indirect columns and an optional nullable valid mask. Only key and residual columns
 * are materialized, preserving the input row geometry.
 *
 * Each `next` returns a `[task_private]` completion token, not relational rows.
 * Only `mark_join_complete_task` consumes these tokens. Every input, including an empty
 * chunk, emits a token. No input returns the terminal early-exit signal.
 *
 * # Hash map ownership
 *
 * The core uses a CuCo multiset of (31-bit hash, build-row index) pairs. The high hash bit records
 * matches. `initialize` constructs one execution-owned map from the retained build columns.
 * All `next` calls sharing that accumulator mark the same map. Independent accumulators have
 * independent maps, and no probe mutates dependency inputs or immutable column states.
 *
 * ## Thread and stream safety
 *
 * Concurrent probes atomically mark the execution-owned map. The completion barrier
 * orders all probes before scan reads their marks. The executor provides forward stream ordering.
 * Tasks reverse-order retained payloads, maps, and bitmaps after their asynchronous uses.
 *
 * ## Memory resources
 *
 * GQE-owned allocations, AST parser scratch, and literal scalars use `mr` and `stream`.
 * The cuDF `table_device_view` factory has no resource argument, so cuDF controls descriptor
 * scratch for equality keys and AST evaluation. No process-global resource is changed.
 *
 * No match-pair lists or Bloom filters are created. Direct key and residual columns are borrowed.
 */
class mark_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Configure right-side probing.
   *
   * @param[in] probe_key_indices Nonempty, nonnegative right equality-key indices.
   * @param[in] predicate Optional residual expression in [left..., right...] coordinates.
   *
   * @throws std::invalid_argument If probe_key_indices are empty or contain a negative index.
   */
  explicit mark_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                                std::unique_ptr<expression> predicate = nullptr);

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
  std::unique_ptr<expression> _predicate;
};

/**
 * @brief Establish the end-of-probe barrier and retain one token per probe accumulator.
 *
 * # Input and output shapes
 *
 * `initialize` returns a `[task_private]` completion accumulator. Each `next` consumes
 * one `[task_private]` probe token. `finalize` merges the completed marks once and releases
 * probe state. The resulting bitmap is shared read-only by all scan splits. Zero input calls
 * are valid and mean that every active build row is unmatched.
 *
 * ## Thread and stream safety
 *
 * `next` retains token ownership under a mutex. It never reads device marks while concurrent
 * probes may modify them. The executor's fold boundary orders all probe device work before scan.
 */
class mark_join_complete_task : public fold_task {
 public:
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
};

/**
 * @brief Scan completed marks once and emit left semi/anti rows as indirect columns.
 *
 * # Input and output shapes
 *
 * `initialize` takes exactly two `[task_private]` states in `[build, completion]` order.
 * It retains the original build chunks and the completion task's merged bitmap.
 * With no probe tokens every active left row is unmatched.
 *
 * Each `next` call claims one original build chunk, selects marked rows for semi or unmarked
 * rows for anti, and constructs indirect output retaining the original payload sources. Both
 * selections exclude masked-out build rows and compose any existing indirection. An empty
 * projection scans all chunks in the first `next` call and emits one total `[row_count]`.
 * Absent build chunks produce schema-correct empty columns. `next` returns `std::nullopt` after
 * all outputs have been claimed.
 *
 * The static executor obtains the required split count from build predecessor 0 using
 * `required_split_count_from_fold_input()`. It creates one split per build-input adapter,
 * without applying the query partition cap again. Each split emits one retained chunk, or a
 * schema-correct empty output when that adapter slot has no retained chunk. This preserves
 * indirect payloads through the adapter. Initialization rejects a split count that does not
 * cover the retained chunks. Count-only scans use configuration-based selection with
 * `max_splits() == 1` and emit one total.
 *
 * ## Thread, stream, and memory safety
 *
 * The completion barrier makes marks read-only during scanning. The executor orders
 * initialization before `next` calls, including calls on other streams, and waits for all calls
 * before finalization. A mutex protects chunk claims. Selection and output
 * construction run outside the mutex on each caller's stream using its supplied `mr`. Retained
 * input columns and the shared bitmap are reverse-ordered after their stream uses.
 *
 * Each worker uses temporary storage for its current chunk and shares the completion bitmap
 * without copying it.
 */
class mark_join_scan_task : public iterate_task, public static_iterate_split_trait {
 public:
  /**
   * @brief Configure preserved-row output.
   *
   * @param[in] join_type Left-semi or left-anti semantics.
   * @param[in] projection_indices Nonnegative left column indices, in output order. Duplicates are
   * valid.
   * @param[in] output_data_types Matching fixed-width or STRING types. Empty means count-only
   * output.
   *
   * @throws std::invalid_argument If the join type, indices, or output metadata are invalid.
   */
  mark_join_scan_task(join_type_type join_type,
                      std::vector<cudf::size_type> projection_indices,
                      std::vector<cudf::data_type> output_data_types);

  /** @copydoc static_iterate_split_trait::required_split_count_from_fold_input */
  [[nodiscard]] std::optional<std::size_t> required_split_count_from_fold_input()
    const noexcept override;

  /** @copydoc static_iterate_split_trait::max_splits */
  [[nodiscard]] cudf::size_type max_splits() const noexcept override;

  /** @copydoc static_iterate_split_trait::split */
  [[nodiscard]] std::unique_ptr<iterate_task> split(cudf::size_type split_idx,
                                                    cudf::size_type total_splits) const override;

  /** @copydoc iterate_task::initialize */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc iterate_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::optional<cudf::size_type> _split_index;
  cudf::size_type _split_count{1};
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
