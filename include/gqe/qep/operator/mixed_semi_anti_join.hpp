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

#include <gqe/executor/eval.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/qep/detail/join_materialized_column_layout.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/query_context.hpp>

#include <memory>
#include <vector>

namespace gqe::qep {

/**
 * @brief Retain logical-right equality and predicate columns for streamed mixed semi/anti joins.
 *
 * Each logical input chunk is compacted once for its validity mask. Only the union of keys and
 * predicate columns is materialized. Finalization concatenates the chunks into immutable shared
 * state. No reusable mixed hash table is cached because cuDF builds it for each probe invocation.
 * No input chunks and schema-bearing empty chunks both represent an empty right relation.
 *
 * Concurrent `next` calls prepare chunks outside the accumulator mutex. The mutex protects only
 * the atomic append of a complete chunk. The fold lifecycle orders all appends before finalization.
 * The executor orders producer and consumer streams. Tasks never synchronize those streams.
 * All allocations use the supplied stream and resource. Retained columns order deallocation after
 * each probe use, and the caller keeps explicit streams and resources alive through deallocation.
 */
class mixed_semi_anti_join_build_task : public fold_task {
 public:
  /**
   * @brief Configure selective logical-right materialization.
   * @param[in] right_column_count Original logical-right width.
   * @param[in] right_keys Nonempty side-local equality keys.
   * @param[in] right_predicate_columns Side-local residual operands.
   * @param[in] null_equality Equality-key null policy, independent of predicate null semantics.
   * @throws std::invalid_argument If keys are empty or indices are invalid.
   */
  mixed_semi_anti_join_build_task(cudf::size_type right_column_count,
                                  std::vector<cudf::size_type> right_keys,
                                  std::vector<cudf::size_type> right_predicate_columns,
                                  cudf::null_equality null_equality);

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
  std::vector<cudf::size_type> _right_keys;
  detail::join_materialized_column_layout _right_layout;
  cudf::null_equality _null_equality;
};

/**
 * @brief Select logical-left rows with or without a true equality-and-residual match.
 *
 * Both modes require broadcast-right execution. Semi emits each matching left row once. Anti emits
 * left rows with no true match, including rows for which all residual results are false or null.
 * Equality null matching follows the build configuration independently of residual SQL nulls.
 * Inputs support direct, masked, indirect, and masked-indirect logical tables. Only equality keys
 * and residual operands are compacted. Unconsumed payload values remain deferred.
 *
 * Projection indices address only logical-left columns. Output indirect columns share one logical
 * row selection, composing existing indirection once per distinct source. Order and repeated
 * projections are preserved. Empty projections return row_count. Match order is unspecified.
 *
 * `initialize` retains shared immutable build state. Missing build operands fail during
 * initialization when build columns exist. No-chunk builds skip operand validation. Each probe
 * binds its physical input tables to the original logical predicate without rewriting it. Clones
 * own independent logical predicates. Concurrent probes may use different streams. The
 * executor owns readiness ordering, and each probe orders retained input deallocation after use.
 * cuDF AST nodes and literal scalars are owned through the call, with scalar deallocation ordered
 * on the same stream. All allocations, including scratch, use the supplied resource.
 *
 * Only fixed-width and STRING columns and the shared cuDF AST subset are supported.
 * Invalid metadata fails before work is published.
 */
class mixed_semi_anti_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Configure a mixed logical-left semi/anti probe.
   * @param[in] left_column_count Original logical-left width.
   * @param[in] right_column_count Original logical-right width.
   * @param[in] left_keys Nonempty side-local equality keys.
   * @param[in] residual Owned predicate in combined logical-left/logical-right coordinates.
   * @param[in] join_type Either left_semi or left_anti.
   * @param[in] projection_indices Logical-left output indices, including repetitions.
   * @param[in] output_data_types Output types in projection order.
   * @throws std::invalid_argument If the mode, indices, predicate, or metadata is unsupported.
   */
  mixed_semi_anti_join_probe_task(cudf::size_type left_column_count,
                                  cudf::size_type right_column_count,
                                  std::vector<cudf::size_type> left_keys,
                                  std::unique_ptr<expression> residual,
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
  cudf::size_type _left_column_count;
  cudf::size_type _right_column_count;
  std::vector<cudf::size_type> _left_keys;
  std::unique_ptr<expression> _residual;
  std::vector<cudf::size_type>
    _right_predicate_columns;  ///< Side-local operands checked at initialization.
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
  detail::join_materialized_column_layout _left_layout;
  expression_binding _left_binding;  ///< Immutable mapping shared across probe batches.
};

}  // namespace gqe::qep
