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
 * @brief Materialize the retained input of a streamed conditional inner join.
 *
 * Only predicate operands and projected columns are compacted and concatenated. Predicates must
 * reference both inputs. No chunks and schema-bearing empty chunks both denote an empty relation.
 * Build layout indices are side-local.
 *
 * Concurrent `next` calls prepare chunks outside the accumulator mutex, which protects complete
 * chunk appends. The fold lifecycle orders finalization after all appends. The executor orders
 * producer and consumer streams. Tasks use the supplied stream and resource for all allocations.
 * Chunk finalization may synchronize on failure to safely release retained inputs. Retained owners
 * order deallocation after probe reads. Callers keep streams, resources, and borrowed storage alive
 * through their final use.
 */
class conditional_inner_join_build_task : public fold_task {
 public:
  /**
   * @brief Configure selective build materialization.
   * @param[in] build_column_count Original build width.
   * @param[in] build_predicate_columns Side-local residual operands.
   * @param[in] build_projected_columns Side-local projected payload columns.
   * @throws std::invalid_argument If the width or indices are invalid.
   */
  conditional_inner_join_build_task(cudf::size_type build_column_count,
                                    std::vector<cudf::size_type> build_predicate_columns,
                                    std::vector<cudf::size_type> build_projected_columns);

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
  detail::join_materialized_column_layout _build_layout;
};

/**
 * @brief Emit every pair whose predicate is true using `cudf::conditional_inner_join`.
 *
 * The predicate and projections use logical `[left, right]` coordinates. Either side may be
 * retained. cuDF receives tables in logical order regardless of execution orientation. False and
 * null predicate results do not match. Each matching pair survives, including duplicate inputs.
 * Projections preserve order and repetition. Empty projections emit `row_count`.
 *
 * Inputs must be representable by `logical_table_view`. Probe materialization reads only predicate
 * operands. Projected probe payloads remain indirect, and existing positions compose
 * once per source. Both match-position lists are retained by projected indirect output.
 *
 * Initialization shares immutable build state. Concurrent probes may enqueue on different streams.
 * The executor owns readiness ordering. Each call orders retained input deallocation after use.
 * Predicate AST nodes and scalar literals remain owned through cuDF evaluation. All allocations
 * use the supplied stream and resource, whose lifetimes remain the caller's responsibility.
 *
 * Only fixed-width and STRING consumed columns and the shared cuDF AST subset are supported.
 * Invalid metadata throws before publication. cuDF reports unsupported predicate types and
 * oversized outputs.
 */
class conditional_inner_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Configure a conditional inner probe.
   * @param[in] left_column_count Original logical-left width.
   * @param[in] right_column_count Original logical-right width.
   * @param[in] residual Owned predicate in combined logical-left/logical-right coordinates.
   * @param[in] is_build_left Whether the retained input is logical left.
   * @param[in] projection_indices Logical combined output indices, including repetitions.
   * @param[in] output_data_types Output types in projection order.
   * @throws std::invalid_argument If the mode, indices, predicate, or metadata is unsupported.
   */
  conditional_inner_join_probe_task(cudf::size_type left_column_count,
                                    cudf::size_type right_column_count,
                                    std::unique_ptr<expression> residual,
                                    bool is_build_left,
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
  std::unique_ptr<expression> _residual;
  bool _is_build_left;

  // Output helpers use build-first coordinates, unlike the constructor's logical coordinates.
  std::vector<cudf::size_type> _projection_indices;

  std::vector<cudf::data_type> _output_data_types;
  detail::join_materialized_column_layout _probe_layout;
  expression_binding _probe_binding;  ///< Immutable mapping shared across probe batches.
};

}  // namespace gqe::qep
