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
 * @brief Compact logical-right predicate columns into immutable shared build state.
 *
 * Inputs must be representable by `logical_table_view`. Only predicate columns are materialized.
 *
 * Concurrent calls append complete chunks under a mutex. Finalization concatenates them after
 * the fold ends. No chunks and zero-row chunks both represent an empty right relation.
 * The executor orders streams. Concatenation synchronizes on failure to protect input lifetimes.
 * Allocations use the supplied stream and resource, which must outlive their uses.
 */
class conditional_semi_anti_join_build_task : public fold_task {
 public:
  /**
   * @brief Configure selective logical-right materialization.
   * @param[in] right_column_count Original logical-right width.
   * @param[in] right_predicate_columns Side-local predicate operands.
   * @throws std::invalid_argument If the width or indices are invalid or no predicate columns are
   * supplied.
   */
  conditional_semi_anti_join_build_task(cudf::size_type right_column_count,
                                        std::vector<cudf::size_type> right_predicate_columns);

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
  detail::join_materialized_column_layout _right_layout;
};

/**
 * @brief Select logical-left rows with (semi) or without (anti) a true predicate match.
 *
 * Requires broadcast-right execution and a supported predicate referencing both inputs.
 * False or null predicate results are not matches. Inputs must be representable by
 * `logical_table_view`.
 * Only fixed-width and STRING operands are supported. Unconsumed payload stays deferred.
 * Left-only projections preserve order and repetitions. Empty projections return `row_count`.
 *
 * Initialization shares immutable build state and checks operand coverage unless no chunks exist.
 * Probes may run concurrently on executor-ordered streams and order input release after use.
 * All allocations use the supplied stream and resource, which must outlive their uses.
 */
class conditional_semi_anti_join_probe_task : public stateful_transform_task {
 public:
  /**
   * @brief Configure a conditional logical-left semi/anti probe.
   * @param[in] left_column_count Original logical-left width.
   * @param[in] right_column_count Original logical-right width.
   * @param[in] residual Owned predicate in combined logical-left/logical-right coordinates.
   * @param[in] join_type Either left_semi or left_anti.
   * @param[in] projection_indices Logical-left output indices, including repetitions.
   * @param[in] output_data_types Output types in projection order.
   * @throws std::invalid_argument If the mode, indices, predicate, or metadata is unsupported.
   */
  conditional_semi_anti_join_probe_task(cudf::size_type left_column_count,
                                        cudf::size_type right_column_count,
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
  std::unique_ptr<expression> _residual;
  join_type_type _join_type;
  std::vector<cudf::size_type> _projection_indices;
  std::vector<cudf::data_type> _output_data_types;
  detail::join_materialized_column_layout _left_layout;
  expression_binding _left_binding;  ///< Immutable mapping shared across probe batches.
  std::vector<cudf::size_type> _right_predicate_columns;
};

}  // namespace gqe::qep
