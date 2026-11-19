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
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/query_context.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class expression;

namespace qep {

/**
 * @brief Column selections for equality-match candidates consumed by a mixed-join filter.
 *
 * Logical indices address the original combined `[left columns..., right columns...]` schema.
 * Conditional and output indices address columns of the candidate table. The filter task
 * validates these selections against its logical widths, predicate, and schemas at construction.
 */
struct mixed_join_candidate_layout {
  /** Original combined logical index represented by each candidate column. */
  std::vector<cudf::size_type> logical_candidate_indices;

  /** Candidate columns selected for the logical-left predicate input. */
  std::vector<cudf::size_type> left_conditional_indices;

  /** Candidate columns selected for the logical-right predicate input. */
  std::vector<cudf::size_type> right_conditional_indices;

  /** Candidate columns emitted in final projection order, including repeated selections. */
  std::vector<cudf::size_type> output_indices;
};

/**
 * @brief Filter hash-join candidate position lists with a residual predicate.
 *
 * The input is an unmasked indirect candidate table. Each row represents either a pair that
 * matched the equality keys or, for a left join, an unmatched logical-left row with null-extended
 * right columns. Candidate columns retain their indices in the original combined logical schema.
 *
 * The residual retains its original logical coordinates. The task prepares immutable expression
 * bindings from its selected columns. Each batch borrows those bindings to resolve references
 * against source tables or a temporary table of gathered predicate operands without rewriting the
 * expression. Concurrent calls share only immutable bindings.
 *
 * Predicate evaluation uses one of two paths:
 *
 * - When each logical side has one compatible gather source, evaluate through its existing
 *   source columns and candidate positions.
 * - For inner joins with multiple sources per side or inherited nullifying sources, gather the
 *   predicate operands into candidate-row order before evaluation. Gathering resolves missing
 *   source rows to null values. This temporary table is used only to evaluate the predicate.
 *
 * Both paths retain projected payloads as indirect columns. They do not materialize the final
 * output payloads.
 *
 * Conditional mappings select the candidate columns addressed by the residual predicate. Output
 * mappings select the columns returned in final projection order.
 * Left joins retain one null-extended row for each logical-left row with no candidate satisfying
 * the residual predicate.
 *
 * Inner joins support multiple underlying sources per logical side. Left joins require one source
 * per side and reject multiple-source input. A zero-row plain table with the candidate schema is
 * also accepted for inner joins.
 */
class filter_join_indices_task : public stateless_transform_task {
 public:
  /**
   * @brief Construct a residual candidate filter.
   *
   * @param[in] join_type Join semantics. Only `inner` and `left` are supported.
   * @param[in] residual Residual predicate in original combined logical-input coordinates.
   * @param[in] left_column_count Original logical-left schema width.
   * @param[in] right_column_count Original logical-right schema width.
   * @param[in] layout Candidate columns and their predicate-input and output selections.
   * @param[in] candidate_data_types Input candidate schema.
   * @param[in] output_data_types Final projected output schema.
   * @throws std::invalid_argument If logical widths, mappings, output types, or residual references
   * are inconsistent. Constructor checks require every residual operand in its side's selection.
   */
  filter_join_indices_task(join_type_type join_type,
                           std::unique_ptr<expression> residual,
                           cudf::size_type left_column_count,
                           cudf::size_type right_column_count,
                           mixed_join_candidate_layout layout,
                           std::vector<cudf::data_type> candidate_data_types,
                           std::vector<cudf::data_type> output_data_types);

  /** @copydoc stateless_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  /** @brief Gather predicate operands, select candidate rows, and retain indirect output payloads.
   */
  [[nodiscard]] state_container filter_with_gathered_predicate_operands(
    state_container_view inputs,
    indirect_table_view const& candidates,
    query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const;

  /** @brief Evaluate source columns through candidate positions and retain indirect output
   * payloads. */
  [[nodiscard]] state_container filter_with_source_positions(
    state_container_view inputs,
    indirect_table_view const& candidates,
    indirect_gather_source const& left_source,
    indirect_gather_source const& right_source,
    query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const;

  join_type_type _join_type;
  std::unique_ptr<expression> _residual;
  cudf::size_type _left_column_count;
  cudf::size_type _right_column_count;
  mixed_join_candidate_layout _layout;
  std::vector<cudf::data_type> _candidate_data_types;
  std::vector<cudf::data_type> _output_data_types;
  std::vector<cudf::size_type> _conditional_indices;
  expression_binding _left_binding;
  expression_binding _right_binding;
  expression_binding _combined_binding;
};

}  // namespace qep
}  // namespace gqe
