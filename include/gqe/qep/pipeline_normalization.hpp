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

#include <gqe/qep/algorithms.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/task_metadata.hpp>

#include <cudf/types.hpp>

#include <optional>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Executor-neutral graph metadata computed for a normalized QEP.
 *
 * Task pointers refer to tasks owned by the associated `pipeline_normalization_result`.
 */
struct qep_metadata {
  std::vector<task const*> topological_order;
  pipeline_partition pipelines;
  std::vector<pipeline_id> pipeline_topological_order;
};

/**
 * @brief Own a normalized QEP together with its reusable graph metadata.
 *
 * The metadata remains paired with the plan it describes and contains non-owning pointers to its
 * tasks. These pointers remain valid when the `pipeline_normalization_result` is moved and become
 * dangling when the owning plan is destroyed.
 */
class pipeline_normalization_result {
 public:
  pipeline_normalization_result(pipeline_normalization_result const&)                = delete;
  pipeline_normalization_result& operator=(pipeline_normalization_result const&)     = delete;
  pipeline_normalization_result(pipeline_normalization_result&&) noexcept            = default;
  pipeline_normalization_result& operator=(pipeline_normalization_result&&) noexcept = default;

  /** @brief Return the owned normalized QEP. */
  [[nodiscard]] query_execution_plan const& plan() const noexcept { return _plan; }

  /** @brief Return graph metadata whose task pointers refer to `plan()`. */
  [[nodiscard]] qep_metadata const& metadata() const noexcept { return _metadata; }

 private:
  friend pipeline_normalization_result pipeline_normalization(
    query_execution_plan const& source,
    task_output_data_types const& output_data_types,
    task_stable_keys const& stable_keys,
    task_physical_relations const& physical_relations,
    std::optional<cudf::size_type> morsel_max_rows);

  pipeline_normalization_result(query_execution_plan plan, qep_metadata metadata);

  query_execution_plan _plan;
  qep_metadata _metadata;
};

/**
 * @brief Insert missing pipeline boundaries into a QEP.
 *
 * # Normalization
 *
 * Rebuilds `source` as an independent plan in which every pipeline starts with a pipeline
 * initiator and ends with a pipeline breaker. An `iterate_task` is a pipeline initiator: it turns
 * materialized state into pipelined output. A `fold_task` is a pipeline breaker: it turns pipelined
 * input into materialized state.
 *
 * The transformation inserts an `intermediate_state_read_task` when a pipeline breaker feeds a
 * task that requires pipelined input. It inserts `gather_task` followed by
 * `intermediate_state_write_task` wherever pipelined output must be materialized. This boundary is
 * unconditional and does not analyze the predecessor's logical-column representation. Breaker
 * terminals are not generalized into result writers because some, such as storage writers, emit
 * non-relational statistics. Physical lowering handles the terminal Top-K exception explicitly.
 *
 * A stateful transform and the tasks that prepare its internal materialized state are lowered from
 * the same physical relation. A breaker from that relation therefore remains a materialized input.
 * A breaker from a different relation represents an external input and is read as a pipelined
 * input. Normalization requires physical-relation provenance for both tasks when classifying such
 * an edge so that a missing entry cannot silently select the materialized-input behavior.
 *
 * The analysis then considers grouping tasks connected by streaming edges into pipelines, in
 * task topological order. If grouping two tasks would create a pipeline dependency cycle or
 * remove a required pipeline boundary, their streaming edge is replaced with an intermediate
 * writer and reader. The writer materializes the producer's output, and the reader supplies it
 * to the consumer as pipelined input. Producer tasks are not duplicated.
 *
 * Boundaries inserted for the same producer reuse one writer, but each boundary has a separate
 * reader. Sharing a reader could reconnect pipelines that must execute separately. This supports
 * diamonds in which a streaming path bypasses one or more pipeline breakers and rejoins their
 * output at a downstream task.
 *
 * For each proposed boundary, the consumer and all its descendants must have at most one
 * pipelined predecessor. This restriction is checked after the task-family boundary rules above
 * have been applied and includes descendants across pipeline breakers. Independent writers can
 * concatenate partitions in different orders, so their outputs cannot guarantee matching rows
 * for tasks that combine columns from multiple pipelined inputs. Unsupported plans are rejected
 * before the proposed boundaries are inserted.
 *
 * For `V` tasks and `E` edges after applying the task-family boundary rules, this analysis takes
 * expected `O(E * (V + E))` time and `O(V + E)` auxiliary space. Boundary selection is greedy and
 * does not minimize materialization cost. Each inserted writer concatenates its input chunks,
 * and its readers replay the complete materialized result in chunks bounded by `morsel_max_rows`
 * when configured for unordered results. Readers of ordered results emit the complete input once,
 * including explicitly supplied readers that were configured with a row bound. Otherwise, each
 * reader emits one chunk.
 *
 * SORT and TOP-K establish complete-result ordering. Tasks whose `ordering_policy()` preserves
 * input order inherit that guarantee from one relational predecessor. Normalization keeps these
 * paths single-chunk so concurrent chunk processing cannot change FETCH selection or terminal
 * row order. This guarantee also applies to empty results. An unknown task ends the guarantee.
 * Materialized task-private initialization inputs do not count as relational predecessors.
 * Preserving tasks that combine ordered input with multiple relational predecessors are rejected
 * because the QEP has no explicit alignment contract for their sequences.
 *
 * # Constraints
 *
 * The normalized plan satisfies these constraints:
 *
 * - Predecessor order is preserved.
 * - Boundaries inserted to separate pipelines reuse one writer per producer and have separate
 *   readers for each consumer.
 * - Every task without predecessors is an `iterate_task`.
 * - Every `stateful_transform_task` has at least one pipelined predecessor.
 * - The source plan is not modified.
 *
 * @param[in] source Source QEP to clone and normalize.
 * @param[in] output_data_types Output column types for every source task that may require an
 *   inserted writer. An empty type list denotes count-only output.
 * @param[in] stable_keys Unique stable task key for every task in `source`.
 * @param[in] physical_relations Physical-relation provenance for source tasks. Relation pointer
 *   identity defines cohort membership: tasks mapped to the same relation belong to the same
 *   cohort. For a stateful transform, a breaker from a different cohort is converted to a
 *   pipelined input; a breaker from the same cohort remains materialized. For a manually
 *   constructed QEP, assign one stable relation instance to each intended cohort. Pointers are
 *   non-owning, must be non-null when used for classification, and must remain valid for the
 *   duration of the call.
 * @param[in] morsel_max_rows Preferred maximum rows per morsel. This bounds unordered table outputs
 *   emitted by inserted intermediate-state readers. Ordered results override the bound and emit
 *   once so their sequence survives concurrent execution. `std::nullopt` selects
 *   `intermediate_state_read_task::single_output`, which emits each materialized input once.
 *   Count-only input always emits once. A configured value must be positive.
 *
 * @throws std::invalid_argument If `morsel_max_rows` is not positive.
 * @throws std::logic_error If stable task keys are missing or duplicated, output types are missing
 *         for an inserted writer, physical-relation provenance is missing or null when classifying
 *         a pipeline-breaker input to a stateful transform, or the source contains a cycle, has a
 *         non-iterate task without predecessors, has a stateful transform without a pipelined
 *         predecessor, or requires a boundary whose consumer or descendants have multiple
 *         pipelined predecessors, or an order-preserving task combines ordered input with multiple
 *         relational predecessors.
 *
 * @return An independently owned normalized QEP and its executor-neutral graph metadata.
 */
[[nodiscard]] pipeline_normalization_result pipeline_normalization(
  query_execution_plan const& source,
  task_output_data_types const& output_data_types,
  task_stable_keys const& stable_keys,
  task_physical_relations const& physical_relations = {},
  std::optional<cudf::size_type> morsel_max_rows    = std::nullopt);

}  // namespace qep
}  // namespace gqe
