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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/physical/plan.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/task_metadata.hpp>
#include <gqe/storage/table_provider.hpp>

namespace gqe {

struct optimization_parameters;

namespace qep {

/**
 * @brief Own a physically lowered QEP and metadata required by pipeline normalization.
 *
 * The metadata maps are keyed by tasks owned by `plan` and therefore remain coupled to the
 * plan. The physical-relation pointers are non-owning; the physical relation graph must remain
 * alive until pipeline normalization completes. Task pointers in the metadata remain valid across
 * plan moves and become dangling when the plan that owns the tasks is destroyed.
 */
struct physical_qep_transformation_result {
  /** @brief Owned physically lowered QEP. */
  query_execution_plan plan;

  /** @brief Output types keyed by tasks owned by `plan`. */
  task_output_data_types output_data_types;

  /** @brief Non-owning physical relation associated with each task in `plan`. */
  task_physical_relations physical_relations;

  /** @brief Stable task keys keyed by tasks owned by `plan`. */
  task_stable_keys stable_keys;
};

/**
 * @brief Transformation pass that lowers a physical plan to a Query Execution Plan.
 *
 * Lowers a `physical::relation` tree to a `query_execution_plan`. Each physical relation is
 * lowered to one or more QEP tasks. The lowering ensures that physical plan nodes reached
 * via multiple dependency paths are lowered exactly once.
 *
 * Relations without QEP operator support are rejected during lowering.
 *
 * # Optimizer preconditions
 *
 * - Scalar subqueries in conditions must be unnested into joins.
 * - Sort keys, aggregate grouping keys, and aggregate value expressions must be bare column
 *   references. Run the `complex_expression_extraction_into_project` optimizer rule to materialize
 *   computed expressions upstream.
 * - Nested aggregate key or value types are unsupported because the current task output-type
 *   metadata does not represent nested child schemas.
 * - Aggregation functions must be distributive. Non-distributive functions (i.e., algebraic and
 *   holistic functions, per Gray's classification) are rejected during lowering or by the
 *   aggregate and reduce operators. Run an optimizer rule such as `mean_decomposition` to
 *   transform them into distributive functions.
 *
 * # Relational normalization
 *
 * Logical operators consume the logical-relation contract and may preserve masked or indirect
 * column states across their edges. Lowering does not infer the representation of their output.
 * It inserts `gather_task` unconditionally only at boundaries whose consumer requires an ordinary
 * table, including storage writers, materialized reader haystacks, and merge sort.
 *
 * A non-terminal `top_k_task` remains indirect for facade-aware successors. When the complete
 * physical root ends at Top-K, lowering appends a gather because Top-K is a pipeline breaker and
 * therefore does not receive pipeline normalization's terminal writer bridge. Other active
 * terminals are normalized later by the existing gather-and-intermediate-writer bridge.
 *
 * @param[in] physical_plan Identified physical plan. Every lowered relation must have a node ID,
 *   and the relation graph must remain alive until pipeline normalization of the result completes.
 * @param[in] table_provider Resolves table-name references in the physical plan to
 *   `storage::table` instances. With deferred in-memory reads, it must outlive the returned tasks
 *   and every emitted deferred state. Otherwise, it must outlive the call.
 * @param[in] parameters Runtime optimization gates that constrain eligible physical-plan
 *   optimizations. `join_use_unique_keys` selects task implementations during lowering.
 *   `source_reader` independently selects one-shot, chunked, or deferred storage reads. Deferred
 *   reads lower to a reference iterator followed by a resolver. Changing parameters during
 *   execution does not change the selected tasks. Backend compatibility is enforced when
 *   configuring execution rather than during this lowering pass.
 *
 * @throws std::invalid_argument If relation metadata is invalid.
 * @throws std::logic_error If the plan contains an unsupported relation, violates an optimizer
 *         precondition, lacks a required node ID, or produces inconsistent stable task metadata.
 *
 * @return The lowered QEP, its task output types, stable task keys, and
 *         task-to-physical-relation provenance required by `pipeline_normalization`. The returned
 *         QEP is not pipeline-normalized.
 */
[[nodiscard]] physical_qep_transformation_result physical_qep_transformation(
  physical::identified_plan const& physical_plan,
  storage::table_provider const* table_provider,
  optimization_parameters const& parameters);

}  // namespace qep
}  // namespace gqe
