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

#include <gqe/context_reference.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/query_execution_plan.hpp>

#include <memory>

namespace gqe {

/**
 * @brief Lower a QEP into a static task graph for the existing executor.
 *
 * Constructs an adapter task for each QEP task and wires them into a task graph.
 *
 * # Task graph lowering
 *
 * The lowering of the QEP to a task graph consists of the following phases:
 *
 *   1. The QEP is partitioned into pipelines using a connected components graph partitioning.
 *   2. All QEP tasks are sorted topologically.
 *   3. The pipelines are sorted topologically.
 *   4. Each pipeline is assigned to task graph stage using a depth-layered staging algorithm.
 *   5. Each QEP task is lowered to task graph task.
 *   6. The stage roots are registered.
 *
 * Topological sort ensures that (a) tasks (and pipelines) are lowered in-order
 * so that all predecessors are accessible, and (b) the execution order
 * pipelines the intermediate results.
 *
 * # Depth-layered staging algorithm
 *
 * The lowering of the QEP to a task graph uses a depth-layered staging
 * algorithm.  The goal is to minimize the task graph stages - and thus the
 * number of synchronization barriers - needed to execute the query, while
 * increasing the (potential) parallelism of each stage.
 *
 * The pipelines form a dependency DAG. This DAG's longest dependency chain
 * determines the minimum number of task graph stages needed to lower the QEP.
 * The key insight is that all other dependency chains can be mapped into the
 * same stages, and therefore execute concurrently.
 *
 * The algorithm works as follows:
 *
 *   1. An upward "as-soon-as-possible" sweep determines the earliest possible stage in which a
 * pipeline _can_ execute, based on its predecessors. This is computed as the maximum pipeline ID of
 * its predecessors.
 *   2. A downward "as-late-as-possible" sweep determines the latest possible stage in which a
 * pipeline _must_ execute, based on its successors. This is computed as the minimum pipeline ID of
 * its successors.
 *
 * The result is a layering of pipelines by their depth in the DAG, as seen from the terminal
 * pipeline(s). The algorithm guarantees that each stage in non-empty, because the longest
 * dependency chain determines the number of stages, and thus there must be at least one pipeline
 * per stage.
 *
 * Pipeline breakers are a special case in that they require a dedicated stage. The task graph can
 * instantiate multiple concurrent instances of a given QEP task. The fan-in of the predecessor
 * intermediate results requires a stage boundary. The fan-out of the pipeline breaker's
 * intermediate result also requires a stage boundary. Thus, inserting a dedicated stage ensures a
 * correct execution order.
 *
 * # Stage roots
 *
 * The task graph executor dispatches a stage's root tasks, which recursively dispatch their
 * same-stage predecessors. All tasks in a previous stage must have already run. This leads to the
 * following observation:
 *
 * "A task is a root exactly when no successor shares its stage."
 *
 * As every pipeline breaker succeeds its predecessors, the root task set is purely structural —it
 * doesn't require reverse-edge bookkeeping. The root tasks are:
 *
 *   - a pipeline breaker's fan-in tasks (consumed by the pipeline breaker in the next stage),
 *   - a pipeline breaker's output (consumed further downstream), and
 *   - non-breaker QEP terminals with no consumer.
 *
 * In the lowering, a pipeline breaker's visitor registers the first two as it lowers the QEP task.
 * The the lowering transformation registers the terminals.
 *
 * @param[in] ctx_ref Context shared with every constructed adapter task.
 * @param[in] qep The QEP to lower.
 *
 * @return The constructed task graph.
 */
[[nodiscard]] std::unique_ptr<task_graph> qep_task_graph_transform(
  context_reference ctx_ref, qep::query_execution_plan const& qep);

}  // namespace gqe
