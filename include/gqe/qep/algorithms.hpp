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

#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/utility/concepts.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <stack>
#include <unordered_map>
#include <vector>

namespace gqe {
namespace qep {

// -----------------------------------------------------------------------------
// Topological sort
// -----------------------------------------------------------------------------

/**
 * @brief Kahn-style topological sort with a LIFO ready stack.
 *
 * Generic over the node type. The caller supplies callables to enumerate each node's
 * predecessors (used for the initial in-degree count) and successors (used to decrement
 * downstream in-degrees as nodes are emitted).
 *
 * The LIFO ready stack drives each successor chain deep before peeling off siblings,
 * producing the pipeline-like ordering that minimizes in-flight intermediates.
 *
 * Returns the nodes in topological order, or `std::nullopt` if the graph is cyclic. A cycle
 * has no topological order, so rather than expose a partial ordering the sort returns nothing.
 *
 * Complexity: `O(V + E)`.
 *
 * @pre Every node returned by `predecessors(n)` and `successors(n)` must also appear in
 *      `nodes`. Violations are caught by an assertion in debug builds.
 *
 * @tparam Node Node type. Constrained to be equality-comparable and hashable via `std::hash`,
 *   since it is used as an `std::unordered_map` key.
 * @tparam Predecessors Callable `Node -> sized range of Node`. The result's `.size()` is
 *   read once per node to build the initial in-degree count.
 * @tparam Successors Callable `Node -> range of Node`. Iterated once per emitted node.
 *
 * @param[in] nodes All nodes in the graph.
 * @param[in] predecessors Callable returning the predecessors of a node.
 * @param[in] successors Callable returning the successors of a node.
 *
 * @return The nodes in topological order, or `std::nullopt` if the graph contains a cycle.
 */
template <typename Node, typename Predecessors, typename Successors>
  requires std::equality_comparable<Node> && utility::hashable<Node> &&
           requires(Predecessors& predecessors, Successors& successors, Node const& n) {
             // predecessors(n): a sized range of Node (only its size is read).
             { predecessors(n) } -> std::ranges::sized_range;
             requires std::convertible_to<std::ranges::range_value_t<decltype(predecessors(n))>,
                                          Node>;
             // successors(n): an input range of Node (iterated; each element used as a Node).
             { successors(n) } -> std::ranges::input_range;
             requires std::convertible_to<std::ranges::range_value_t<decltype(successors(n))>,
                                          Node>;
           }
[[nodiscard]] std::optional<std::vector<Node>> sort_topologically(std::span<Node const> nodes,
                                                                  Predecessors&& predecessors,
                                                                  Successors&& successors)
{
  std::unordered_map<Node, std::size_t> in_degree;
  in_degree.reserve(nodes.size());

  std::stack<Node, std::vector<Node>> ready;
  for (auto const& n : nodes) {
    auto const& preds = predecessors(n);
    auto const deg    = static_cast<std::size_t>(preds.size());
    in_degree[n]      = deg;
    if (deg == 0) { ready.push(n); }
  }

  std::vector<Node> ordered;
  ordered.reserve(nodes.size());
  while (!ready.empty()) {
    auto n = ready.top();
    ready.pop();
    ordered.push_back(n);
    for (auto const& s : successors(n)) {
      auto it = in_degree.find(s);
      assert(it != in_degree.end() &&
             "sort_topologically: successors(n) returned a node "
             "that is not in the input nodes span");
      if (--(it->second) == 0) { ready.push(s); }
    }
  }

  if (ordered.size() != nodes.size()) { return std::nullopt; }
  return ordered;
}

// -----------------------------------------------------------------------------
// Pipeline partitioning
// -----------------------------------------------------------------------------

/**
 * @brief Result of partitioning a QEP into pipelines.
 *
 * A pipeline is a maximal set of QEP tasks connected by streaming edges. An edge `u → v`
 * is a *streaming* edge when `u->is_pipeline_breaker()` is false. Any edge emerging from
 * a pipeline-breaker task is a *cross-pipeline* edge, and `u`'s downstream consumers
 * therefore belong to a different pipeline than `u`.
 *
 * `pipeline_to_tasks`, `predecessors`, and `successors` are pre-populated with an entry
 * for every pipeline id in `[0, num_pipelines)`, so `.at(p)` is safe for any id in that
 * range (even pipelines with no predecessors or no successors).
 */
struct pipeline_partition {
  std::size_t num_pipelines = 0;

  /// Which pipeline each task belongs to.
  std::unordered_map<task const*, pipeline_id> task_to_pipeline;

  /// Tasks in each pipeline.
  std::unordered_map<pipeline_id, std::vector<task const*>> pipeline_to_tasks;

  /// Pipelines feeding into each pipeline.
  std::unordered_map<pipeline_id, std::vector<pipeline_id>> predecessors;

  /// Pipelines fed by each pipeline.
  std::unordered_map<pipeline_id, std::vector<pipeline_id>> successors;
};

/**
 * @brief Partition the QEP's tasks into pipelines.
 *
 * Union-find over streaming edges: two tasks are in the same pipeline iff they are
 * connected by a path of streaming edges. Edges emerging from any task whose
 * `is_pipeline_breaker()` returns true are treated as cross-pipeline boundaries.
 *
 * Complexity: `O(K · (V + E) · α(V))`, where:
 *
 *  - `K` is the number of pipeline breakers in the QEP.
 *  - `α` is the inverse Ackermann from union-find.
 *
 * The bypass-detection pass walks each breaker's descendant subgraph independently, so the
 * breaker count enters multiplicatively. In typical query plans `K` is a small constant (a
 * handful of folds) and the bound is effectively linear; pathological inputs with
 * `K = Θ(V)` push it to `Θ(V²)`.
 *
 * @param[in] qep The query execution plan to partition.
 *
 * @throws std::logic_error If the QEP is ill-formed — specifically, if a streaming-only
 *         path bypasses a pipeline breaker and reunites it with one of its downstream
 *         consumers, which no pipelining scheme can honor.
 *
 * @return The partition.
 */
[[nodiscard]] pipeline_partition partition_into_pipelines(query_execution_plan const& qep);

/**
 * @brief Return the terminals of a subgraph.
 *
 * A *terminal* of the subgraph is a node in the subgraph that either:
 *
 *  - has no successors at all, or
 *  - has at least one successor outside the subgraph.
 *
 * Pass `qep.tasks()` to get the QEP-wide terminals; pass a pipeline's task list (from
 * `pipeline_partition::pipeline_to_tasks`) to get that pipeline's exits.
 *
 * Complexity: `O(S + E_out)`, where:
 *
 *  - `S` is `subgraph_nodes.size()`.
 *  - `E_out` is the total number of outgoing edges from the subgraph's nodes.
 *
 * @param[in] qep The QEP that supplies the successor relation.
 * @param[in] subgraph_nodes The nodes that define the subgraph (membership is determined
 *                            by pointer identity).
 *
 * @return Terminal tasks of the subgraph, in the order they appear in `subgraph_nodes`.
 */
[[nodiscard]] std::vector<task const*> terminals(query_execution_plan const& qep,
                                                 std::span<task const* const> subgraph_nodes);

}  // namespace qep
}  // namespace gqe
