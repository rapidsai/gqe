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

#include <gqe/qep/algorithms.hpp>

#include <gqe/qep/task.hpp>
#include <gqe/utility/error.hpp>

#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Union-find over `task const*` keys with path compression and union-by-rank.
 *
 * Both heuristics together yield amortized `O(α(V))` per operation. The partitioning
 * doesn't care which task names a component, only that connected tasks share a root.
 */
class task_union_find {
 public:
  void make_set(task const* t)
  {
    if (!_parent.contains(t)) {
      _parent[t] = t;
      _rank[t]   = 0;
    }
  }

  task const* find(task const* t)
  {
    auto* p = _parent[t];
    if (p == t) {
      return t;
    } else {
      return _parent[t] = find(p);
    }
  }

  void unite(task const* a, task const* b)
  {
    auto* ra = find(a);
    auto* rb = find(b);
    if (ra == rb) { return; }
    auto const rank_a = _rank[ra];
    auto const rank_b = _rank[rb];
    if (rank_a < rank_b) {
      _parent[ra] = rb;
    } else if (rank_a > rank_b) {
      _parent[rb] = ra;
    } else {
      _parent[rb] = ra;
      ++_rank[ra];
    }
  }

 private:
  std::unordered_map<task const*, task const*> _parent;
  std::unordered_map<task const*, std::size_t> _rank;
};

}  // namespace

pipeline_partition partition_into_pipelines(query_execution_plan const& qep)
{
  auto const all = qep.tasks();

  // Step 1: union-find each task with the destinations of its streaming outgoing edges.
  task_union_find uf;
  for (auto* t : all) {
    uf.make_set(t);
  }
  for (auto* t : all) {
    if (t->is_pipeline_breaker()) {
      // Edges from this task are cross-pipeline; do not union with successors.
      continue;
    }
    for (auto* s : qep.successors(t)) {
      uf.make_set(s);
      uf.unite(t, s);
    }
  }

  // Step 2: reject ill-formed shapes.
  //
  // Every pipeline breaker `b` strictly precedes its descendants (cross-pipeline edges
  // out of `b` are sequencing fences), so no descendant of `b` may end up in `b`'s own
  // pipeline. If one does, a streaming-only path bypassed `b` and reunited it with a
  // downstream task — directly:
  //
  //   A ────► B (breaker) ────► D
  //   │                         ▲
  //   └──► C ───────────────────┘
  //
  // …or transitively through another breaker (creating a cycle in the would-be pipeline
  // graph that would prevent it from being a DAG):
  //
  //                  ┌──────────────────────────────────────┐
  //                  ▼                                      │
  //   A ────► B1 (breaker) ────► S ────► B2 (breaker) ────► T
  //   │                                                     ▲
  //   └─────────────────────────────────────────────────────┘
  //
  // No pipelining scheme honors either shape — materializing the streaming feed would
  // itself be a pipeline break, and the upstream source can't be duplicated.
  for (auto* t : all) {
    if (!t->is_pipeline_breaker()) { continue; }
    std::stack<task const*, std::vector<task const*>> stack;
    for (auto* s : qep.successors(t)) {
      stack.push(s);
    }
    std::unordered_set<task const*> visited;
    while (!stack.empty()) {
      auto* d = stack.top();
      stack.pop();
      if (!visited.insert(d).second) { continue; }
      GQE_EXPECTS(uf.find(t) != uf.find(d),
                  "partition_into_pipelines: ill-formed QEP — a streaming-only path "
                  "bypasses a pipeline breaker and reunites it with one of its "
                  "downstream consumers.",
                  std::logic_error);
      for (auto* ds : qep.successors(d)) {
        stack.push(ds);
      }
    }
  }

  // Step 3: assign dense pipeline ids to the union-find roots. Pre-populate every
  // pipeline's entry in `pipeline_to_tasks`, `predecessors`, and `successors` so callers
  // can use `.at(p)` unconditionally for every pipeline id.
  std::unordered_map<task const*, pipeline_id> root_to_id;
  pipeline_partition out;
  for (auto* t : all) {
    auto* r = uf.find(t);
    auto it = root_to_id.find(r);
    pipeline_id id;
    if (it == root_to_id.end()) {
      id            = pipeline_id(out.num_pipelines++);
      root_to_id[r] = id;
      out.pipeline_to_tasks[id];
      out.predecessors[id];
      out.successors[id];
    } else {
      id = it->second;
    }
    out.task_to_pipeline[t] = id;
    out.pipeline_to_tasks[id].push_back(t);
  }

  // Step 4: build the pipeline-DAG by walking cross-pipeline edges.
  //
  // Determinism: entries in `predecessors`/`successors` appear in the order their
  // corresponding cross-pipeline edges are first encountered. The outer loop iterates
  // `all` (which mirrors `qep.tasks()` order), so downstream consumers see a stable,
  // QEP-derived ordering.
  std::unordered_map<pipeline_id, std::unordered_set<std::size_t>> succ_sets;
  std::unordered_map<pipeline_id, std::unordered_set<std::size_t>> pred_sets;
  for (auto* t : all) {
    auto const t_pipe = out.task_to_pipeline.at(t);
    for (auto* s : qep.successors(t)) {
      auto const s_pipe = out.task_to_pipeline.at(s);
      if (s_pipe == t_pipe) { continue; }  // intra-pipeline edge.
      if (succ_sets[t_pipe].insert(s_pipe.value()).second) {
        out.successors[t_pipe].push_back(s_pipe);
      }
      if (pred_sets[s_pipe].insert(t_pipe.value()).second) {
        out.predecessors[s_pipe].push_back(t_pipe);
      }
    }
  }

  return out;
}

std::vector<task const*> terminals(query_execution_plan const& qep,
                                   std::span<task const* const> subgraph_nodes)
{
  std::unordered_set<task const*> subgraph(subgraph_nodes.begin(), subgraph_nodes.end());

  std::vector<task const*> result;
  for (auto* t : subgraph_nodes) {
    auto const succs       = qep.successors(t);
    bool const is_terminal = succs.empty() || std::any_of(succs.begin(), succs.end(), [&](auto* s) {
                               return !subgraph.contains(s);
                             });
    if (is_terminal) { result.push_back(t); }
  }
  return result;
}

}  // namespace qep
}  // namespace gqe
