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

#include <gqe/qep/query_execution_plan.hpp>

#include <gqe/utility/error.hpp>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

// -----------------------------------------------------------------------------
// query_execution_plan
// -----------------------------------------------------------------------------

query_execution_plan::query_execution_plan(
  std::vector<std::unique_ptr<task>>&& tasks,
  std::unordered_map<task const*, std::vector<task const*>>&& predecessors,
  std::unordered_map<task const*, std::vector<task const*>>&& successors)
  : _tasks(std::move(tasks)),
    _predecessors(std::move(predecessors)),
    _successors(std::move(successors))
{
}

void query_execution_plan::accept(qep_visitor& visitor) const
{
  for (std::unique_ptr<task> const& t : _tasks) {
    t->accept(visitor);
  }
}

std::vector<task const*> query_execution_plan::tasks() const
{
  std::vector<task const*> result;
  result.reserve(_tasks.size());
  for (auto const& t : _tasks) {
    result.push_back(t.get());
  }
  return result;
}

std::vector<task const*> query_execution_plan::predecessors(task const* t) const
{
  auto it = _predecessors.find(t);
  return it != _predecessors.end() ? it->second : std::vector<task const*>{};
}

std::vector<task const*> query_execution_plan::successors(task const* t) const
{
  auto it = _successors.find(t);
  return it != _successors.end() ? it->second : std::vector<task const*>{};
}

// -----------------------------------------------------------------------------
// query_execution_plan_builder
// -----------------------------------------------------------------------------

query_execution_plan_builder::query_execution_plan_builder() = default;

query_execution_plan_builder& query_execution_plan_builder::add_task(std::unique_ptr<task> new_task)
{
  GQE_EXPECTS(new_task != nullptr, "add_task received null task");

  task const* t_ptr = new_task.get();

  GQE_EXPECTS(!_tasks.contains(t_ptr), "Task already exists in the QEP");

  _tasks.insert(std::move(new_task));

  return *this;
}

query_execution_plan_builder& query_execution_plan_builder::add_successor(task const* predecessor,
                                                                          task const* successor)
{
  GQE_EXPECTS(_tasks.contains(predecessor), "Predecessor not registered in the QEP");
  GQE_EXPECTS(_tasks.contains(successor), "Successor not registered in the QEP");

  auto& succs = _successors[predecessor];
  GQE_EXPECTS(std::find(succs.begin(), succs.end(), successor) == succs.end(),
              "Successor relationship already registered");

  // Track predecessors in `add_successor` call order so QEP transforms can rely on a
  // deterministic predecessor order (e.g. stateful_transform tasks distinguish the init and
  // streaming predecessors by position).
  succs.push_back(successor);
  _predecessors[successor].push_back(predecessor);

  return *this;
}

query_execution_plan query_execution_plan_builder::build()
{
  bool has_root = false;
  for (auto const& t : _tasks) {
    if (!_predecessors.contains(t.get())) {
      has_root = true;
      break;
    }
  }
  GQE_EXPECTS(has_root, "query_execution_plan_builder::build: QEP has no root tasks");

  // Move owned tasks from the set into a vector for the plan.
  std::vector<std::unique_ptr<task>> tasks;
  tasks.reserve(_tasks.size());
  while (!_tasks.empty()) {
    auto node = _tasks.extract(_tasks.begin());
    tasks.push_back(std::move(node.value()));
  }

  return query_execution_plan(std::move(tasks), std::move(_predecessors), std::move(_successors));
}

}  // namespace qep
}  // namespace gqe
