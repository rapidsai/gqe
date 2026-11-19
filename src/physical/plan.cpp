/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/physical/plan.hpp>

#include <gqe/utility/error.hpp>

#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe::physical {
namespace {

[[nodiscard]] std::vector<relation const*> collect_relations(relation const* root)
{
  std::vector<relation const*> relations;
  std::unordered_set<relation const*> visited;
  std::vector<relation const*> pending{root};
  while (!pending.empty()) {
    auto const* current = pending.back();
    pending.pop_back();
    GQE_EXPECTS(current != nullptr,
                "identified_plan: physical relation graph contains a null relation",
                std::invalid_argument);
    auto const insertion = visited.insert(current);
    if (!insertion.second) { continue; }
    relations.push_back(current);

    auto const children   = current->children_unsafe();
    auto const subqueries = current->subqueries_unsafe();
    pending.insert(pending.end(), subqueries.rbegin(), subqueries.rend());
    pending.insert(pending.end(), children.rbegin(), children.rend());
  }
  return relations;
}

}  // namespace

identified_plan::identified_plan(std::shared_ptr<relation> root, plan_node_ids node_ids)
  : _root(std::move(root)), _node_ids(std::move(node_ids))
{
  GQE_EXPECTS(_root != nullptr, "identified_plan: root must not be null", std::invalid_argument);

  auto const relation_order = collect_relations(_root.get());
  std::unordered_set<relation const*> const relations(relation_order.begin(), relation_order.end());
  std::unordered_set<plan_node_id> unique_ids;
  unique_ids.reserve(_node_ids.size());
  for (auto const& [relation, node_id] : _node_ids) {
    GQE_EXPECTS(relation != nullptr,
                "identified_plan: node ID map contains a null relation",
                std::invalid_argument);
    GQE_EXPECTS(relations.contains(relation),
                "identified_plan: node ID map contains a relation outside the plan",
                std::invalid_argument);
    GQE_EXPECTS(node_id > 0, "identified_plan: node IDs must be positive", std::invalid_argument);
    auto const insertion = unique_ids.insert(node_id);
    GQE_EXPECTS(insertion.second,
                "identified_plan: a node ID is assigned to more than one relation",
                std::invalid_argument);
  }
}

plan_node_id identified_plan::node_id(relation const* relation) const
{
  auto const found = _node_ids.find(relation);
  GQE_EXPECTS(found != _node_ids.end(),
              "identified_plan: physical relation has no plan node ID",
              std::logic_error);
  return found->second;
}

identified_plan make_identified_plan(std::shared_ptr<relation> root)
{
  GQE_EXPECTS(
    root != nullptr, "make_identified_plan: root must not be null", std::invalid_argument);

  auto const relations = collect_relations(root.get());
  GQE_EXPECTS(
    relations.size() <= static_cast<std::size_t>(std::numeric_limits<plan_node_id>::max()),
    "make_identified_plan: physical plan has too many relations",
    std::overflow_error);
  plan_node_ids node_ids;
  node_ids.reserve(relations.size());
  for (std::size_t index = 0; index < relations.size(); ++index) {
    node_ids.emplace(relations[index], static_cast<plan_node_id>(index + 1));
  }
  return identified_plan{std::move(root), std::move(node_ids)};
}

}  // namespace gqe::physical
