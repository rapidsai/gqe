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

#include <gqe/optimizer/rules/identity_project_elimination.hpp>

#include <gqe/logical/project.hpp>
#include <gqe/logical/relation.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace gqe::optimizer {

using relation_t = gqe::logical::relation::relation_type;

namespace {

/**
 * @brief If `child` is an identity project, return the child it wraps; otherwise `nullptr`.
 *
 * Unwraps a single level. Callers loop on this to collapse a chain of stacked identity projects;
 * that terminates because removal preserves the column count, so it can never turn a
 * non-identity parent into an identity one.
 */
std::shared_ptr<gqe::logical::relation> unwrap_identity_project(
  std::shared_ptr<gqe::logical::relation> const& child)
{
  if (child->type() != relation_t::project) { return nullptr; }
  auto project = std::static_pointer_cast<gqe::logical::project_relation>(child);
  if (!project->is_identity()) { return nullptr; }
  auto grandchildren = project->children_safe();
  assert(grandchildren.size() == 1);
  return grandchildren[0];
}

}  // namespace

/**
 * @brief Post-order visitor that splices identity projects out of each node's children and
 * subqueries.
 *
 * A subquery root belongs to no parent's child list, so children and subqueries are fixed up in
 * separate passes. The visitor rewrites a node's attachments, so it cannot remove a project that
 * is the root of the whole plan — `apply()` handles that case separately.
 */
class identity_project_elimination::apply_visitor : public gqe::logical::relation_visitor {
 public:
  explicit apply_visitor(bool& rule_applied) : _rule_applied{rule_applied} {}

  void visit_relation(gqe::logical::relation* rel) override
  {
    visit_children(rel);  // post-order: collapse descendants first

    collapse_identity_chains(
      rel->children_safe(),
      [rel](std::size_t i, std::shared_ptr<gqe::logical::relation> const& replacement) {
        identity_project_elimination::replace_child_at(rel, i, replacement);
      });
    collapse_identity_chains(
      rel->subqueries_safe(),
      [rel](std::size_t i, std::shared_ptr<gqe::logical::relation> const& replacement) {
        identity_project_elimination::replace_subquery_at(rel, i, replacement);
      });
  }

 private:
  /**
   * @brief Splice out the chain of identity projects rooted at each entry of `nodes`.
   *
   * @param nodes Relations to inspect — a copy of the parent's child or subquery list.
   * @param replace Installs a replacement relation at the given index on the parent.
   */
  template <typename ReplaceFn>
  void collapse_identity_chains(std::vector<std::shared_ptr<gqe::logical::relation>> nodes,
                                ReplaceFn replace)
  {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      while (auto replacement = unwrap_identity_project(nodes[i])) {
        replace(i, replacement);
        nodes[i]      = replacement;
        _rule_applied = true;
      }
    }
  }

  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> identity_project_elimination::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{rule_applied};
  root->accept(visitor);

  // The visitor rewrites children from each parent, so a root that is itself an identity project
  // (or a chain of them) has no parent to splice it out — handle that here.
  while (auto replacement = unwrap_identity_project(root)) {
    root         = replacement;
    rule_applied = true;
  }
  return root;
}

}  // namespace gqe::optimizer
