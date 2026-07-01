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

#include <gqe/optimizer/rules/constant_folding.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace {

bool is_trivial_true_literal(gqe::expression const* expr)
{
  if (!expr) return false;
  auto const* lit = dynamic_cast<gqe::literal_expression<bool> const*>(expr);
  return lit != nullptr && !lit->is_null() && lit->value();
}

// Returns a project_relation over the child if `filter` has a trivial-true condition; nullptr
// otherwise. projection_indices are mapped 1:1 to column_reference_expressions; an empty index
// list produces a zero-output-expression project (0-column schema), matching the filter's contract.
// Currently, we only handle the case where the filter is a trivial-true literal. For other cases,
// e.g. true AND x, we return nullptr. In the future, we can also handle such case: true AND x → x.
std::shared_ptr<gqe::logical::relation> try_rewrite_trivial_filter(
  gqe::logical::filter_relation* filter)
{
  if (!is_trivial_true_literal(filter->condition())) return nullptr;
  auto child          = filter->children_safe()[0];
  auto const& indices = filter->projection_indices();
  std::vector<std::unique_ptr<gqe::expression>> output_expressions;
  output_expressions.reserve(indices.size());
  for (auto idx : indices)
    output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(idx));
  return std::make_shared<gqe::logical::project_relation>(
    child, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(output_expressions));
}

}  // namespace

namespace gqe::optimizer {

class constant_folding::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(constant_folding const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit_relation(gqe::logical::relation* rel) override
  {
    visit_children(rel);  // post-order: recurse into children first

    // Clear trivial-true partial_filter on read relations.
    if (rel->type() == gqe::logical::relation::relation_type::read) {
      auto* read = static_cast<gqe::logical::read_relation*>(rel);
      if (is_trivial_true_literal(read->partial_filter_unsafe())) {
        constant_folding::clear_partial_filter(read);
        _rule_applied = true;
      }
    }

    // Rewrite children that are trivial-true filter wrappers.
    auto children = rel->children_safe();
    for (std::size_t i = 0; i < children.size(); ++i) {
      auto& child = children[i];
      if (child->type() != gqe::logical::relation::relation_type::filter) continue;
      auto* filter = static_cast<gqe::logical::filter_relation*>(child.get());
      if (auto rewritten = try_rewrite_trivial_filter(filter)) {
        constant_folding::replace_child_at(rel, i, rewritten);
        _rule_applied = true;
      }
    }
  }

 private:
  constant_folding const& _rule;
  bool& _rule_applied;
};

}  // namespace gqe::optimizer

std::shared_ptr<gqe::logical::relation> gqe::optimizer::constant_folding::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);

  // Handle root separately — the visitor only sees children, not the root itself.
  if (root->type() == logical::relation::relation_type::filter) {
    auto* filter = static_cast<logical::filter_relation*>(root.get());
    if (auto rewritten = try_rewrite_trivial_filter(filter)) {
      rule_applied = true;
      return rewritten;
    }
  }
  return root;
}
