/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/optimizer/rules/sort_fetch_to_top_k.hpp>

#include <gqe/logical/fetch.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/top_k.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gqe::optimizer {

namespace {

/**
 * @brief Give the logical-plan root a parent so the visitor can replace it like any other child.
 *
 * The sentinel exists only while the rule is running and is never returned as part of the plan.
 */
class root_sentinel_relation final : public logical::relation {
 public:
  explicit root_sentinel_relation(std::shared_ptr<logical::relation> root)
    : relation({std::move(root)}, {})
  {
  }

  [[nodiscard]] relation_type type() const noexcept override { return relation_type::user_defined; }

  void accept(logical::relation_visitor& visitor) override { visitor.visit_relation(this); }

  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    return children_unsafe().front()->data_types();
  }

  [[nodiscard]] std::string to_string() const override
  {
    return children_unsafe().front()->to_string();
  }

  bool operator==(logical::relation const& other) const override
  {
    auto const* sentinel = dynamic_cast<root_sentinel_relation const*>(&other);
    return sentinel != nullptr &&
           *children_unsafe().front() == *sentinel->children_unsafe().front();
  }
};

}  // namespace

/**
 * @brief Find one eligible Fetch-Sort node and replace every reference to it.
 *
 * Traversal is post-order. Once a match is found, the visitor keeps walking only to locate all
 * references to the same Fetch in a DAG; a subsequent visitor pass handles the next match.
 */
class sort_fetch_to_top_k::apply_visitor : public logical::relation_visitor {
 public:
  [[nodiscard]] bool rewrote() const noexcept { return _replacement != nullptr; }

  void visit_relation(logical::relation* relation) override
  {
    visit_children(relation);
    replace_rewritten_inputs(relation);
  }

  void visit(logical::fetch_relation* fetch) override
  {
    visit_relation(fetch);
    if (_replacement != nullptr) { return; }

    _replacement = sort_fetch_to_top_k::rewrite(fetch);
    if (_replacement != nullptr) { _rewritten_fetch = fetch; }
  }

 private:
  /**
   * @brief Replace child and subquery references to the matched Fetch with the same Top-K node.
   */
  void replace_rewritten_inputs(logical::relation* relation)
  {
    if (_replacement == nullptr) { return; }

    auto const children = relation->children_safe();
    for (std::size_t i = 0; i < children.size(); ++i) {
      if (children[i].get() == _rewritten_fetch) {
        if (_rewritten_fetch_owner == nullptr) { _rewritten_fetch_owner = children[i]; }
        replace_child_at(relation, i, _replacement);
      }
    }

    auto const subqueries = relation->subqueries_safe();
    for (std::size_t i = 0; i < subqueries.size(); ++i) {
      if (subqueries[i].get() == _rewritten_fetch) {
        if (_rewritten_fetch_owner == nullptr) { _rewritten_fetch_owner = subqueries[i]; }
        replace_subquery_at(relation, i, _replacement);
      }
    }
  }

  logical::fetch_relation* _rewritten_fetch{};
  std::shared_ptr<logical::relation> _rewritten_fetch_owner;
  std::shared_ptr<logical::relation> _replacement;
};

std::shared_ptr<logical::relation> sort_fetch_to_top_k::rewrite(logical::fetch_relation* fetch)
{
  auto const fetch_child = fetch->children_safe().front();
  if (fetch_child->type() != logical::relation::relation_type::sort) { return nullptr; }
  auto const sort = std::static_pointer_cast<logical::sort_relation>(fetch_child);

  // The physical top-K relation supports a single column-reference key. Leave unsupported sorts
  // unchanged so they retain the existing Sort + Fetch execution path.
  auto const expressions = sort->expressions_unsafe();
  auto const orders      = sort->column_orders();
  auto const null_orders = sort->null_orders();
  if (expressions.size() != 1 || orders.size() != 1 || null_orders.size() != 1 ||
      expressions.front()->type() != gqe::expression::expression_type::column_reference) {
    return nullptr;
  }

  // The physical top-K relation requires non-negative bounds. Preserve the existing plan for
  // malformed fetches so this optimizer rule does not introduce a new validation failure.
  if (fetch->offset() < 0 || fetch->count() < 0) return nullptr;

  // Top-K inherits its key, ordering, and subqueries from Sort; Fetch contributes only its bounds.
  auto top_k = std::make_shared<logical::top_k_relation>(sort->children_safe().front(),
                                                         sort->subqueries_safe(),
                                                         expressions.front()->clone(),
                                                         orders.front(),
                                                         null_orders.front(),
                                                         fetch->offset(),
                                                         fetch->count());
  top_k->set_relation_traits(std::make_unique<relation_traits>(fetch->relation_traits()));
  return top_k;
}

std::shared_ptr<logical::relation> sort_fetch_to_top_k::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  root_sentinel_relation sentinel(std::move(root));
  bool did_rewrite = false;
  do {
    apply_visitor visitor;
    sentinel.accept(visitor);
    did_rewrite = visitor.rewrote();
    rule_applied |= did_rewrite;
  } while (did_rewrite);
  return sentinel.children_safe().front();
}

}  // namespace gqe::optimizer
