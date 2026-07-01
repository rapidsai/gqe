/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/logical_optimization.hpp>

#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/rules/aggregate_perfect_hash.hpp>
#include <gqe/optimizer/rules/column_name_assignment.hpp>
#include <gqe/optimizer/rules/constant_folding.hpp>
#include <gqe/optimizer/rules/extract_complex_expressions_into_project.hpp>
#include <gqe/optimizer/rules/fix_partial_filter_column_references.hpp>
#include <gqe/optimizer/rules/join_children_swap.hpp>
#include <gqe/optimizer/rules/join_unique_keys.hpp>
#include <gqe/optimizer/rules/mean_decomposition.hpp>
#include <gqe/optimizer/rules/not_not.hpp>
#include <gqe/optimizer/rules/projection_pushdown.hpp>
#include <gqe/optimizer/rules/string_to_int_literal.hpp>
#include <gqe/optimizer/rules/uniqueness_propagation.hpp>
#include <gqe/physical/join.hpp>

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::shared_ptr<gqe::logical::relation> gqe::optimizer::logical_optimizer::optimize(
  std::shared_ptr<logical::relation> logical_relation)
{
  for (auto& rule : _rules) {
    bool rule_applied = false;
    logical_relation  = rule->apply(std::move(logical_relation), rule_applied);
    if (rule_applied) {
      _applied_rule_counts[rule->type()] = _applied_rule_counts[rule->type()] + 1;
    }
  }
  return logical_relation;
}

std::unique_ptr<gqe::optimizer::optimization_rule> gqe::optimizer::logical_optimizer::_make_rule(
  logical_optimization_rule_type rule_type, catalog const* cat)
{
  switch (rule_type) {
    case logical_optimization_rule_type::constant_folding:
      return std::make_unique<constant_folding>(cat);
    case logical_optimization_rule_type::string_to_int_literal:
      return std::make_unique<string_to_int_literal>(cat);
    case logical_optimization_rule_type::not_not_rewrite:
      return std::make_unique<not_not_rewrite>(cat);
    case logical_optimization_rule_type::join_children_swap:
      return std::make_unique<join_children_swap>(cat, gqe::physical::broadcast_policy::right);
    case logical_optimization_rule_type::projection_pushdown:
      return std::make_unique<projection_pushdown>(cat);
    case logical_optimization_rule_type::uniqueness_propagation:
      return std::make_unique<uniqueness_propagation>(cat);
    case logical_optimization_rule_type::join_unique_keys:
      return std::make_unique<join_unique_keys>(cat);
    case logical_optimization_rule_type::aggregate_perfect_hash:
      return std::make_unique<aggregate_perfect_hash>(cat);
    case logical_optimization_rule_type::fix_partial_filter_column_references:
      return std::make_unique<fix_partial_filter_column_references>(cat);
    case logical_optimization_rule_type::column_name_assignment:
      return std::make_unique<column_name_assignment>(cat);
    case logical_optimization_rule_type::complex_expression_extraction_into_project:
      return std::make_unique<complex_expression_extraction_into_project>(cat);
    case logical_optimization_rule_type::mean_decomposition:
      return std::make_unique<mean_decomposition>(cat);
    default:
      throw std::runtime_error("Logical Optimizer: logical_optimization_rule_type " +
                               std::to_string(static_cast<size_t>(rule_type)) + " not supported");
  }
}

namespace {
void traverse_expression_list(
  std::vector<std::unique_ptr<gqe::expression>>& expressions,
  std::vector<cudf::data_type> const& column_types,
  gqe::optimizer::optimization_rule::expression_modifier_functor traverse_f) noexcept
{
  for (auto& expression : expressions) {
    auto new_expr = traverse_f(expression.get(), column_types);
    if (new_expr) { expression = std::move(new_expr); }
  }
}
}  // namespace

class gqe::optimizer::optimization_rule::rewrite_expressions_visitor
  : public gqe::logical::relation_visitor {
 public:
  rewrite_expressions_visitor(optimization_rule const& rule, expression_modifier_functor traverse)
    : _rule{rule}, _traverse{std::move(traverse)}
  {
  }

  // This visitor operates on a single relation's expressions only — no recursion.
  void visit_relation(gqe::logical::relation*) override {}

  void visit(gqe::logical::aggregate_relation* agg) override
  {
    auto column_types = agg->children_unsafe()[0]->data_types();
    traverse_expression_list(agg->_keys, column_types, _traverse);
    auto measures = agg->measures_unsafe();
    for (size_t m_idx = 0; m_idx < measures.size(); m_idx++) {
      auto new_expr = _traverse(measures[m_idx].second, column_types);
      if (new_expr) { agg->_measures[m_idx].second = std::move(new_expr); }
    }
  }

  void visit(gqe::logical::filter_relation* filter) override
  {
    auto condition     = filter->condition();
    auto new_condition = _traverse(condition, filter->children_unsafe()[0]->data_types());
    if (new_condition) { filter->_condition = std::move(new_condition); }
  }

  void visit(gqe::logical::join_relation* join) override
  {
    auto const left_types  = join->children_unsafe()[0]->data_types();
    auto const right_types = join->children_unsafe()[1]->data_types();
    std::vector<cudf::data_type> full_data_types;
    full_data_types.reserve(left_types.size() + right_types.size());
    full_data_types.insert(full_data_types.end(), left_types.begin(), left_types.end());
    full_data_types.insert(full_data_types.end(), right_types.begin(), right_types.end());
    auto condition     = join->condition();
    auto new_condition = _traverse(condition, full_data_types);
    if (new_condition) { join->_condition = std::move(new_condition); }
  }

  void visit(gqe::logical::project_relation* project) override
  {
    traverse_expression_list(
      project->_output_expressions, project->children_unsafe()[0]->data_types(), _traverse);
  }

  void visit(gqe::logical::read_relation* read) override
  {
    auto partial_filter = read->partial_filter_unsafe();
    if (partial_filter) {
      auto new_partial_filter =
        _traverse(partial_filter, _rule._catalog->column_types(read->table_name()));
      if (new_partial_filter) { read->_partial_filter = std::move(new_partial_filter); }
    }
  }

  void visit(gqe::logical::sort_relation* sort) override
  {
    traverse_expression_list(
      sort->_expressions, sort->children_unsafe()[0]->data_types(), _traverse);
  }

  void visit(gqe::logical::window_relation* window) override
  {
    auto column_types = window->children_unsafe()[0]->data_types();
    traverse_expression_list(window->_arguments, column_types, _traverse);
    traverse_expression_list(window->_order_by, column_types, _traverse);
    traverse_expression_list(window->_partition_by, column_types, _traverse);
  }

 private:
  optimization_rule const& _rule;
  expression_modifier_functor _traverse;
};

void gqe::optimizer::optimization_rule::rewrite_relation_expressions(
  logical::relation* relation, expression_modifier_functor f, transform_direction direction) const
{
  if (!relation) return;

  expression_modifier_functor traverse;
  if (direction == transform_direction::UP) {
    traverse = [&](expression* expr, std::vector<cudf::data_type> const& column_types) {
      return _expression_rewrite_up(expr, column_types, f);
    };
  } else if (direction == transform_direction::DOWN) {
    traverse = [&](expression* expr, std::vector<cudf::data_type> const& column_types) {
      return _expression_rewrite_down(expr, column_types, f);
    };
  } else {
    traverse = [&](expression* expr, std::vector<cudf::data_type> const& column_types) {
      return _expression_rewrite(expr, column_types, f);
    };
  }

  rewrite_expressions_visitor visitor{*this, std::move(traverse)};
  relation->accept(visitor);
}

std::unique_ptr<gqe::expression> gqe::optimizer::optimization_rule::rewrite_expression(
  expression* expr, expression_modifier_functor f, transform_direction direction) const
{
  if (!expr) return nullptr;
  // No relation context here, so column types are unavailable; pass an empty list. The functor
  // must not depend on column types (callers such as column-reference index shifting only inspect
  // the expression itself).
  if (direction == transform_direction::UP) {
    return _expression_rewrite_up(expr, {}, std::move(f));
  } else if (direction == transform_direction::DOWN) {
    return _expression_rewrite_down(expr, {}, std::move(f));
  }
  return _expression_rewrite(expr, {}, std::move(f));
}

std::unique_ptr<gqe::expression> gqe::optimizer::optimization_rule::_expression_rewrite(
  expression* expr, std::vector<cudf::data_type> const& column_types, expression_modifier_functor f)
{
  // Try to apply the rewrite rule define in the expression modifier functor
  auto new_expr = f(expr, column_types);
  return new_expr;
}

std::unique_ptr<gqe::expression> gqe::optimizer::optimization_rule::_expression_rewrite_down(
  expression* expr, std::vector<cudf::data_type> const& column_types, expression_modifier_functor f)
{
  // Try to rewrite the current expression in the expression tree
  auto new_expr = _expression_rewrite(expr, column_types, f);
  if (new_expr) { expr = new_expr.get(); }
  // Recursively trying optimize children
  auto children = expr->children();
  for (size_t index = 0; index < children.size(); index++) {
    auto new_child = _expression_rewrite_down(children[index], column_types, f);
    if (new_child) { expr->_children[index] = std::move(new_child); }
  }
  return new_expr;
}

std::unique_ptr<gqe::expression> gqe::optimizer::optimization_rule::_expression_rewrite_up(
  expression* expr, std::vector<cudf::data_type> const& column_types, expression_modifier_functor f)
{
  // Recursively trying optimize children
  auto children = expr->children();
  for (size_t index = 0; index < children.size(); index++) {
    auto new_child = _expression_rewrite_up(children[index], column_types, f);
    if (new_child) { expr->_children[index] = std::move(new_child); }
  }
  // Try to rewrite the current expression in the expression tree
  auto new_expr = _expression_rewrite(expr, column_types, f);
  if (new_expr) { expr = new_expr.get(); }
  return new_expr;
}

void gqe::optimizer::optimization_rule::replace_child_at(logical::relation* relation,
                                                         std::size_t child_idx,
                                                         std::shared_ptr<logical::relation> child)
{
  relation->_children[child_idx] = child;
}

void gqe::optimizer::optimization_rule::clear_partial_filter(logical::read_relation* read)
{
  read->_partial_filter.reset();
}

void gqe::optimizer::optimization_rule::clear_subqueries(logical::relation* relation)
{
  relation->_subqueries.clear();
}

void gqe::optimizer::optimization_rule::set_join_projection_indices(
  logical::join_relation* join, std::vector<cudf::size_type> projection_indices)
{
  join->_projection_indices = std::move(projection_indices);
}

void gqe::optimizer::optimization_rule::set_relation_property(logical::relation* relation,
                                                              std::size_t col_idx,
                                                              column_property::property_id prop)
{
  relation->_relation_traits->_properties.add_column_property(col_idx, prop);
}

void gqe::optimizer::optimization_rule::add_relation_unique_key(logical::relation* relation,
                                                                std::vector<cudf::size_type> key)
{
  relation->_relation_traits->_properties.add_unique_key(std::move(key));
}
