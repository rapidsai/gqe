/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/rules/column_name_assignment.hpp>
#include <gqe/optimizer/rules/fix_partial_filter_column_references.hpp>
#include <gqe/optimizer/rules/join_children_swap.hpp>
#include <gqe/optimizer/rules/join_unique_keys.hpp>
#include <gqe/optimizer/rules/not_not.hpp>
#include <gqe/optimizer/rules/projection_pushdown.hpp>
#include <gqe/optimizer/rules/string_to_int_literal.hpp>
#include <gqe/optimizer/rules/uniqueness_propagation.hpp>
#include <gqe/physical/join.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

std::shared_ptr<gqe::logical::relation> gqe::optimizer::logical_optimizer::optimize(
  std::shared_ptr<logical::relation> logical_relation)
{
  for (auto& rule : _rules) {
    bool rule_applied = false;
    // Attempt to apply the rule given the direction
    switch (rule->direction()) {
      case optimization_rule::transform_direction::NONE:
        logical_relation = _optimize(logical_relation, *rule, rule_applied);
        break;
      case optimization_rule::transform_direction::DOWN:
        logical_relation = _optimize_down(logical_relation, *rule, rule_applied);
        break;
      case optimization_rule::transform_direction::UP:
        logical_relation = _optimize_up(logical_relation, *rule, rule_applied);
        break;
    }
    // Update rule application counts
    if (rule_applied) {
      _applied_rule_counts[rule->type()] = _applied_rule_counts[rule->type()] + 1;
    }
  }

  return logical_relation;
}

std::shared_ptr<gqe::logical::relation> gqe::optimizer::logical_optimizer::_optimize(
  std::shared_ptr<logical::relation> logical_relation,
  const optimization_rule& rule,
  bool& rule_applied)
{
  // Try to apply the rule
  auto optimized_plan = rule.try_optimize(logical_relation, rule_applied);
  assert(optimized_plan);
  return optimized_plan;
}

std::shared_ptr<gqe::logical::relation> gqe::optimizer::logical_optimizer::_optimize_down(
  std::shared_ptr<logical::relation> logical_relation,
  const optimization_rule& rule,
  bool& rule_applied)
{
  // Try to optimize the current relation
  logical_relation = this->_optimize(logical_relation, rule, rule_applied);
  // Recursively trying optimize children
  auto children = logical_relation->children_safe();
  for (size_t index = 0; index < logical_relation->children_size(); index++) {
    logical_relation->_children[index] = this->_optimize_down(children[index], rule, rule_applied);
  }
  return logical_relation;
}

std::shared_ptr<gqe::logical::relation> gqe::optimizer::logical_optimizer::_optimize_up(
  std::shared_ptr<logical::relation> logical_relation,
  const optimization_rule& rule,
  bool& rule_applied)
{
  // Recursively trying optimize children
  auto children = logical_relation->children_safe();
  for (size_t index = 0; index < logical_relation->children_size(); index++) {
    logical_relation->_children[index] = this->_optimize_up(children[index], rule, rule_applied);
  }
  // Try to optimize the current relation
  return this->_optimize(logical_relation, rule, rule_applied);
}

std::unique_ptr<gqe::optimizer::optimization_rule> gqe::optimizer::logical_optimizer::_make_rule(
  logical_optimization_rule_type rule_type, catalog const* cat)
{
  switch (rule_type) {
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
    case logical_optimization_rule_type::fix_partial_filter_column_references:
      return std::make_unique<fix_partial_filter_column_references>(cat);
    case logical_optimization_rule_type::column_name_assignment:
      return std::make_unique<column_name_assignment>(cat);
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

  switch (relation->type()) {
    case relation_t::aggregate: {
      auto agg          = dynamic_cast<logical::aggregate_relation*>(relation);
      auto column_types = agg->children_unsafe()[0]->data_types();
      // Apply to keys
      traverse_expression_list(agg->_keys, column_types, traverse);
      // Apply to measures
      auto measures = agg->measures_unsafe();
      for (size_t m_idx = 0; m_idx < measures.size(); m_idx++) {
        auto new_expr = traverse(measures[m_idx].second, column_types);
        if (new_expr) { agg->_measures[m_idx].second = std::move(new_expr); }
      }

      break;
    }
    case relation_t::filter: {
      auto filter = dynamic_cast<logical::filter_relation*>(relation);
      // Apply to condition
      auto condition     = filter->condition();
      auto new_condition = traverse(condition, filter->children_unsafe()[0]->data_types());
      if (new_condition) { filter->_condition = std::move(new_condition); }
      break;
    }
    case relation_t::join: {
      auto join = dynamic_cast<logical::join_relation*>(relation);

      std::vector<cudf::data_type> full_data_types;
      for (auto const& column_type : join->children_unsafe()[0]->data_types())
        full_data_types.push_back(column_type);
      for (auto const& column_type : join->children_unsafe()[1]->data_types())
        full_data_types.push_back(column_type);

      // Apply to condition
      auto condition     = join->condition();
      auto new_condition = traverse(condition, full_data_types);
      if (new_condition) { join->_condition = std::move(new_condition); }
      break;
    }
    case relation_t::project: {
      auto project = dynamic_cast<logical::project_relation*>(relation);
      // Apply to output expressions
      traverse_expression_list(
        project->_output_expressions, project->children_unsafe()[0]->data_types(), traverse);
      break;
    }
    case relation_t::read: {
      auto read = dynamic_cast<logical::read_relation*>(relation);
      // Apply to partial filter
      auto partial_filter = read->partial_filter_unsafe();
      if (partial_filter) {
        auto new_partial_filter =
          traverse(partial_filter, _catalog->column_types(read->table_name()));
        if (new_partial_filter) { read->_partial_filter = std::move(new_partial_filter); }
      }
      break;
    }
    case relation_t::sort: {
      auto sort = dynamic_cast<logical::sort_relation*>(relation);
      // Apply to expressions
      traverse_expression_list(
        sort->_expressions, sort->children_unsafe()[0]->data_types(), traverse);
      break;
    }
    case relation_t::window: {
      auto window       = dynamic_cast<logical::window_relation*>(relation);
      auto column_types = window->children_unsafe()[0]->data_types();
      // Apply to arguments
      traverse_expression_list(window->_arguments, column_types, traverse);
      // Apply to order-by keys
      traverse_expression_list(window->_order_by, column_types, traverse);
      // Apply to partition-by keys
      traverse_expression_list(window->_partition_by, column_types, traverse);
      break;
    }
    case relation_t::fetch:
    case relation_t::set:
    case relation_t::user_defined:
    default: return;
  }
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

void gqe::optimizer::optimization_rule::set_relation_property(logical::relation* relation,
                                                              std::size_t col_idx,
                                                              column_property::property_id prop)
{
  relation->_relation_traits->_properties.add_column_property(col_idx, prop);
}
