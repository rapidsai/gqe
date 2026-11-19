/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/extract_join_predicate_into_project.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/optimizer/rules/join_extraction_utils.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe::optimizer {

namespace {

using expr_type = gqe::expression::expression_type;

[[nodiscard]] std::vector<cudf::data_type> combined_column_types(
  gqe::logical::relation const* left, gqe::logical::relation const* right)
{
  auto const left_types  = left->data_types();
  auto const right_types = right->data_types();
  std::vector<cudf::data_type> out;
  out.reserve(left_types.size() + right_types.size());
  out.insert(out.end(), left_types.begin(), left_types.end());
  out.insert(out.end(), right_types.begin(), right_types.end());
  return out;
}

[[nodiscard]] bool is_decimal_arithmetic_op(cudf::binary_operator op)
{
  return op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
         op == cudf::binary_operator::MUL || op == cudf::binary_operator::DIV;
}

bool contains_ast_unsupported_expr(gqe::expression const* expr,
                                   std::vector<cudf::data_type> const& column_types,
                                   bool parent_is_decimal64_arith = false)
{
  if (expr->type() == expr_type::cast) {
    auto const* cast_expr = static_cast<gqe::cast_expression const*>(expr);
    if (cudf::is_fixed_point(cast_expr->out_type())) { return true; }
  }

  bool is_decimal64_arith = false;
  if (expr->type() == expr_type::binary_op) {
    auto const* bop = static_cast<gqe::binary_op_expression const*>(expr);
    auto const op   = bop->binary_operator();
    if (is_decimal_arithmetic_op(op)) {
      auto const result_type = expr->data_type(column_types);
      if (result_type.id() == cudf::type_id::DECIMAL128) { return true; }
      if (result_type.id() == cudf::type_id::DECIMAL64) {
        if (parent_is_decimal64_arith) { return true; }
        is_decimal64_arith = true;
      }
    }
  }

  for (auto const* child : expr->children()) {
    if (contains_ast_unsupported_expr(
          child, column_types, parent_is_decimal64_arith || is_decimal64_arith)) {
      return true;
    }
  }
  return false;
}

void collect_join_predicate_operands(gqe::expression* condition,
                                     cudf::size_type num_left,
                                     std::vector<gqe::expression*>& left_operands,
                                     std::vector<gqe::expression*>& right_operands)
{
  if (condition->type() == expr_type::unary_op) {
    auto const* unary = static_cast<gqe::unary_op_expression const*>(condition);
    if (unary->unary_operator() == cudf::unary_operator::NOT) {
      auto children = condition->children();
      if (children.size() == 1) {
        collect_join_predicate_operands(children[0], num_left, left_operands, right_operands);
      }
    }
    return;
  }

  if (condition->type() != expr_type::binary_op) return;
  auto* binary  = static_cast<gqe::binary_op_expression*>(condition);
  auto children = condition->children();
  if (children.size() != 2) return;

  auto const op = binary->binary_operator();
  if (op == cudf::binary_operator::LOGICAL_AND || op == cudf::binary_operator::NULL_LOGICAL_AND ||
      op == cudf::binary_operator::LOGICAL_OR || op == cudf::binary_operator::NULL_LOGICAL_OR) {
    collect_join_predicate_operands(children[0], num_left, left_operands, right_operands);
    collect_join_predicate_operands(children[1], num_left, left_operands, right_operands);
    return;
  }

  auto const side0 = join_extraction_utils::classify_operand(children[0], num_left);
  auto const side1 = join_extraction_utils::classify_operand(children[1], num_left);
  if (side0 == join_extraction_utils::operand_side::left &&
      side1 == join_extraction_utils::operand_side::right) {
    left_operands.push_back(children[0]);
    right_operands.push_back(children[1]);
  } else if (side0 == join_extraction_utils::operand_side::right &&
             side1 == join_extraction_utils::operand_side::left) {
    left_operands.push_back(children[1]);
    right_operands.push_back(children[0]);
  }
}

[[nodiscard]] std::vector<gqe::expression*> dedupe_by_pointer(
  std::vector<gqe::expression*> const& exprs)
{
  std::vector<gqe::expression*> out;
  out.reserve(exprs.size());
  std::unordered_set<gqe::expression const*> seen;
  seen.reserve(exprs.size());
  for (auto* expr : exprs) {
    if (seen.insert(expr).second) { out.push_back(expr); }
  }
  return out;
}

}  // namespace

class join_predicate_extraction_into_project::apply_visitor
  : public gqe::logical::relation_visitor {
 public:
  apply_visitor(join_predicate_extraction_into_project const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);
    _rule.extract_join(join, _rule_applied);
  }

 private:
  join_predicate_extraction_into_project const& _rule;
  bool& _rule_applied;
};

void join_predicate_extraction_into_project::extract_join(gqe::logical::join_relation* join,
                                                          bool& rule_applied) const
{
  auto* condition = join->condition();
  if (condition == nullptr || gqe::contains_subquery(condition)) { return; }

  auto children         = join->children_safe();
  auto const num_left   = children[0]->num_columns();
  auto const num_right  = children[1]->num_columns();
  auto const full_types = combined_column_types(children[0].get(), children[1].get());

  std::vector<gqe::expression*> left_operands;
  std::vector<gqe::expression*> right_operands;
  collect_join_predicate_operands(condition, num_left, left_operands, right_operands);
  left_operands  = dedupe_by_pointer(left_operands);
  right_operands = dedupe_by_pointer(right_operands);

  std::vector<gqe::expression*> left_to_materialize;
  std::vector<gqe::expression*> right_to_materialize;
  left_to_materialize.reserve(left_operands.size());
  right_to_materialize.reserve(right_operands.size());

  for (auto* expr : left_operands) {
    if (contains_ast_unsupported_expr(expr, full_types)) { left_to_materialize.push_back(expr); }
  }
  for (auto* expr : right_operands) {
    if (contains_ast_unsupported_expr(expr, full_types)) { right_to_materialize.push_back(expr); }
  }
  if (left_to_materialize.empty() && right_to_materialize.empty()) return;

  auto rewrite_with_shift = [this](gqe::expression* expr, cudf::size_type shift) {
    return rewrite_expression(expr,
                              join_extraction_utils::make_column_reference_shift_modifier(shift),
                              transform_direction::DOWN);
  };

  auto const n_left_added  = static_cast<cudf::size_type>(left_to_materialize.size());
  auto const n_right_added = static_cast<cudf::size_type>(right_to_materialize.size());
  if (n_left_added > 0) {
    replace_child_at(join,
                     0,
                     join_extraction_utils::build_side_project(children[0],
                                                               num_left,
                                                               left_to_materialize,
                                                               /*shift=*/0,
                                                               rewrite_with_shift));
  }
  if (n_right_added > 0) {
    replace_child_at(join,
                     1,
                     join_extraction_utils::build_side_project(children[1],
                                                               num_right,
                                                               right_to_materialize,
                                                               /*shift=*/num_left,
                                                               rewrite_with_shift));
  }

  std::unordered_map<gqe::expression const*, cudf::size_type> operand_target;
  for (cudf::size_type i = 0; i < n_left_added; ++i) {
    operand_target[left_to_materialize[i]] = num_left + i;
  }
  for (cudf::size_type i = 0; i < n_right_added; ++i) {
    operand_target[right_to_materialize[i]] = (num_left + n_left_added) + num_right + i;
  }
  // Inserting columns on the left child shifts all original right-side references in the
  // join condition by `n_left_added`. Apply this shift before remapping materialized operands.
  if (n_left_added > 0) {
    rewrite_relation_expressions(
      join,
      [num_left, n_left_added](gqe::expression* e, std::vector<cudf::data_type> const&)
        -> std::unique_ptr<gqe::expression> {
        if (e->type() == expr_type::column_reference) {
          auto const idx = static_cast<gqe::column_reference_expression const*>(e)->column_idx();
          if (idx >= num_left) {
            return std::make_unique<gqe::column_reference_expression>(idx + n_left_added);
          }
        }
        return nullptr;
      },
      transform_direction::DOWN);
  }
  rewrite_relation_expressions(
    join, join_extraction_utils::make_operand_remapper(operand_target), transform_direction::DOWN);
  set_join_projection_indices(join,
                              join_extraction_utils::remap_projection_for_left_insertion(
                                join->projection_indices(), num_left, n_left_added));

  rule_applied = true;
}

std::shared_ptr<gqe::logical::relation> join_predicate_extraction_into_project::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
