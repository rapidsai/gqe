/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/rules/join_children_swap.hpp>
#include <gqe/optimizer/utility.hpp>

#include <cudf/types.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
bool is_valid_join_condition(gqe::expression* condition, cudf::size_type n_cols_left, bool is_left)
{
  if (condition->type() == gqe::expression::expression_type::binary_op) {
    auto binary_expr = dynamic_cast<gqe::binary_op_expression*>(condition);
    auto children    = binary_expr->children();
    assert(children.size() == 2);
    if (binary_expr->binary_operator() == cudf::binary_operator::EQUAL) {
      // Check left and right
      return is_valid_join_condition(children[0], n_cols_left, true) &&
             is_valid_join_condition(children[1], n_cols_left, false);
    }
  } else if (condition->type() == gqe::expression::expression_type::column_reference) {
    auto col_ref = dynamic_cast<gqe::column_reference_expression*>(condition);
    auto col_idx = col_ref->column_idx();
    if (is_left && (col_idx >= n_cols_left)) return false;
    if (!is_left && (col_idx < n_cols_left)) return false;
  }
  return true;
}
}  // namespace
using relation_t = gqe::logical::relation::relation_type;

std::shared_ptr<gqe::logical::relation> gqe::optimizer::join_children_swap::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  if (logical_relation->type() != relation_t::join) return logical_relation;
  // Extract predicate
  auto join = dynamic_cast<logical::join_relation*>(logical_relation.get());
  // If the join type is inner join, we broadcast the smaller table.
  if (join->join_type() == join_type_type::inner) {
    assert(join->children_size() == 2);
    auto condition = join->condition();
    if (condition->type() == expression::expression_type::binary_op) {
      auto children             = join->children_safe();
      auto estimator            = this->get_estimator();
      auto const left_num_rows  = estimator(children[0].get()).num_rows;
      auto const right_num_rows = estimator(children[1].get()).num_rows;
      if ((default_broadcast_policy() == physical::broadcast_policy::left &&
           right_num_rows < left_num_rows) ||
          (default_broadcast_policy() == physical::broadcast_policy::right &&
           left_num_rows < right_num_rows)) {
        // Get join keys
        // Swap children
        auto child_0 = children[0];
        auto child_1 = children[1];
        replace_child_at(join, 0, child_1);
        replace_child_at(join, 1, child_0);
        // Rewrite join condition
        swap_join_keys_inplace(join, child_0->num_columns(), child_1->num_columns());
        // Update swapped projection indices
        optimizer::utility::swap_projection_indices_inplace(
          join->_projection_indices, child_0->num_columns(), child_1->num_columns());
        // Update that rule has been applied
        rule_applied = true;
      }
    }
  }
  return logical_relation;
}

void gqe::optimizer::join_children_swap::swap_join_keys_inplace(gqe::logical::join_relation* join,
                                                                cudf::size_type n_cols_left,
                                                                cudf::size_type n_cols_right) const
{
  // Check if both sides of equal operations have appropriate column references
  // So the left and right side can be parsed as keys appropriately
  if (!is_valid_join_condition(join->condition(), n_cols_left, true))
    throw std::runtime_error("join_children_swap: invalid join condition");

  // Add offset
  auto colref_modifier =
    [=](gqe::expression* expr,
        std::vector<cudf::data_type> const& column_types) -> std::unique_ptr<gqe::expression> {
    // Look for column reference
    if (expr->type() == gqe::expression::expression_type::column_reference) {
      auto colref = dynamic_cast<gqe::column_reference_expression*>(expr);
      // Check value and adjust index
      auto idx = colref->column_idx();
      if (idx < n_cols_left) {
        return std::make_unique<gqe::column_reference_expression>(idx + n_cols_right);
      } else {
        return std::make_unique<gqe::column_reference_expression>(idx - n_cols_left);
      }
    }
    // Not column reference
    return nullptr;
  };
  rewrite_relation_expressions(join, colref_modifier, transform_direction::DOWN);

  // Switch children of each equal operation
  auto eq_expr_children_swap =
    [](gqe::expression* expr,
       std::vector<cudf::data_type> const& column_types) -> std::unique_ptr<gqe::expression> {
    std::cout << expr->to_string() << std::endl;
    // Look for equal binary expression
    if (expr->type() == gqe::expression::expression_type::binary_op) {
      auto binary_expr = dynamic_cast<gqe::binary_op_expression*>(expr);
      auto children    = binary_expr->children();
      assert(children.size() == 2);
      if (binary_expr->binary_operator() == cudf::binary_operator::EQUAL ||
          binary_expr->binary_operator() == cudf::binary_operator::NULL_EQUALS) {
        // Swap children
        return std::make_unique<gqe::equal_expression>(children[1]->clone(), children[0]->clone());
      }
    }
    // Not an equal binary expression
    return nullptr;
  };
  rewrite_relation_expressions(join, eq_expr_children_swap, transform_direction::UP);
}
