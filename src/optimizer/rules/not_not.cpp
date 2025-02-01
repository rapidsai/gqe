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
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/optimizer/rules/not_not.hpp>

#include <cudf/unary.hpp>
#include <iostream>
#include <memory>

std::shared_ptr<gqe::logical::relation> gqe::optimizer::not_not_rewrite::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  // Define how to optimize expressions
  auto expr_modifier = [&](expression* expr, std::vector<cudf::data_type> const& column_types) {
    // Look for outer `not`
    if (expr->type() == gqe::expression::expression_type::unary_op) {
      auto op = dynamic_cast<gqe::unary_op_expression*>(expr);
      assert(op->children().size() == 1);
      auto inner_expr = op->children()[0];
      if (op->unary_operator() == cudf::unary_operator::NOT) {
        // Look for inner `not`
        if (inner_expr->type() == gqe::expression::expression_type::unary_op) {
          auto inner_op = dynamic_cast<gqe::unary_op_expression*>(inner_expr);
          assert(inner_op->children().size() == 1);
          if (inner_op->unary_operator() == cudf::unary_operator::NOT) {
            // Found not(not(expression)) pattern
            rule_applied = true;
            return inner_op->children()[0]->clone();
          }
        }
      }
    }
    // Pattern not found
    return static_cast<std::unique_ptr<gqe::expression>>(nullptr);
  };

  // Note that transform_direction::UP would also work in this case
  rewrite_relation_expressions(logical_relation.get(), expr_modifier, transform_direction::DOWN);
  return logical_relation;
}
