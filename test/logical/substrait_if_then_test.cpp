/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/task_manager_context.hpp>

#include <gtest/gtest.h>

#include <substrait/algebra.pb.h>
#include <substrait/plan.pb.h>

#include <memory>
#include <vector>

namespace {

/**
 * @brief Helper function to create a simple Substrait literal expression
 */
substrait::Expression create_literal_expression(int32_t value)
{
  substrait::Expression expr;
  auto* literal = expr.mutable_literal();
  literal->set_i32(value);
  return expr;
}

/**
 * @brief Helper function to create a Substrait field reference expression
 */
substrait::Expression create_field_reference_expression(int field_index)
{
  substrait::Expression expr;
  auto* selection  = expr.mutable_selection();
  auto* direct_ref = selection->mutable_direct_reference();
  direct_ref->mutable_struct_field()->set_field(field_index);
  selection->mutable_root_reference();
  return expr;
}

/**
 * @brief Helper function to create a Substrait boolean literal expression
 */
substrait::Expression create_boolean_literal_expression(bool value)
{
  substrait::Expression expr;
  auto* literal = expr.mutable_literal();
  literal->set_boolean(value);
  return expr;
}

/**
 * @brief Helper function to create a Substrait IfThen expression with multiple WHEN clauses
 */
substrait::Expression create_if_then_expression(
  std::vector<std::pair<substrait::Expression, substrait::Expression>> when_clauses,
  substrait::Expression else_expr)
{
  substrait::Expression expr;
  auto* if_then = expr.mutable_if_then();

  for (auto& [condition, value] : when_clauses) {
    auto* if_clause            = if_then->add_ifs();
    *if_clause->mutable_if_()  = std::move(condition);
    *if_clause->mutable_then() = std::move(value);
  }

  *if_then->mutable_else_() = std::move(else_expr);
  return expr;
}

}  // namespace

class SubstraitIfThenTest : public ::testing::Test {
 protected:
  SubstraitIfThenTest()
  {
    task_manager_ctx = std::make_unique<gqe::task_manager_context>();
    catalog          = std::make_unique<gqe::catalog>(task_manager_ctx.get());
    parser           = std::make_unique<gqe::substrait_parser>(catalog.get());
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx;
  std::unique_ptr<gqe::catalog> catalog;
  std::unique_ptr<gqe::substrait_parser> parser;
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
};

/**
 * @brief Test parsing a simple IfThen expression with a single WHEN clause
 */
TEST_F(SubstraitIfThenTest, SingleWhenClause)
{
  // Create: CASE WHEN true THEN 10 ELSE 20
  auto condition  = create_boolean_literal_expression(true);
  auto then_value = create_literal_expression(10);
  auto else_value = create_literal_expression(20);

  std::vector<std::pair<substrait::Expression, substrait::Expression>> when_clauses;
  when_clauses.emplace_back(std::move(condition), std::move(then_value));

  auto substrait_expr = create_if_then_expression(std::move(when_clauses), std::move(else_value));

  // Parse the expression
  auto parsed_expr = parser->parse_expression(substrait_expr, subquery_relations);

  // Verify it's an if_then_else_expression
  ASSERT_NE(parsed_expr, nullptr);
  auto* ite_expr = dynamic_cast<gqe::if_then_else_expression*>(parsed_expr.get());
  ASSERT_NE(ite_expr, nullptr);

  // Verify structure: IF (true) THEN 10 ELSE 20
  auto children = ite_expr->children();
  ASSERT_EQ(children.size(), 3);
  // The condition, then, and else expressions should be present
}

/**
 * @brief Test parsing an IfThen expression with two WHEN clauses
 */
TEST_F(SubstraitIfThenTest, TwoWhenClauses)
{
  // Create: CASE WHEN true THEN 10 WHEN false THEN 20 ELSE 30
  auto condition1  = create_boolean_literal_expression(true);
  auto then_value1 = create_literal_expression(10);

  auto condition2  = create_boolean_literal_expression(false);
  auto then_value2 = create_literal_expression(20);

  auto else_value = create_literal_expression(30);

  std::vector<std::pair<substrait::Expression, substrait::Expression>> when_clauses;
  when_clauses.emplace_back(std::move(condition1), std::move(then_value1));
  when_clauses.emplace_back(std::move(condition2), std::move(then_value2));

  auto substrait_expr = create_if_then_expression(std::move(when_clauses), std::move(else_value));

  // Parse the expression
  auto parsed_expr = parser->parse_expression(substrait_expr, subquery_relations);

  // Verify it's an if_then_else_expression (nested structure)
  ASSERT_NE(parsed_expr, nullptr);
  auto* outer_ite = dynamic_cast<gqe::if_then_else_expression*>(parsed_expr.get());
  ASSERT_NE(outer_ite, nullptr);

  // Verify nested structure: IF (true) THEN 10 ELSE (IF (false) THEN 20 ELSE 30)
  auto children = outer_ite->children();
  ASSERT_EQ(children.size(), 3);

  // The else branch should be another if_then_else_expression
  auto* inner_ite = dynamic_cast<gqe::if_then_else_expression*>(children[2]);
  ASSERT_NE(inner_ite, nullptr);
  ASSERT_EQ(inner_ite->children().size(), 3);
}

/**
 * @brief Test parsing an IfThen expression with three WHEN clauses
 */
TEST_F(SubstraitIfThenTest, ThreeWhenClauses)
{
  // Create: CASE WHEN true THEN 10 WHEN false THEN 20 WHEN true THEN 25 ELSE 30
  auto condition1  = create_boolean_literal_expression(true);
  auto then_value1 = create_literal_expression(10);

  auto condition2  = create_boolean_literal_expression(false);
  auto then_value2 = create_literal_expression(20);

  auto condition3  = create_boolean_literal_expression(true);
  auto then_value3 = create_literal_expression(25);

  auto else_value = create_literal_expression(30);

  std::vector<std::pair<substrait::Expression, substrait::Expression>> when_clauses;
  when_clauses.emplace_back(std::move(condition1), std::move(then_value1));
  when_clauses.emplace_back(std::move(condition2), std::move(then_value2));
  when_clauses.emplace_back(std::move(condition3), std::move(then_value3));

  auto substrait_expr = create_if_then_expression(std::move(when_clauses), std::move(else_value));

  // Parse the expression
  auto parsed_expr = parser->parse_expression(substrait_expr, subquery_relations);

  // Verify it's an if_then_else_expression (nested structure)
  ASSERT_NE(parsed_expr, nullptr);
  auto* level1_ite = dynamic_cast<gqe::if_then_else_expression*>(parsed_expr.get());
  ASSERT_NE(level1_ite, nullptr);

  // Verify nested structure with 3 levels
  // Level 1: IF (true) THEN 10 ELSE (level2)
  auto level1_children = level1_ite->children();
  ASSERT_EQ(level1_children.size(), 3);

  // Level 2: IF (false) THEN 20 ELSE (level3)
  auto* level2_ite = dynamic_cast<gqe::if_then_else_expression*>(level1_children[2]);
  ASSERT_NE(level2_ite, nullptr);
  auto level2_children = level2_ite->children();
  ASSERT_EQ(level2_children.size(), 3);

  // Level 3: IF (true) THEN 25 ELSE 30
  auto* level3_ite = dynamic_cast<gqe::if_then_else_expression*>(level2_children[2]);
  ASSERT_NE(level3_ite, nullptr);
  auto level3_children = level3_ite->children();
  ASSERT_EQ(level3_children.size(), 3);
}

/**
 * @brief Test that parsing fails when IfThen expression has no WHEN clauses
 */
TEST_F(SubstraitIfThenTest, NoWhenClausesThrows)
{
  substrait::Expression expr;
  auto* if_then = expr.mutable_if_then();
  // No ifs added
  *if_then->mutable_else_() = create_literal_expression(0);

  // Should throw an error
  EXPECT_THROW(parser->parse_expression(expr, subquery_relations), std::runtime_error);
}

/**
 * @brief Test parsing with multiple WHEN clauses using different column references
 */
TEST_F(SubstraitIfThenTest, MultipleWhenClausesDifferentColumns)
{
  // Create: CASE WHEN true THEN col_1 WHEN false THEN col_3 ELSE col_0
  auto condition1  = create_boolean_literal_expression(true);
  auto then_value1 = create_field_reference_expression(1);

  auto condition2  = create_boolean_literal_expression(false);
  auto then_value2 = create_field_reference_expression(3);

  auto else_value = create_field_reference_expression(0);

  std::vector<std::pair<substrait::Expression, substrait::Expression>> when_clauses;
  when_clauses.emplace_back(std::move(condition1), std::move(then_value1));
  when_clauses.emplace_back(std::move(condition2), std::move(then_value2));

  auto substrait_expr = create_if_then_expression(std::move(when_clauses), std::move(else_value));

  // Parse the expression
  auto parsed_expr = parser->parse_expression(substrait_expr, subquery_relations);

  // Verify it parses successfully
  ASSERT_NE(parsed_expr, nullptr);
  auto* outer_ite = dynamic_cast<gqe::if_then_else_expression*>(parsed_expr.get());
  ASSERT_NE(outer_ite, nullptr);

  // Verify nested structure
  auto children = outer_ite->children();
  ASSERT_EQ(children.size(), 3);

  auto* inner_ite = dynamic_cast<gqe::if_then_else_expression*>(children[2]);
  ASSERT_NE(inner_ite, nullptr);
}
