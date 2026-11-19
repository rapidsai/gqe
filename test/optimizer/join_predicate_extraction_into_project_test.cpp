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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <memory>
#include <utility>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;
using expr_t     = gqe::expression::expression_type;
using rule_t     = gqe::optimizer::logical_optimization_rule_type;

class JoinPredicateExtractionIntoProjectTest : public ::testing::Test {
 protected:
  static constexpr const char* kLeftTable  = "left_t";
  static constexpr const char* kRightTable = "right_t";

  JoinPredicateExtractionIntoProjectTest()
  {
    task_manager_ctx_ = std::make_unique<gqe::task_manager_context>();
    catalog_          = std::make_unique<gqe::catalog>(task_manager_ctx_.get());

    auto const dec = cudf::data_type(cudf::type_id::DECIMAL64, -2);
    catalog_->register_table(kLeftTable,
                             {{"k", cudf::data_type(cudf::type_id::INT32)}, {"x", dec}, {"y", dec}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});
    catalog_->register_table(kRightTable,
                             {{"k2", cudf::data_type(cudf::type_id::INT32)}, {"v", dec}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});

    config_ = gqe::optimizer::optimization_configuration(
      {rule_t::join_predicate_extraction_into_project}, {});
    optimizer_ = std::make_unique<gqe::optimizer::logical_optimizer>(&config_, catalog_.get());
  }

  std::shared_ptr<gqe::logical::read_relation> make_left_read()
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    auto const dec = cudf::data_type(cudf::type_id::DECIMAL64, -2);
    return std::make_shared<gqe::logical::read_relation>(
      sq,
      std::vector<std::string>{"k", "x", "y"},
      std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT32), dec, dec},
      kLeftTable,
      nullptr);
  }

  std::shared_ptr<gqe::logical::read_relation> make_right_read()
  {
    std::vector<std::shared_ptr<gqe::logical::relation>> sq;
    auto const dec = cudf::data_type(cudf::type_id::DECIMAL64, -2);
    return std::make_shared<gqe::logical::read_relation>(
      sq,
      std::vector<std::string>{"k2", "v"},
      std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT32), dec},
      kRightTable,
      nullptr);
  }

  static std::shared_ptr<gqe::expression> col(cudf::size_type idx)
  {
    return std::make_shared<gqe::column_reference_expression>(idx);
  }

  static cudf::size_type operand_col_idx(gqe::expression const* expr)
  {
    auto const* cr = dynamic_cast<gqe::column_reference_expression const*>(expr);
    EXPECT_NE(cr, nullptr);
    return cr ? cr->column_idx() : -1;
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx_;
  std::unique_ptr<gqe::catalog> catalog_;
  gqe::optimizer::optimization_configuration config_;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer_;
};

TEST_F(JoinPredicateExtractionIntoProjectTest, MaterializesFixedPointCastOperand)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq = std::make_shared<gqe::equal_expression>(col(0), col(3));
  auto cast_left_x =
    std::make_shared<gqe::cast_expression>(col(1), cudf::data_type(cudf::type_id::DECIMAL64, -3));
  auto lt        = std::make_shared<gqe::less_expression>(cast_left_x, col(4));
  auto condition = std::make_unique<gqe::logical_and_expression>(eq, lt);

  auto join = std::make_shared<gqe::logical::join_relation>(make_left_read(),
                                                            make_right_read(),
                                                            sq,
                                                            std::move(condition),
                                                            gqe::join_type_type::inner,
                                                            std::vector<cudf::size_type>{0, 1, 4});

  auto expected_types = join->data_types();
  auto optimized      = optimizer_->optimize(join);
  ASSERT_EQ(optimized->type(), relation_t::join);
  auto* out = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  ASSERT_EQ(left->output_expressions_unsafe().size(), 4u);
  EXPECT_EQ(left->output_expressions_unsafe()[3]->type(), expr_t::cast);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);

  auto* and_expr = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(and_expr, nullptr);
  ASSERT_TRUE(and_expr->binary_operator() == cudf::binary_operator::LOGICAL_AND ||
              and_expr->binary_operator() == cudf::binary_operator::NULL_LOGICAL_AND);
  auto* rewritten_lt = dynamic_cast<gqe::binary_op_expression const*>(and_expr->children()[1]);
  ASSERT_NE(rewritten_lt, nullptr);
  ASSERT_EQ(rewritten_lt->binary_operator(), cudf::binary_operator::LESS);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[0]), 3);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[1]), 5);

  EXPECT_EQ(out->projection_indices(), (std::vector<cudf::size_type>{0, 1, 5}));
  EXPECT_EQ(out->data_types(), expected_types);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::join_predicate_extraction_into_project), 1u);
}

TEST_F(JoinPredicateExtractionIntoProjectTest, SkipsRepresentablePredicateOperands)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq        = std::make_shared<gqe::equal_expression>(col(0), col(3));
  auto lt        = std::make_shared<gqe::less_expression>(col(1), col(4));
  auto condition = std::make_unique<gqe::logical_and_expression>(eq, lt);

  auto join = std::make_shared<gqe::logical::join_relation>(make_left_read(),
                                                            make_right_read(),
                                                            sq,
                                                            std::move(condition),
                                                            gqe::join_type_type::inner,
                                                            std::vector<cudf::size_type>{0, 1, 4});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(out->children_safe()[1].get()), nullptr);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::join_predicate_extraction_into_project), 0u);
}

TEST_F(JoinPredicateExtractionIntoProjectTest, MaterializesUnsupportedOperandUnderOr)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq = std::make_shared<gqe::equal_expression>(col(0), col(3));
  auto cast_left_x =
    std::make_shared<gqe::cast_expression>(col(1), cudf::data_type(cudf::type_id::DECIMAL64, -3));
  auto lt        = std::make_shared<gqe::less_expression>(cast_left_x, col(4));
  auto condition = std::make_unique<gqe::logical_or_expression>(eq, lt);

  auto join = std::make_shared<gqe::logical::join_relation>(make_left_read(),
                                                            make_right_read(),
                                                            sq,
                                                            std::move(condition),
                                                            gqe::join_type_type::inner,
                                                            std::vector<cudf::size_type>{0, 1, 4});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  ASSERT_EQ(left->output_expressions_unsafe().size(), 4u);
  EXPECT_EQ(left->output_expressions_unsafe()[3]->type(), expr_t::cast);

  auto* or_expr = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(or_expr, nullptr);
  ASSERT_TRUE(or_expr->binary_operator() == cudf::binary_operator::LOGICAL_OR ||
              or_expr->binary_operator() == cudf::binary_operator::NULL_LOGICAL_OR);
  auto* rewritten_lt = dynamic_cast<gqe::binary_op_expression const*>(or_expr->children()[1]);
  ASSERT_NE(rewritten_lt, nullptr);
  ASSERT_EQ(rewritten_lt->binary_operator(), cudf::binary_operator::LESS);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[0]), 3);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[1]), 5);

  EXPECT_EQ(optimizer_->get_rule_count(rule_t::join_predicate_extraction_into_project), 1u);
}

TEST_F(JoinPredicateExtractionIntoProjectTest, MaterializesUnsupportedOperandUnderNot)
{
  std::vector<std::shared_ptr<gqe::logical::relation>> sq;
  auto eq = std::make_shared<gqe::equal_expression>(col(0), col(3));
  auto cast_left_x =
    std::make_shared<gqe::cast_expression>(col(1), cudf::data_type(cudf::type_id::DECIMAL64, -3));
  auto lt        = std::make_shared<gqe::less_expression>(cast_left_x, col(4));
  auto not_lt    = std::make_shared<gqe::not_expression>(lt);
  auto condition = std::make_unique<gqe::logical_and_expression>(eq, not_lt);

  auto join = std::make_shared<gqe::logical::join_relation>(make_left_read(),
                                                            make_right_read(),
                                                            sq,
                                                            std::move(condition),
                                                            gqe::join_type_type::inner,
                                                            std::vector<cudf::size_type>{0, 1, 4});

  auto optimized = optimizer_->optimize(join);
  auto* out      = static_cast<gqe::logical::join_relation*>(optimized.get());

  auto* left = dynamic_cast<gqe::logical::project_relation*>(out->children_safe()[0].get());
  ASSERT_NE(left, nullptr);
  ASSERT_EQ(left->output_expressions_unsafe().size(), 4u);
  EXPECT_EQ(left->output_expressions_unsafe()[3]->type(), expr_t::cast);

  auto* and_expr = dynamic_cast<gqe::binary_op_expression const*>(out->condition());
  ASSERT_NE(and_expr, nullptr);
  ASSERT_TRUE(and_expr->binary_operator() == cudf::binary_operator::LOGICAL_AND ||
              and_expr->binary_operator() == cudf::binary_operator::NULL_LOGICAL_AND);
  auto* rewritten_not = and_expr->children()[1];
  ASSERT_NE(rewritten_not, nullptr);
  ASSERT_EQ(rewritten_not->type(), expr_t::unary_op);

  auto not_children = rewritten_not->children();
  ASSERT_EQ(not_children.size(), 1u);
  auto* rewritten_lt = dynamic_cast<gqe::binary_op_expression const*>(not_children[0]);
  ASSERT_NE(rewritten_lt, nullptr);
  ASSERT_EQ(rewritten_lt->binary_operator(), cudf::binary_operator::LESS);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[0]), 3);
  EXPECT_EQ(operand_col_idx(rewritten_lt->children()[1]), 5);

  EXPECT_EQ(optimizer_->get_rule_count(rule_t::join_predicate_extraction_into_project), 1u);
}
