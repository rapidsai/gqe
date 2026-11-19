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

using expr_t     = gqe::expression::expression_type;
using relation_t = gqe::logical::relation::relation_type;
using rule_t     = gqe::optimizer::logical_optimization_rule_type;

class NormalizeDecimalWidthsTest : public ::testing::Test {
 protected:
  static std::shared_ptr<gqe::column_reference_expression> col(cudf::size_type idx)
  {
    return std::make_shared<gqe::column_reference_expression>(idx);
  }

  std::shared_ptr<gqe::logical::read_relation> make_decimal_read(
    std::vector<cudf::data_type> const& types)
  {
    std::vector<std::string> names;
    names.reserve(types.size());
    for (std::size_t i = 0; i < types.size(); ++i) {
      names.push_back("c" + std::to_string(i));
    }
    return std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>{}, names, types, "t", nullptr);
  }

  NormalizeDecimalWidthsTest()
  {
    task_manager_ctx_ = std::make_unique<gqe::task_manager_context>();
    catalog_          = std::make_unique<gqe::catalog>(task_manager_ctx_.get());
    config_    = gqe::optimizer::optimization_configuration({rule_t::normalize_decimal_widths}, {});
    optimizer_ = std::make_unique<gqe::optimizer::logical_optimizer>(&config_, catalog_.get());
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx_;
  std::unique_ptr<gqe::catalog> catalog_;
  gqe::optimizer::optimization_configuration config_;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer_;
};

TEST_F(NormalizeDecimalWidthsTest, JoinConditionCastsNarrowerOperand)
{
  auto left  = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL32, -2}});
  auto right = make_decimal_read(
    {cudf::data_type{cudf::type_id::DECIMAL64, -2}, cudf::data_type{cudf::type_id::INT32}});

  auto condition = std::make_unique<gqe::equal_expression>(col(0), col(1));
  auto join      = std::make_shared<gqe::logical::join_relation>(
    left,
    right,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>{0, 1, 2});

  auto optimized = optimizer_->optimize(join);
  ASSERT_GE(optimizer_->get_rule_count(rule_t::normalize_decimal_widths), 1u);

  auto* opt_join = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(opt_join, nullptr);
  // The rule now normalizes decimal width directly in the join condition by wrapping
  // the narrower operand with a cast, instead of injecting a pre-join project.
  EXPECT_EQ(opt_join->children_safe()[0]->type(), relation_t::read);
  auto* lhs = opt_join->condition()->children()[0];
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->type(), expr_t::cast);
  auto const* lhs_cast = dynamic_cast<gqe::cast_expression const*>(lhs);
  ASSERT_NE(lhs_cast, nullptr);
  EXPECT_EQ(lhs_cast->out_type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(lhs_cast->out_type().scale(), -2);
}

TEST_F(NormalizeDecimalWidthsTest, JoinConditionCastsNarrowerRhsOperand)
{
  auto left  = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL128, -2}});
  auto right = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL64, -3}});

  auto condition = std::make_unique<gqe::greater_expression>(col(0), col(1));
  auto join      = std::make_shared<gqe::logical::join_relation>(
    left,
    right,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>{0, 1});

  auto optimized = optimizer_->optimize(join);
  ASSERT_GE(optimizer_->get_rule_count(rule_t::normalize_decimal_widths), 1u);

  auto* opt_join = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(opt_join, nullptr);
  auto* lhs = opt_join->condition()->children()[0];
  auto* rhs = opt_join->condition()->children()[1];
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  EXPECT_NE(lhs->type(), expr_t::cast);
  EXPECT_EQ(rhs->type(), expr_t::cast);
  auto const* rhs_cast = dynamic_cast<gqe::cast_expression const*>(rhs);
  ASSERT_NE(rhs_cast, nullptr);
  EXPECT_EQ(rhs_cast->out_type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(rhs_cast->out_type().scale(), -3);
}

TEST_F(NormalizeDecimalWidthsTest, NoOpWhenWidthsAlreadyMatch)
{
  auto left  = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL64, -2}});
  auto right = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL64, -3}});

  auto condition = std::make_unique<gqe::equal_expression>(col(0), col(1));
  auto join      = std::make_shared<gqe::logical::join_relation>(
    left,
    right,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>{0, 1});

  optimizer_->optimize(join);
  EXPECT_EQ(optimizer_->get_rule_count(rule_t::normalize_decimal_widths), 0u);
}

TEST_F(NormalizeDecimalWidthsTest, NormalizesNestedComparisonUnderNotWrapper)
{
  auto left  = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL32, -2}});
  auto right = make_decimal_read({cudf::data_type{cudf::type_id::DECIMAL64, -2}});

  auto nested_eq = std::make_shared<gqe::equal_expression>(col(0), col(1));
  auto condition = std::make_unique<gqe::not_expression>(nested_eq);
  auto join      = std::make_shared<gqe::logical::join_relation>(
    left,
    right,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>{0, 1});

  auto optimized = optimizer_->optimize(join);
  ASSERT_GE(optimizer_->get_rule_count(rule_t::normalize_decimal_widths), 1u);

  auto* opt_join = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(opt_join, nullptr);
  ASSERT_NE(opt_join->condition(), nullptr);
  EXPECT_EQ(opt_join->condition()->type(), expr_t::unary_op);

  auto const not_children = opt_join->condition()->children();
  ASSERT_EQ(not_children.size(), 1u);
  auto const* nested_cmp = not_children[0];
  ASSERT_NE(nested_cmp, nullptr);
  ASSERT_EQ(nested_cmp->type(), expr_t::binary_op);

  auto const cmp_children = nested_cmp->children();
  ASSERT_EQ(cmp_children.size(), 2u);
  auto const* lhs = cmp_children[0];
  ASSERT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->type(), expr_t::cast);
  auto const* lhs_cast = dynamic_cast<gqe::cast_expression const*>(lhs);
  ASSERT_NE(lhs_cast, nullptr);
  EXPECT_EQ(lhs_cast->out_type().id(), cudf::type_id::DECIMAL64);
  EXPECT_EQ(lhs_cast->out_type().scale(), -2);
}
