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
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>

#include <cudf/types.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

class ExpressionEqualTest : public ::testing::Test {
 protected:
  ExpressionEqualTest()
  {
    int32_literal_one  = std::make_shared<gqe::literal_expression<int32_t>>(1);
    int32_literal_two  = std::make_shared<gqe::literal_expression<int32_t>>(2);
    int64_literal_one  = std::make_shared<gqe::literal_expression<int64_t>>(1);
    bool_literal_true  = std::make_shared<gqe::literal_expression<bool>>(true);
    bool_literal_false = std::make_shared<gqe::literal_expression<bool>>(false);

    col_0 = std::make_shared<gqe::column_reference_expression>(0);
    col_1 = std::make_shared<gqe::column_reference_expression>(1);
    col_2 = std::make_shared<gqe::column_reference_expression>(2);
    col_3 = std::make_shared<gqe::column_reference_expression>(3);
  }

  template <typename T>
  void test_binary()
  {
    auto expr1 = std::make_unique<T>(col_0, col_1);
    auto expr2 = std::make_unique<T>(col_0, col_2);
    EXPECT_FALSE(*expr1 == *expr2);
    EXPECT_EQ(*expr1, *expr1->clone());
  }

  template <typename T>
  void test_unary()
  {
    auto expr1 = std::make_unique<T>(col_0);
    auto expr2 = std::make_unique<T>(col_1);
    EXPECT_FALSE(*expr1 == *expr2);
    EXPECT_EQ(*expr1, *expr1->clone());
  }

  std::shared_ptr<gqe::literal_expression<int32_t>> int32_literal_one;
  std::shared_ptr<gqe::literal_expression<int32_t>> int32_literal_two;
  std::shared_ptr<gqe::literal_expression<int64_t>> int64_literal_one;
  std::shared_ptr<gqe::literal_expression<bool>> bool_literal_true;
  std::shared_ptr<gqe::literal_expression<bool>> bool_literal_false;

  std::shared_ptr<gqe::column_reference_expression> col_0;
  std::shared_ptr<gqe::column_reference_expression> col_1;
  std::shared_ptr<gqe::column_reference_expression> col_2;
  std::shared_ptr<gqe::column_reference_expression> col_3;

  std::vector<std::shared_ptr<gqe::expression>> base_cols_0;
  std::vector<std::shared_ptr<gqe::expression>> base_cols_1;
};

TEST_F(ExpressionEqualTest, BinaryOp)
{
  test_binary<gqe::add_expression>();
  test_binary<gqe::subtract_expression>();
  test_binary<gqe::multiply_expression>();
  test_binary<gqe::divide_expression>();
  test_binary<gqe::logical_and_expression>();
  test_binary<gqe::logical_or_expression>();
  test_binary<gqe::equal_expression>();
  test_binary<gqe::nulls_equal_expression>();
  test_binary<gqe::not_equal_expression>();
  test_binary<gqe::less_expression>();
  test_binary<gqe::greater_expression>();
  test_binary<gqe::less_equal_expression>();
  test_binary<gqe::greater_equal_expression>();
}

TEST_F(ExpressionEqualTest, Cast)
{
  auto cast_expr1 =
    std::make_unique<gqe::cast_expression>(col_0, cudf::data_type(cudf::type_id::BOOL8));
  auto cast_expr2 =
    std::make_unique<gqe::cast_expression>(col_1, cudf::data_type(cudf::type_id::BOOL8));
  auto cast_expr3 =
    std::make_unique<gqe::cast_expression>(col_1, cudf::data_type(cudf::type_id::INT16));
  EXPECT_FALSE(*cast_expr1 == *cast_expr2);  // different input
  EXPECT_FALSE(*cast_expr2 == *cast_expr3);  // different cast output type
  EXPECT_EQ(*cast_expr1, *cast_expr1->clone());
}

TEST_F(ExpressionEqualTest, ColumnReference)
{
  EXPECT_FALSE(*col_0 == *col_1);  // different index
  // TODO: add different data_types
  EXPECT_EQ(*col_0, *col_0->clone());
}

TEST_F(ExpressionEqualTest, Literal)
{
  EXPECT_FALSE(*int32_literal_one == *int32_literal_two);  // different values
  EXPECT_FALSE(*int32_literal_one == *int64_literal_one);  // different types
  EXPECT_EQ(*int32_literal_one, *int32_literal_one->clone());
}

TEST_F(ExpressionEqualTest, IfThenElse)
{
  auto ite1     = std::make_unique<gqe::if_then_else_expression>(bool_literal_true, col_0, col_1);
  auto ite1_dup = std::make_unique<gqe::if_then_else_expression>(bool_literal_true, col_0, col_1);
  auto ite2 =
    std::make_unique<gqe::if_then_else_expression>(bool_literal_true, int32_literal_one, col_1);
  auto ite3 =
    std::make_unique<gqe::if_then_else_expression>(bool_literal_true, col_0, int32_literal_one);
  auto ite4 =
    std::make_unique<gqe::if_then_else_expression>(bool_literal_false, col_0, int32_literal_one);
  EXPECT_FALSE(*ite1 == *ite2);  // different then_expr
  EXPECT_FALSE(*ite1 == *ite3);  // different else_expr
  EXPECT_FALSE(*ite3 == *ite4);  // different if_expr
  EXPECT_EQ(*ite1, *ite1_dup);
  EXPECT_EQ(*ite1, *ite1->clone());
}

TEST_F(ExpressionEqualTest, InPredicateSubquery)
{
  base_cols_0.push_back(col_0);
  base_cols_1.push_back(col_1);
  auto inpred_expr1 = std::make_unique<gqe::in_predicate_expression>(base_cols_0, 0);  // {col_0}
  auto inpred_expr2 = std::make_unique<gqe::in_predicate_expression>(base_cols_1, 0);  // {col_1}
  base_cols_0.push_back(col_2);
  auto inpred_expr3 =
    std::make_unique<gqe::in_predicate_expression>(base_cols_0, 0);  // {col_0, col_2}
  base_cols_0.back() = col_3;
  auto inpred_expr4 =
    std::make_unique<gqe::in_predicate_expression>(base_cols_0, 0);  // {col_0, col_3}
  auto inpred_expr5 = std::make_unique<gqe::in_predicate_expression>(base_cols_0, 1);
  EXPECT_FALSE(*inpred_expr1 ==
               *inpred_expr2);  // different needles (same length) {col_0} vs {col_1}
  EXPECT_FALSE(*inpred_expr1 ==
               *inpred_expr3);  // different needles (different length) {col_0} vs {col_0, col_2}
  EXPECT_FALSE(*inpred_expr3 == *inpred_expr4);  // different needles (different non-first element)
                                                 // {col_0, col_2} vs {col_0, col_3}
  EXPECT_FALSE(*inpred_expr4 == *inpred_expr5);  // different haystack_relation_index
  EXPECT_EQ(*inpred_expr1, *inpred_expr1->clone());
  EXPECT_EQ(*inpred_expr3, *inpred_expr3->clone());
}

TEST_F(ExpressionEqualTest, DatepartScalarFunction)
{
  auto datepart_expr1 = std::make_unique<gqe::datepart_expression>(
    col_0, gqe::datepart_expression::datetime_component::day);
  auto datepart_expr2 = std::make_unique<gqe::datepart_expression>(
    col_1, gqe::datepart_expression::datetime_component::day);
  auto datepart_expr3 = std::make_unique<gqe::datepart_expression>(
    col_1, gqe::datepart_expression::datetime_component::second);

  EXPECT_FALSE(*datepart_expr1 == *datepart_expr2);  // different input
  EXPECT_FALSE(*datepart_expr2 == *datepart_expr3);  // different datepart
  EXPECT_EQ(*datepart_expr1, *datepart_expr1->clone());
}

TEST_F(ExpressionEqualTest, LikeScalarFunction)
{
  auto like_expr1  = std::make_unique<gqe::like_expression>(col_0, "pattern1%", "", false);
  auto like_expr2  = std::make_unique<gqe::like_expression>(col_1, "pattern1%", "", false);
  auto like_expr3  = std::make_unique<gqe::like_expression>(col_1, "pattern2%", "", false);
  auto like_expr4  = std::make_unique<gqe::like_expression>(col_1, "pattern2%", "\\", false);
  auto ilike_expr1 = std::make_unique<gqe::like_expression>(col_0, "pattern1%", "", true);

  EXPECT_FALSE(*like_expr1 == *like_expr2);   // different input
  EXPECT_FALSE(*like_expr2 == *like_expr3);   // different pattern
  EXPECT_FALSE(*like_expr3 == *like_expr4);   // different escape character
  EXPECT_FALSE(*like_expr1 == *ilike_expr1);  // like vs ilike
  EXPECT_EQ(*like_expr1, *like_expr1->clone());
  EXPECT_EQ(*ilike_expr1, *ilike_expr1->clone());
}

TEST_F(ExpressionEqualTest, RoundScalarFunction)
{
  auto round_expr1 = std::make_unique<gqe::round_expression>(col_0, 1);
  auto round_expr2 = std::make_unique<gqe::round_expression>(col_1, 1);
  auto round_expr3 = std::make_unique<gqe::round_expression>(col_1, 10);

  EXPECT_FALSE(*round_expr1 == *round_expr2);  // different input
  EXPECT_FALSE(*round_expr2 == *round_expr3);  // different number of decimal points
  EXPECT_EQ(*round_expr1, *round_expr1->clone());
}

TEST_F(ExpressionEqualTest, SubstrScalarFunction)
{
  auto substr_expr1 = std::make_unique<gqe::substr_expression>(col_0, 0, 10);
  auto substr_expr2 = std::make_unique<gqe::substr_expression>(col_1, 0, 10);
  auto substr_expr3 = std::make_unique<gqe::substr_expression>(col_1, 5, 10);
  auto substr_expr4 = std::make_unique<gqe::substr_expression>(col_1, 5, 15);

  EXPECT_FALSE(*substr_expr1 == *substr_expr2);  // different input
  EXPECT_FALSE(*substr_expr2 == *substr_expr3);  // different start
  EXPECT_FALSE(*substr_expr3 == *substr_expr4);  // different length
  EXPECT_EQ(*substr_expr1, *substr_expr1->clone());
}

TEST_F(ExpressionEqualTest, UnaryOp) { test_unary<gqe::not_expression>(); }
