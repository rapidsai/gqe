/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/eval.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <memory>
#include <string>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

TEST(EvalExpressionsTest, ColumnReference)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0                                  = gqe::column_reference_expression(0);
  std::vector<gqe::expression const*> expressions = {&col_ref_0};

  auto const& expected                   = c_0;
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerColumnIntegerColumnEquality)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerColumnIntegerColumnEqualityMixedTypes)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int64_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, StringColumnStringColumnEquality)
{
  auto c_0   = cudf::test::strings_column_wrapper{"asdf", "qwer"};
  auto c_1   = cudf::test::strings_column_wrapper{"yxcv", "qwer"};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{false, true};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerColumnIntegerLiteralEquality)
{
  auto c_0   = column_wrapper<int32_t>{42, 20, 42, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto lit       = std::make_shared<gqe::literal_expression<int32_t>>(42);

  auto eq                                         = gqe::equal_expression(col_ref_0, lit);
  std::vector<gqe::expression const*> expressions = {&eq};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerColumnIntegerLiteralEqualityMixedTypes)
{
  auto c_0   = column_wrapper<int32_t>{42, 20, 42, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto lit       = std::make_shared<gqe::literal_expression<int64_t>>(42);

  auto eq                                         = gqe::equal_expression(col_ref_0, lit);
  std::vector<gqe::expression const*> expressions = {&eq};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, StringColumnStringLiteralEquality)
{
  auto c_0   = cudf::test::strings_column_wrapper{"A", "B", "A", "C"};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto lit       = std::make_shared<gqe::literal_expression<std::string>>("A");

  auto eq                                         = gqe::equal_expression(col_ref_0, lit);
  std::vector<gqe::expression const*> expressions = {&eq};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, HeterogeneousEvaluationStrategy)
{
  auto c_0   = cudf::test::strings_column_wrapper{"A", "B", "A", "C"};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);
  auto lit0      = std::make_shared<gqe::literal_expression<std::string>>("A");
  auto lit1      = std::make_shared<gqe::literal_expression<int32_t>>(1);

  auto eq0  = std::make_shared<gqe::equal_expression>(col_ref_0, lit0);
  auto eq1  = std::make_shared<gqe::equal_expression>(col_ref_1, lit1);
  auto root = gqe::logical_and_expression(eq0, eq1);
  std::vector<gqe::expression const*> expressions = {&root};

  auto expected                          = column_wrapper<bool>{false, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerAddition)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<int64_t>{5, 4, 3, 3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto add_expr = gqe::add_expression(std::make_shared<gqe::column_reference_expression>(0),
                                      std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&add_expr};

  auto expected                          = column_wrapper<int64_t>{7, 9, 6, 9};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, FloatAddition)
{
  auto c_0   = column_wrapper<int64_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<float>{5.0, 4.0, 3.0, 3.0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto add_expr = gqe::add_expression(std::make_shared<gqe::column_reference_expression>(0),
                                      std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&add_expr};

  auto expected                          = column_wrapper<double>{7.0, 9.0, 6.0, 9.0};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerSubtraction)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<uint32_t>{5, 4, 3, 3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto subtract_expr =
    gqe::subtract_expression(std::make_shared<gqe::column_reference_expression>(0),
                             std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&subtract_expr};

  auto expected                          = column_wrapper<int64_t>{-3, 1, 0, 3};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerMultiplication)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<int32_t>{5, 4, 3, 3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto multiply_expr =
    gqe::multiply_expression(std::make_shared<gqe::column_reference_expression>(0),
                             std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&multiply_expr};

  auto expected                          = column_wrapper<int64_t>{10, 20, 9, 18};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, IntegerDivision)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<int64_t>{5, 4, 3, 3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto divide_expr = gqe::divide_expression(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&divide_expr};

  auto expected                          = column_wrapper<double>{0.4, 1.25, 1.0, 2.0};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, FloatDivision)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<float>{5.0, 4.0, 3.0, 3.0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto divide_expr = gqe::divide_expression(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&divide_expr};

  auto expected                          = column_wrapper<double>{0.4, 1.25, 1.0, 2.0};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, SimpleConditional)
{
  auto c_0   = column_wrapper<bool>{true, false, false, true};
  auto c_1   = column_wrapper<int32_t>{42, 43, 44, 45};
  auto c_2   = column_wrapper<int32_t>{52, 53, 54, 55};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto cond_expr =
    gqe::if_then_else_expression(std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(1),
                                 std::make_shared<gqe::column_reference_expression>(2));
  std::vector<gqe::expression const*> expressions = {&cond_expr};

  auto expected                          = column_wrapper<int32_t>{42, 53, 54, 45};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, ComplexConditional)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 1, 9};
  auto c_1   = column_wrapper<int32_t>{42, 43, 44, 45};
  auto c_2   = column_wrapper<int32_t>{52, 53, 54, 55};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto ref_1 = std::make_shared<gqe::column_reference_expression>(1);
  auto ref_2 = std::make_shared<gqe::column_reference_expression>(2);

  auto if_expr = std::make_shared<gqe::greater_expression>(
    ref_0, std::make_shared<gqe::literal_expression<int32_t>>(3));
  auto then_expr = std::make_shared<gqe::add_expression>(ref_0, ref_1);
  auto else_expr = std::make_shared<gqe::add_expression>(ref_0, ref_2);

  // IF (ref_0 > 3) THEN return ref_0 + ref_1 ELSE return ref_0 + ref_2 FI
  auto cond_expr = gqe::if_then_else_expression(if_expr, then_expr, else_expr);
  std::vector<gqe::expression const*> expressions = {&cond_expr};

  auto expected                          = column_wrapper<int64_t>{54, 48, 55, 54};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, NumericalLiteral)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<float>{5.0, 4.0, 3.0, 3.0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto literal_expr                               = gqe::literal_expression<int32_t>(1);
  auto literal_expr_null                          = gqe::literal_expression<int32_t>(1, true);
  std::vector<gqe::expression const*> expressions = {&literal_expr, &literal_expr_null};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto expected = column_wrapper<int32_t>{1, 1, 1, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);

  auto expected_null = column_wrapper<int32_t>{{1, 1, 1, 1}, {false, false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, evaluated_results[1]);
}

TEST(EvalExpressionsTest, StringLiteral)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<float>{5.0, 4.0, 3.0, 3.0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto literal_expr      = gqe::literal_expression<std::string>("apple");
  auto literal_expr_null = gqe::literal_expression<std::string>("apple", true);
  std::vector<gqe::expression const*> expressions = {&literal_expr, &literal_expr_null};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto expected = cudf::test::strings_column_wrapper{"apple", "apple", "apple", "apple"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);

  auto expected_null = cudf::test::strings_column_wrapper{{"apple", "apple", "apple", "apple"},
                                                          {false, false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, evaluated_results[1]);
}

TEST(EvalExpressionsTest, Cast)
{
  auto c_0   = column_wrapper<int32_t>{2, 5, 3, 6};
  auto c_1   = column_wrapper<float>{5.0, 4.0, 3.0, 3.0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto cast_expr_0 = gqe::cast_expression(std::make_shared<gqe::column_reference_expression>(0),
                                          cudf::data_type(cudf::type_id::INT64));
  auto cast_expr_1 = gqe::cast_expression(std::make_shared<gqe::column_reference_expression>(1),
                                          cudf::data_type(cudf::type_id::FLOAT64));
  auto cast_expr_2 = gqe::cast_expression(std::make_shared<gqe::column_reference_expression>(0),
                                          cudf::data_type(cudf::type_id::INT8));
  std::vector<gqe::expression const*> expressions = {&cast_expr_0, &cast_expr_1, &cast_expr_2};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto expected_0 = column_wrapper<int64_t>{2, 5, 3, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_0, evaluated_results[0]);

  auto expected_1 = column_wrapper<double>{5.0, 4.0, 3.0, 3.0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, evaluated_results[1]);

  auto expected_2 = column_wrapper<int8_t>{2, 5, 3, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, evaluated_results[2]);
}

TEST(EvalExpressionsTest, EmptyInput)
{
  auto c_0   = column_wrapper<int32_t>{};
  auto c_1   = column_wrapper<int32_t>{};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionTest, SimpleUnaryNot)
{
  auto c_0   = column_wrapper<bool>{false, true, true, false};
  auto table = cudf::table_view{{c_0}};

  auto not_expr = gqe::not_expression(std::make_shared<gqe::column_reference_expression>(0));
  std::vector<gqe::expression const*> expressions = {&not_expr};

  auto expected                          = column_wrapper<bool>{true, false, false, true};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}
