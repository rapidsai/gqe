/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/eval.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

// TODO - consider typed tests.

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

// Used for decimal tests, where columns can be fixed or decimal type.
template <template <typename> typename TWrapper, typename TData>
auto make_numeric_column_wrapper(std::initializer_list<TData> values, numeric::scale_type scale)
{
  using wrapper_type = TWrapper<TData>;
  if constexpr (std::is_same_v<wrapper_type, cudf::test::fixed_width_column_wrapper<TData>>) {
    std::vector<TData> scaled_values(values);
    for (auto& val : scaled_values) {
      val = static_cast<int>(val * pow(10.0, double(scale)));
    }
    return cudf::test::fixed_width_column_wrapper<TData>(scaled_values.begin(),
                                                         scaled_values.end());
  } else if constexpr (std::is_same_v<wrapper_type,
                                      cudf::test::fixed_point_column_wrapper<TData>>) {
    return cudf::test::fixed_point_column_wrapper<TData>(values, scale);
  } else {
    throw std::runtime_error("Unsupported column wrapper.");
  }
}

template <template <typename> typename ResultColType, typename ResultDataType>
void expect_columns_equal(cudf::column_view const& expected,
                          cudf::column_view const& evaluated,
                          float abs_error)
{
  cudf::data_type result_type = expected.type();

  auto diff_col =
    cudf::binary_operation(evaluated, expected, cudf::binary_operator::SUB, result_type);

  auto abs_diff_col = cudf::unary_operation(diff_col->view(), cudf::unary_operator::ABS);

  auto abs_err_col = make_numeric_column_wrapper<ResultColType, ResultDataType>(
    {ResultDataType(abs_error * pow(10, -result_type.scale())),
     ResultDataType(abs_error * pow(10, -result_type.scale())),
     ResultDataType(abs_error * pow(10, -result_type.scale())),
     ResultDataType(abs_error * pow(10, -result_type.scale()))},
    numeric::scale_type{result_type.scale()});

  auto comp_col = cudf::binary_operation(abs_diff_col->view(),
                                         abs_err_col,
                                         cudf::binary_operator::LESS_EQUAL,
                                         cudf::data_type(cudf::type_id::BOOL8));

  auto truth_col = column_wrapper<bool>{true, true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(comp_col->view(), truth_col);
}

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

template <typename FixedPointSizeType>
void DecimalColumnDecimalColumnEqualityTest()
{
  auto c_0   = cudf::test::fixed_point_column_wrapper<FixedPointSizeType>{{FixedPointSizeType(11),
                                                                           FixedPointSizeType(20),
                                                                           FixedPointSizeType(33),
                                                                           FixedPointSizeType(40)},
                                                                          numeric::scale_type{-1}};
  auto c_1   = cudf::test::fixed_point_column_wrapper<FixedPointSizeType>{{FixedPointSizeType(11),
                                                                           FixedPointSizeType(22),
                                                                           FixedPointSizeType(33),
                                                                           FixedPointSizeType(44)},
                                                                          numeric::scale_type{-1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, Decimal32ColumnDecimal32ColumnEquality)
{
  DecimalColumnDecimalColumnEqualityTest<int32_t>();
}

TEST(EvalExpressionsTest, Decimal64lumnDecimal64ColumnEquality)
{
  DecimalColumnDecimalColumnEqualityTest<int64_t>();
}

TEST(EvalExpressionsTest, Decimal128ColumnDecimal128ColumnEquality)
{
  DecimalColumnDecimalColumnEqualityTest<__int128_t>();
}

template <typename FloatType, typename FixedPointSizeType>
void FloatingPointColumnDecimalColumnEqualityTest()
{
  // 1.0 and 3.0 are representable in floating-point and fixed-point, but 2.2 and 4.4 are not
  // representable as doubles.
  auto c_0   = column_wrapper<FloatType>{1.0, 2.2, 3.0, 4.4};
  auto c_1   = cudf::test::fixed_point_column_wrapper<FixedPointSizeType>{{FixedPointSizeType(10),
                                                                           FixedPointSizeType(22),
                                                                           FixedPointSizeType(30),
                                                                           FixedPointSizeType(44)},
                                                                          numeric::scale_type{-1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto col_ref_1 = std::make_shared<gqe::column_reference_expression>(1);

  auto eq_0_1                                     = gqe::equal_expression(col_ref_0, col_ref_1);
  std::vector<gqe::expression const*> expressions = {&eq_0_1};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

// @TODO Enable once we are on the latest version of cudf.

TEST(EvalExpressionsTest, DISABLED_FloatColumnDecimal32ColumnEquality)
{
  FloatingPointColumnDecimalColumnEqualityTest<float, int32_t>();
}

TEST(EvalExpressionsTest, DISABLED_DoubleColumnDecimal64ColumnEquality)
{
  FloatingPointColumnDecimalColumnEqualityTest<double, int64_t>();
}

TEST(EvalExpressionsTest, DISABLED_DoublePointColumnDecimal128ColumnEquality)
{
  FloatingPointColumnDecimalColumnEqualityTest<double, __int128_t>();
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

  auto expected                          = column_wrapper<int64_t>{0, 1, 1, 2};
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

template <template <typename> typename LhsType,
          typename LhsDataType,
          template <typename>
          typename RhsType,
          typename RhsDataType,
          template <typename>
          typename ResultColType,
          typename ResultDataType>
void DecimalAdditionTest()
{
  std::vector<cudf::test::detail::column_wrapper> table_columns;
  std::vector<cudf::column_view> table_column_views;
  std::shared_ptr<gqe::expression> lhs;
  std::shared_ptr<gqe::expression> rhs;
  cudf::data_type lhs_type;
  cudf::data_type rhs_type;

  bool is_lhs_scalar = true;
  bool is_rhs_scalar = true;

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs = std::make_shared<gqe::literal_expression<LhsDataType>>(
      LhsDataType(2, numeric::scale_type(-3)));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs      = std::make_shared<gqe::literal_expression<LhsDataType>>(LhsDataType(2));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>>) {
    auto col = make_numeric_column_wrapper<LhsType, LhsDataType>(
      {LhsDataType(-2000), LhsDataType(5000), LhsDataType(3000), LhsDataType(6000)},
      numeric::scale_type{-3});

    lhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    lhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_lhs_scalar = false;
  }

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs = std::make_shared<gqe::literal_expression<RhsDataType>>(
      RhsDataType(5, numeric::scale_type(-2)));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs      = std::make_shared<gqe::literal_expression<RhsDataType>>(RhsDataType(5));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>>) {
    auto col = make_numeric_column_wrapper<RhsType, RhsDataType>(
      {RhsDataType(-500), RhsDataType(400), RhsDataType(300), RhsDataType(300)},
      numeric::scale_type{-2});

    rhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    rhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_rhs_scalar = false;
  }

  auto table                                      = cudf::table_view{table_column_views};
  auto expr                                       = gqe::add_expression(lhs, rhs);
  std::vector<gqe::expression const*> expressions = {&expr};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto result_type = gqe::arithmetic_output_type(cudf::binary_operator::ADD, lhs_type, rhs_type);

  if (is_lhs_scalar && is_rhs_scalar) {
    assert(0 && "Not Supported");
  } else if (!is_lhs_scalar && is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(3 * pow(10, -result_type.scale())),
       ResultDataType(10 * pow(10, -result_type.scale())),
       ResultDataType(8 * pow(10, -result_type.scale())),
       ResultDataType(11 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else if (is_lhs_scalar && !is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-3 * pow(10, -result_type.scale())),
       ResultDataType(6 * pow(10, -result_type.scale())),
       ResultDataType(5 * pow(10, -result_type.scale())),
       ResultDataType(5 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-7 * pow(10, -result_type.scale())),
       ResultDataType(9 * pow(10, -result_type.scale())),
       ResultDataType(6 * pow(10, -result_type.scale())),
       ResultDataType(9 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  }
}

TEST(EvalExpressionsTest, Decimal32ColDecimal32ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      int32_t>();
}
TEST(EvalExpressionsTest, Decimal64ColDecimal64ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      int64_t,
                      cudf::test::fixed_point_column_wrapper,
                      int64_t,
                      cudf::test::fixed_point_column_wrapper,
                      int64_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal128ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, Decimal32ColDecimal128ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColInt32ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_width_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, FloatColDecimal128ColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_width_column_wrapper,
                      float,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDoubleColAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_width_column_wrapper,
                      double,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}

TEST(EvalExpressionsTest, Decimal128LitDecimal128ColAddition)
{
  DecimalAdditionTest<cudf::fixed_point_scalar,
                      numeric::decimal128,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal32LitAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::fixed_point_scalar,
                      numeric::decimal32,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColFloatLitAddition)
{
  DecimalAdditionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::numeric_scalar,
                      float,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}
TEST(EvalExpressionsTest, DoubleLitDecimal128ColAddition)
{
  DecimalAdditionTest<cudf::numeric_scalar,
                      double,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>();
}

template <template <typename> typename LhsType,
          typename LhsDataType,
          template <typename>
          typename RhsType,
          typename RhsDataType,
          template <typename>
          typename ResultColType,
          typename ResultDataType>
void DecimalSubtractionTest()
{
  std::vector<cudf::test::detail::column_wrapper> table_columns;
  std::vector<cudf::column_view> table_column_views;
  std::shared_ptr<gqe::expression> lhs;
  std::shared_ptr<gqe::expression> rhs;
  cudf::data_type lhs_type;
  cudf::data_type rhs_type;

  bool is_lhs_scalar = true;
  bool is_rhs_scalar = true;

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs = std::make_shared<gqe::literal_expression<LhsDataType>>(
      LhsDataType(2, numeric::scale_type(-3)));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs      = std::make_shared<gqe::literal_expression<LhsDataType>>(LhsDataType(2));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>>) {
    auto col = make_numeric_column_wrapper<LhsType, LhsDataType>(
      {LhsDataType(-2000), LhsDataType(5000), LhsDataType(3000), LhsDataType(3000)},
      numeric::scale_type{-3});

    lhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    lhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_lhs_scalar = false;
  }

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs = std::make_shared<gqe::literal_expression<RhsDataType>>(
      RhsDataType(5, numeric::scale_type(-2)));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs      = std::make_shared<gqe::literal_expression<RhsDataType>>(RhsDataType(5));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>>) {
    auto col = make_numeric_column_wrapper<RhsType, RhsDataType>(
      {RhsDataType(-500), RhsDataType(400), RhsDataType(300), RhsDataType(600)},
      numeric::scale_type{-2});

    rhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    rhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_rhs_scalar = false;
  }

  auto table                                      = cudf::table_view{table_column_views};
  auto expr                                       = gqe::subtract_expression(lhs, rhs);
  std::vector<gqe::expression const*> expressions = {&expr};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto result_type = gqe::arithmetic_output_type(cudf::binary_operator::SUB, lhs_type, rhs_type);

  if (is_lhs_scalar && is_rhs_scalar) {
    assert(0 && "Not Supported");
  } else if (!is_lhs_scalar && is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-7 * pow(10, -result_type.scale())),
       ResultDataType(0 * pow(10, -result_type.scale())),
       ResultDataType(-2 * pow(10, -result_type.scale())),
       ResultDataType(-2 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else if (is_lhs_scalar && !is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(7 * pow(10, -result_type.scale())),
       ResultDataType(-2 * pow(10, -result_type.scale())),
       ResultDataType(-1 * pow(10, -result_type.scale())),
       ResultDataType(-4 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(3 * pow(10, -result_type.scale())),
       ResultDataType(1 * pow(10, -result_type.scale())),
       ResultDataType(0 * pow(10, -result_type.scale())),
       ResultDataType(-3 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  }
}

TEST(EvalExpressionsTest, Decimal32ColDecimal32ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         int32_t,
                         cudf::test::fixed_point_column_wrapper,
                         int32_t,
                         cudf::test::fixed_point_column_wrapper,
                         int32_t>();
}
TEST(EvalExpressionsTest, Decimal64ColDecimal64ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         int64_t,
                         cudf::test::fixed_point_column_wrapper,
                         int64_t,
                         cudf::test::fixed_point_column_wrapper,
                         int64_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal128ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, Decimal32ColDecimal128ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         int32_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColInt32ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_width_column_wrapper,
                         int32_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, FloatColDecimal128ColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_width_column_wrapper,
                         float,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDoubleColSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_width_column_wrapper,
                         double,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}

TEST(EvalExpressionsTest, Decimal128LitDecimal128ColSubtraction)
{
  DecimalSubtractionTest<cudf::fixed_point_scalar,
                         numeric::decimal128,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal32LitSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::fixed_point_scalar,
                         numeric::decimal32,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColFloatLitSubtraction)
{
  DecimalSubtractionTest<cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::numeric_scalar,
                         float,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}
TEST(EvalExpressionsTest, DoubleLitDecimal128ColSubtraction)
{
  DecimalSubtractionTest<cudf::numeric_scalar,
                         double,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t,
                         cudf::test::fixed_point_column_wrapper,
                         __int128_t>();
}

template <template <typename> typename LhsType,
          typename LhsDataType,
          template <typename>
          typename RhsType,
          typename RhsDataType,
          template <typename>
          typename ResultColType,
          typename ResultDataType>
void DecimalMultiplicationTest()
{
  std::vector<cudf::test::detail::column_wrapper> table_columns;
  std::vector<cudf::column_view> table_column_views;
  std::shared_ptr<gqe::expression> lhs;
  std::shared_ptr<gqe::expression> rhs;
  cudf::data_type lhs_type;
  cudf::data_type rhs_type;

  bool is_lhs_scalar = true;
  bool is_rhs_scalar = true;

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs = std::make_shared<gqe::literal_expression<LhsDataType>>(
      LhsDataType(2, numeric::scale_type(-3)));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs      = std::make_shared<gqe::literal_expression<LhsDataType>>(LhsDataType(2));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>>) {
    auto col = make_numeric_column_wrapper<LhsType, LhsDataType>(
      {LhsDataType(-2000), LhsDataType(5000), LhsDataType(3000), LhsDataType(4000)},
      numeric::scale_type{-3});

    lhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    lhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_lhs_scalar = false;
  }

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs = std::make_shared<gqe::literal_expression<RhsDataType>>(
      RhsDataType(5, numeric::scale_type(-2)));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs      = std::make_shared<gqe::literal_expression<RhsDataType>>(RhsDataType(5));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>>) {
    auto col = make_numeric_column_wrapper<RhsType, RhsDataType>(
      {RhsDataType(500), RhsDataType(400), RhsDataType(300), RhsDataType(600)},
      numeric::scale_type{-2});

    rhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    rhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_rhs_scalar = false;
  }

  auto table                                      = cudf::table_view{table_column_views};
  auto expr                                       = gqe::multiply_expression(lhs, rhs);
  std::vector<gqe::expression const*> expressions = {&expr};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto result_type = gqe::arithmetic_output_type(cudf::binary_operator::MUL, lhs_type, rhs_type);

  if (is_lhs_scalar && is_rhs_scalar) {
    assert(0 && "Not Supported");
  } else if (!is_lhs_scalar && is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-10 * pow(10, -result_type.scale())),
       ResultDataType(25 * pow(10, -result_type.scale())),
       ResultDataType(15 * pow(10, -result_type.scale())),
       ResultDataType(20 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else if (is_lhs_scalar && !is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(10 * pow(10, -result_type.scale())),
       ResultDataType(8 * pow(10, -result_type.scale())),
       ResultDataType(6 * pow(10, -result_type.scale())),
       ResultDataType(12 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  } else {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-10 * pow(10, -result_type.scale())),
       ResultDataType(20 * pow(10, -result_type.scale())),
       ResultDataType(9 * pow(10, -result_type.scale())),
       ResultDataType(24 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
  }
}

TEST(EvalExpressionsTest, Decimal32ColDecimal32ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            int32_t,
                            cudf::test::fixed_point_column_wrapper,
                            int32_t,
                            cudf::test::fixed_point_column_wrapper,
                            int32_t>();
}
TEST(EvalExpressionsTest, Decimal64ColDecimal64ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            int64_t,
                            cudf::test::fixed_point_column_wrapper,
                            int64_t,
                            cudf::test::fixed_point_column_wrapper,
                            int64_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal128ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, Decimal32ColDecimal128ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            int32_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColInt32ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_width_column_wrapper,
                            int32_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, FloatColDecimal128ColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_width_column_wrapper,
                            float,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDoubleColMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_width_column_wrapper,
                            double,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}

TEST(EvalExpressionsTest, Decimal128LitDecimal128ColMultiplication)
{
  DecimalMultiplicationTest<cudf::fixed_point_scalar,
                            numeric::decimal128,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColDecimal32LitMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::fixed_point_scalar,
                            numeric::decimal32,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, Decimal128ColFloatLitMultiplication)
{
  DecimalMultiplicationTest<cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::numeric_scalar,
                            float,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}
TEST(EvalExpressionsTest, DoubleLitDecimal128ColMultiplication)
{
  DecimalMultiplicationTest<cudf::numeric_scalar,
                            double,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t,
                            cudf::test::fixed_point_column_wrapper,
                            __int128_t>();
}

template <template <typename> typename LhsType,
          typename LhsDataType,
          template <typename>
          typename RhsType,
          typename RhsDataType,
          template <typename>
          typename ResultColType,
          typename ResultDataType>
void DecimalDivisionTest(double abs_error = 0.0)
{
  std::vector<cudf::test::detail::column_wrapper> table_columns;
  std::vector<cudf::column_view> table_column_views;
  std::shared_ptr<gqe::expression> lhs;
  std::shared_ptr<gqe::expression> rhs;
  cudf::data_type lhs_type;
  cudf::data_type rhs_type;

  bool is_lhs_scalar = true;
  bool is_rhs_scalar = true;

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs = std::make_shared<gqe::literal_expression<LhsDataType>>(
      LhsDataType(2, numeric::scale_type(-3)));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<LhsDataType>, LhsType<LhsDataType>>) {
    lhs      = std::make_shared<gqe::literal_expression<LhsDataType>>(LhsDataType(2));
    lhs_type = static_cast<gqe::literal_expression<LhsDataType>*>(lhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<LhsDataType>,
                                      LhsType<LhsDataType>>) {
    auto col = make_numeric_column_wrapper<LhsType, LhsDataType>(
      {LhsDataType(-2000), LhsDataType(5000), LhsDataType(3000), LhsDataType(8000)},
      numeric::scale_type{-3});

    lhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    lhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_lhs_scalar = false;
  }

  if constexpr (std::is_same_v<cudf::fixed_point_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs = std::make_shared<gqe::literal_expression<RhsDataType>>(
      RhsDataType(5, numeric::scale_type(-2)));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::numeric_scalar<RhsDataType>, RhsType<RhsDataType>>) {
    rhs      = std::make_shared<gqe::literal_expression<RhsDataType>>(RhsDataType(5));
    rhs_type = static_cast<gqe::literal_expression<RhsDataType>*>(rhs.get())->data_type(
      std::vector<cudf::data_type>{});
  } else if constexpr (std::is_same_v<cudf::test::fixed_point_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>> ||
                       std::is_same_v<cudf::test::fixed_width_column_wrapper<RhsDataType>,
                                      RhsType<RhsDataType>>) {
    auto col = make_numeric_column_wrapper<RhsType, RhsDataType>(
      {RhsDataType(500), RhsDataType(400), RhsDataType(300), RhsDataType(200)},
      numeric::scale_type{-2});

    rhs_type = static_cast<cudf::column_view>(col).type();
    table_columns.push_back(std::move(col));
    table_column_views.push_back(table_columns.back());
    rhs           = std::make_shared<gqe::column_reference_expression>(table_columns.size() - 1);
    is_rhs_scalar = false;
  }

  auto table                                      = cudf::table_view{table_column_views};
  auto expr                                       = gqe::divide_expression(lhs, rhs);
  std::vector<gqe::expression const*> expressions = {&expr};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  // Floating point columns and cudf::cast introduce small precision artifacts, even for values
  // that are representable in IEEE754. This seems to happen when casting to larger precision type,
  // like float->DECIMAL128. We use absolute error ignore these artifacts.

  auto result_type = gqe::arithmetic_output_type(cudf::binary_operator::DIV, lhs_type, rhs_type);

  if (is_lhs_scalar && is_rhs_scalar) {
    assert(0 && "Not Supported");
  } else if (!is_lhs_scalar && is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-0.4 * pow(10, -result_type.scale())),
       ResultDataType(1.0 * pow(10, -result_type.scale())),
       ResultDataType(0.6 * pow(10, -result_type.scale())),
       ResultDataType(1.6 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    expect_columns_equal<ResultColType, ResultDataType>(expected, evaluated_results[0], abs_error);
  } else if (is_lhs_scalar && !is_rhs_scalar) {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(0.4 * pow(10, -result_type.scale())),
       ResultDataType(0.5 * pow(10, -result_type.scale())),
       ResultDataType((2.0 / 3.0) * pow(10, -result_type.scale())),
       ResultDataType(1.0 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    expect_columns_equal<ResultColType, ResultDataType>(expected, evaluated_results[0], abs_error);
  } else {
    auto expected = make_numeric_column_wrapper<ResultColType, ResultDataType>(
      {ResultDataType(-0.4 * pow(10, -result_type.scale())),
       ResultDataType(1.25 * pow(10, -result_type.scale())),
       ResultDataType(1.0 * pow(10, -result_type.scale())),
       ResultDataType(4.0 * pow(10, -result_type.scale()))},
      numeric::scale_type{result_type.scale()});
    expect_columns_equal<ResultColType, ResultDataType>(expected, evaluated_results[0], abs_error);
  }
}

constexpr double decimal_division_error = 0.001;

TEST(EvalExpressionsTest, Decimal32ColDecimal32ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      int64_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal64ColDecimal64ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      int64_t,
                      cudf::test::fixed_point_column_wrapper,
                      int64_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal128ColDecimal128ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal32ColDecimal128ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal128ColInt32ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_width_column_wrapper,
                      int32_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, FloatColDecimal128ColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_width_column_wrapper,
                      float,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal128ColDoubleColDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_width_column_wrapper,
                      double,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}

TEST(EvalExpressionsTest, Decimal128LitDecimal128ColDivision)
{
  DecimalDivisionTest<cudf::fixed_point_scalar,
                      numeric::decimal128,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal128ColDecimal32LitDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::fixed_point_scalar,
                      numeric::decimal32,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, Decimal128ColFloatLitDivision)
{
  DecimalDivisionTest<cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::numeric_scalar,
                      float,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
}
TEST(EvalExpressionsTest, DoubleLitDecimal128ColDivision)
{
  DecimalDivisionTest<cudf::numeric_scalar,
                      double,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t,
                      cudf::test::fixed_point_column_wrapper,
                      __int128_t>(decimal_division_error);
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

TEST(EvalExpressionTest, ScalarFunctionLike)
{
  auto c_0   = cudf::test::strings_column_wrapper{"azaa", "ababaabba", "aaxa"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr =
    gqe::like_expression(std::make_shared<gqe::column_reference_expression>(0), "%a_aa%", "");
  std::vector<gqe::expression const*> expressions = {&like_expr};

  auto expected                          = column_wrapper<bool>{true, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionTest, ScalarFunctionSubstr)
{
  auto c_0   = cudf::test::strings_column_wrapper{"azaa", "ababaabba", "aaxa"};
  auto table = cudf::table_view{{c_0}};

  auto substr_expr =
    gqe::substr_expression(std::make_shared<gqe::column_reference_expression>(0), 1, 2);
  std::vector<gqe::expression const*> expressions = {&substr_expr};

  auto expected                          = cudf::test::strings_column_wrapper{"za", "ba", "ax"};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionTest, ScalarFunctionDatepart)
{
  auto t1 = cuda::std::chrono::duration<int, cuda::std::ratio<86400>>(0);
  auto t2 = cuda::std::chrono::duration<int, cuda::std::ratio<86400>>(19358);

  cudf::timestamp_D ct1{t1};  // 1970-01-01
  cudf::timestamp_D ct2{t2};  // 2023-01-01

  auto c_0   = column_wrapper<cudf::timestamp_D>{ct1, ct2};
  auto table = cudf::table_view{{c_0}};

  auto dt_component = gqe::datepart_expression::datetime_component::year;
  auto dp_expr =
    gqe::datepart_expression(std::make_shared<gqe::column_reference_expression>(0), dt_component);
  std::vector<gqe::expression const*> expressions = {&dp_expr};

  auto expected                          = column_wrapper<int16_t>{1970, 2023};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_results[0]);
}

TEST(EvalExpressionsTest, LogicalOperators)
{
  // [true, false, null]
  auto c_0 = column_wrapper<bool>{{true, false, true}, {true, true, false}};
  // [null, null, null]
  auto c_1   = column_wrapper<bool>{{false, true, true}, {false, false, false}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto and_expr =
    gqe::logical_and_expression(std::make_shared<gqe::column_reference_expression>(0),
                                std::make_shared<gqe::column_reference_expression>(1));
  auto or_expr = gqe::logical_or_expression(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(1));
  std::vector<gqe::expression const*> expressions = {&and_expr, &or_expr};

  // [null, false, null]
  auto expected_and = column_wrapper<bool>{{false, false, false}, {false, true, false}};
  // [true, null, null]
  auto expected_or = column_wrapper<bool>{{true, true, true}, {true, false, false}};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_and, evaluated_results[0]);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_or, evaluated_results[1]);
}

TEST(EvalExpressionsTest, IsNull)
{
  auto c_0 = column_wrapper<int32_t>({1, 2, 3, 4, 5}, {false, true, false, true, false});
  auto c_1 = cudf::test::strings_column_wrapper({"azaa", "ababaabba", "aaxa", "fsadfsa", "dsfdsaf"},
                                                {true, false, true, false, true});

  auto table = cudf::table_view{{c_0, c_1}};

  auto is_null_expr_0 =
    gqe::is_null_expression(std::make_shared<gqe::column_reference_expression>(0));
  auto is_null_expr_1 =
    gqe::is_null_expression(std::make_shared<gqe::column_reference_expression>(1));

  std::vector<gqe::expression const*> expressions = {&is_null_expr_0, &is_null_expr_1};
  auto [evaluated_results, column_cache]          = gqe::evaluate_expressions(table, expressions);

  auto expected_0 = column_wrapper<bool>{{true, false, true, false, true}, {1, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_0, evaluated_results[0]);

  auto expected_1 = column_wrapper<bool>{{false, true, false, true, false}, {1, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, evaluated_results[1]);
}
