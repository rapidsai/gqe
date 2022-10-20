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
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

TEST(EvalExpressionsTest, ColumnReference)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0                                  = gqe::column_reference_expression(0);
  std::vector<gqe::expression const*> expressions = {&col_ref_0};

  auto const& expected                   = c_0;
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  cudf::test::expect_columns_equal(expected, evaluated_results[0], verbosity);
}

TEST(EvalExpressionsTest, ColumnColumnEquality)
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

  cudf::test::expect_columns_equal(expected, evaluated_results[0], verbosity);
}

TEST(EvalExpressionsTest, StrColumnStrColumnEquality)
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

  cudf::test::expect_columns_equal(expected, evaluated_results[0], verbosity);
}

TEST(EvalExpressionsTest, ColumnLiteralEquality)
{
  auto c_0   = column_wrapper<int32_t>{42, 20, 42, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = std::make_shared<gqe::column_reference_expression>(0);
  auto lit42     = std::make_shared<gqe::literal_expression<int32_t>>(42);

  auto eq                                         = gqe::equal_expression(col_ref_0, lit42);
  std::vector<gqe::expression const*> expressions = {&eq};

  auto expected                          = column_wrapper<bool>{true, false, true, false};
  auto [evaluated_results, column_cache] = gqe::evaluate_expressions(table, expressions);

  cudf::test::expect_columns_equal(expected, evaluated_results[0], verbosity);
}
