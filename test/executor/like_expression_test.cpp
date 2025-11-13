/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/like.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <cuda/std/chrono>
#include <memory>
#include <string>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

TEST(EvalExpressionTest, ScalarLikeInvalidSuffixEscapeChar)
{
  auto c_0   = cudf::test::strings_column_wrapper{"az_a^", "abab_abba", "aaxa^"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr =
    gqe::like_expression(std::make_shared<gqe::column_reference_expression>(0), "%a__a%^", "^");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(EvalExpressionTest, ScalarLikeEscapedVsWildcard)
{
  // Test pattern: %hello^_the_e%
  // ^_ is an escaped underscore (literal _)
  // _ after "the" is a single wildcard (matches any single UTF-8 character)
  // escape character: ^
  auto c_0 = cudf::test::strings_column_wrapper{
    "hello_there",                // MATCH: contains "hello_the" + "r" + "e"
    "prefix hello_these suffix",  // MATCH: contains "hello_the" + "s" + "e"
    "hello_theze",                // MATCH: contains "hello_the" + "z" + "e"
    "helloxthese",                // NO MATCH: missing literal _
    "hello_thee"                  // NO MATCH: missing wildcard character between "the" and "e"
  };
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "%hello^_the_e%", "^");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, true, false, false};
  auto evaluated_result = gqe::like(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(EvalExpressionTest, ScalarLikeEscapedVsWildcardPreSuffix)
{
  // Test pattern: %hello^_the_e%
  // ^_ is an escaped underscore (literal _)
  // _ after "the" is a single wildcard (matches any single UTF-8 character)
  // escape character: ^
  auto c_0 = cudf::test::strings_column_wrapper{
    "hello_thereworldeveryonehello_world!",
    "hello_theseprefixeveryonehello_these suffixhello_world.",
    "hello_thezeprefix_everyone_suffixhello_world;",
    "helloxtheseprefixeveryonehello_world!",  // prefix not matched
    "hello_theeeprefixeveryonehellozworld!",  // suffix not matched
    "hello_theeeprefixeveryonehello_world^"   // matched
  };
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(std::make_shared<gqe::column_reference_expression>(0),
                                        "hello^_the_e%everyone%hello^_world_",
                                        "^");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, true, false, false, true};
  auto evaluated_result = gqe::like(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(EvalExpressionTest, ScalarLikeMultiWithoutWildcard)
{
  auto c_0 = cudf::test::strings_column_wrapper{
    "az啊ahello你好世界world", "ababz啊abbahello世界world", "ax啊ahello你好world"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "%z啊a%hello%世界world%", "");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected = column_wrapper<bool>{true, true, false};
  auto evaluated_result =
    gqe::like_utf8_bytewise(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(EvalExpressionTest, ScalarLikeMultiWithoutWildcardPreSuffix)
{
  // the last one should not match due to lack of the suffix "!"
  auto c_0 = cudf::test::strings_column_wrapper{
    "啊az啊ahello你好世界world!", "啊ababz啊abbahello世界world!", "啊ax啊ahello世界你好world"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "啊%z啊a%hello%世界world%!", "");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected = column_wrapper<bool>{true, true, false};
  auto evaluated_result =
    gqe::like_utf8_bytewise(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}
