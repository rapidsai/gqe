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

TEST(LikeExpressionTest, ScalarLikeInvalidPatternSingleEscapeAtEnd)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test", "hello", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with single backslash = invalid
  cudf::string_scalar const escape_char{"^"};
  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like(table.column(0), "%test^", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeValidPatternDoubleEscapeAtEndInputEscaped)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test^^", "hello^^", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with double escape = valid (matches literal ^)
  cudf::string_scalar const escape_char{"^"};
  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like(table.column(0), "%^^", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeInvalidPatternTripleEscapeAtEnd)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test", "hello", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with triple escape = invalid (two escaped, last unescaped)
  cudf::string_scalar const escape_char{"\\"};
  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like(table.column(0), "%test\\\\\\", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeValidPatternQuadrupleEscapeAtEndInputEscaped)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test\\\\", "hello\\\\", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with four escapes = valid (matches two literal backslashes)
  cudf::string_scalar const escape_char{"\\"};
  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like(table.column(0), "%\\\\\\\\", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8InvalidPatternSingleEscapeAtEnd)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test啊", "hello世界", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with single backslash = invalid
  cudf::string_scalar const escape_char{"|"};
  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%test|", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8ValidPatternDoubleEscapeAtEndInputEscaped)
{
  auto c_0   = cudf::test::strings_column_wrapper{"test||", "hello||", "world"};
  auto table = cudf::table_view{{c_0}};

  // Pattern ending with double escape = valid (matches literal |)
  cudf::string_scalar const escape_char{"|"};
  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%||", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapeCharIsPercentBoth)
{
  // Pattern %%hello% with escape '%' ends with single % = INVALID (odd count)
  auto c_0   = cudf::test::strings_column_wrapper{"%hello%", "%helloworld", "hello"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"%"};
  auto expected         = column_wrapper<bool>{true, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%%hello%%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapeCharIsPercentSuffix)
{
  // Pattern %world%% with escape '%' = ends with literal % (has suffix)
  auto c_0   = cudf::test::strings_column_wrapper{"world%", "helloworld", "world"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"%"};
  auto expected         = column_wrapper<bool>{true, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%world%%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapeCharInvalidSuffixInputPrefixEscaped)
{
  // Pattern %%%hello% with escape '%' ends with single % = INVALID (odd count)
  auto c_0   = cudf::test::strings_column_wrapper{"%hello", "%%hello", "hello"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"%"};
  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%%%hello%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapeCharInvalidSuffixInputSuffixEscaped)
{
  // Pattern %world%%% with escape '%' ends with %%% (3 = odd) = INVALID
  auto c_0   = cudf::test::strings_column_wrapper{"world%", "world%%", "world"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"%"};
  auto expected         = column_wrapper<bool>{false, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%world%%%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapedWildcardAtEnd)
{
  // Pattern %world\% with escape '\' = ends with escaped % (has suffix)
  // Should fall back to cudf::strings::like
  auto c_0   = cudf::test::strings_column_wrapper{"world%", "helloworld%", "world"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"\\"};
  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), "%world\\%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapedEscapeBeforeWildcard)
{
  // Pattern %world\\% with escape '\' = ends with \\ (escaped \) then % (wildcard) = no suffix
  auto c_0   = cudf::test::strings_column_wrapper{"world\\", "world\\suffix", "world\\"};
  auto table = cudf::table_view{{c_0}};

  cudf::string_scalar const escape_char{"\\"};
  auto expected = column_wrapper<bool>{true, true, true};
  // this uses the cudf::like although the last '%' is a wildcard
  auto evaluated_result = gqe::like_utf8(table.column(0), "%world\\\\%", escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeInvalidSuffixEscapeChar)
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

TEST(LikeExpressionTest, ScalarLikeEscapedVsWildcard)
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

TEST(LikeExpressionTest, ScalarLikeEscapedVsWildcardPreSuffix)
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

TEST(LikeExpressionTest, ScalarLikeMultiWithoutWildcard)
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

TEST(LikeExpressionTest, ScalarLikeMultiWithoutWildcardPreSuffix)
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

TEST(LikeExpressionTest, ScalarLikeUtf8)
{
  auto c_0   = cudf::test::strings_column_wrapper{"az啊a", "abab啊abba", "aaxa"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr =
    gqe::like_expression(std::make_shared<gqe::column_reference_expression>(0), "%a__a%", "");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, true};
  auto evaluated_result = gqe::like_utf8(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8Multi)
{
  auto c_0 = cudf::test::strings_column_wrapper{
    "az啊ahello你好世界world", "abab啊abbahello世界world", "ax啊ahello你好world"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "%a_啊a%hello%世界world%", "");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8MultiPrefixSuffix)
{
  // the last one should not match due to lack of the suffix "!"
  auto c_0 = cudf::test::strings_column_wrapper{
    "啊az啊ahello你好世界world!", "啊abab啊abbahello世界world!", "啊ax啊ahello世界你好world"};
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "啊%a_啊a%hello%世_world%!", "");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}

TEST(LikeExpressionTest, ScalarLikeUtf8EscapedVsWildcardUnderscore)
{
  // Test pattern: %hello^_the_e%
  // ^_ is an escaped underscore (literal _)
  // _ after "the" is a single wildcard (matches any single UTF-8 character)
  // escape character: ^
  auto c_0 = cudf::test::strings_column_wrapper{
    "hello_there",                // MATCH: contains "hello_the" + "r" + "e"
    "prefix hello_these suffix",  // MATCH: contains "hello_the" + "s" + "e"
    "hello_the世e",               // MATCH: contains "hello_the" + "世" + "e"
    "helloxthese",                // NO MATCH: missing literal _
    "hello_thee"                  // NO MATCH: missing wildcard character between "the" and "e"
  };
  auto table = cudf::table_view{{c_0}};

  auto like_expr = gqe::like_expression(
    std::make_shared<gqe::column_reference_expression>(0), "%hello^_the_e%", "^");
  cudf::string_scalar const escape_char{like_expr.escape_character()};

  auto expected         = column_wrapper<bool>{true, true, true, false, false};
  auto evaluated_result = gqe::like_utf8(table.column(0), like_expr.pattern(), escape_char);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, evaluated_result->view());
}
