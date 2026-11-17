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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/like.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cuda/std/chrono>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
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

/**
 * @brief A LIKE pattern with its input rows and expected results
 *
 * A `nullptr` input row is a null row, and its expected value is ignored.
 */
struct like_pattern_case {
  std::string name;
  std::string pattern;
  std::string escape_character;
  std::vector<char const*> input;
  std::vector<bool> expected;
};

// Keep gtest from printing the bytes of the struct when a test fails
void PrintTo(like_pattern_case const& test_case, std::ostream* os)
{
  *os << "LIKE '" << test_case.pattern << "' ESCAPE '" << test_case.escape_character << "'";
}

class LikePatternTest : public ::testing::TestWithParam<like_pattern_case> {};

TEST_P(LikePatternTest, MatchesCudf)
{
  auto const& test_case = GetParam();

  std::vector<std::string> values;
  std::vector<bool> valids;
  for (auto const* row : test_case.input) {
    values.emplace_back(row != nullptr ? row : "");
    valids.push_back(row != nullptr);
  }
  auto c_0 = cudf::test::strings_column_wrapper(values.begin(), values.end(), valids.begin());
  auto expected =
    column_wrapper<bool>(test_case.expected.begin(), test_case.expected.end(), valids.begin());
  cudf::strings_column_view const input{c_0};
  cudf::string_scalar const escape_char{test_case.escape_character};

  // The expected results must be the ones of the reference implementation
  auto const reference_result = cudf::strings::like(
    input, std::string_view{test_case.pattern}, std::string_view{test_case.escape_character});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, reference_result->view());

  // `like` and `like_utf8_bytewise` share the shift-and kernel and the pattern preprocessing
  auto const like_result = gqe::like(input, test_case.pattern, escape_char);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, like_result->view());

  auto const bytewise_result = gqe::like_utf8_bytewise(input, test_case.pattern, escape_char);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, bytewise_result->view());

  auto const utf8_result = gqe::like_utf8(input, test_case.pattern, escape_char);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, utf8_result->view());
}

// Patterns that are not enclosed in '%' on both ends: a pattern without an unescaped '%' must match
// the whole string, and the prefix and the suffix must not overlap.
INSTANTIATE_TEST_SUITE_P(
  UnanchoredPatterns,
  LikePatternTest,
  ::testing::Values(
    // The example of issue #4
    like_pattern_case{
      "SingleWildcardWithoutMultiWildcard",
      "w_dget",
      "",
      {"Widget", "WIDGET", "widget", "wIdGeT-Pro", "Gadget", "gizmo", "WidgetX", "midget", nullptr},
      {false, false, true, false, false, false, false, false, false}},
    like_pattern_case{"NoWildcard",
                      "widget",
                      "",
                      {"widget", "widgetX", "widge", "", "Widget", nullptr},
                      {true, false, false, false, false, false}},
    like_pattern_case{
      "OnlySingleWildcards", "__", "", {"", "a", "ab", "abc"}, {false, false, true, false}},
    like_pattern_case{
      "EmptyPattern", "", "", {"", "a", "abc", nullptr}, {true, false, false, false}},
    like_pattern_case{"PrefixSuffixOverlap",
                      "ab%ba",
                      "",
                      {"aba", "abba", "ab", "ba", "abxba", "a", "", "abab"},
                      {false, true, false, false, true, false, false, false}},
    like_pattern_case{
      "PrefixSuffixSameChar", "a%a", "", {"a", "aa", "aba", ""}, {false, true, true, false}},
    like_pattern_case{"StringShorterThanPrefix",
                      "abc%",
                      "",
                      {"a", "ab", "abc", "abcd", "", "xbc"},
                      {false, false, true, true, false, false}},
    like_pattern_case{"StringShorterThanPrefixWithMiddle",
                      "ab%z%",
                      "",
                      {"a", "xyz", "abz", "abxz", "ab", nullptr},
                      {false, false, true, true, false, false}},
    like_pattern_case{"StringShorterThanPrefixAndSuffix",
                      "ab%cdef",
                      "",
                      {"a", "ab", "abc", "abcdef", "abxcdef", "cdef"},
                      {false, false, false, true, true, false}},
    like_pattern_case{"StringShorterThanPrefixMiddleAndSuffix",
                      "ab%cd%ef",
                      "",
                      // "f" ends with "ef" if the suffix check starts before the string
                      {"xe", "f", "abcdef", "abcdxef", "abcef", "ef", "a"},
                      {false, false, true, true, false, false, false}},
    like_pattern_case{"SuffixOnly",
                      "%dget",
                      "",
                      {"et", "dget", "gadget", "t", "", "dgetX"},
                      {false, true, true, false, false, false}},
    like_pattern_case{"SuffixOnlyWithSingleWildcard",
                      "%d_et",
                      "",
                      {"et", "dget", "gadxet", ""},
                      {false, true, true, false}},
    like_pattern_case{"EscapedSingleWildcard",
                      "w^_dget",
                      "^",
                      {"w_dget", "w_dgetX", "widget", "w_dge", ""},
                      {true, false, false, false, false}},
    like_pattern_case{"EscapedMultiWildcard",
                      "w^%dget",
                      "^",
                      {"w%dget", "w%dgetX", "widget", "w%", ""},
                      {true, false, false, false, false}},
    like_pattern_case{"EscapedTrailingMultiWildcard",
                      "%a^%",
                      "^",
                      {"xa%", "xa%y", "a", "%", "a%"},
                      {true, false, false, false, true}},
    like_pattern_case{"EscapedEscapeBeforeTrailingMultiWildcard",
                      "%a^^%",
                      "^",
                      {"xa^y", "xa^", "a^", "xa", "a^^", "^a^"},
                      {true, true, true, false, true, true}},
    like_pattern_case{"OnlyEscapedEscapeBetweenMultiWildcards",
                      "%^^%",
                      "^",
                      {"x^y", "^", "x^", "xy"},
                      {true, true, true, false}},
    like_pattern_case{"EscapedBackslashBeforeTrailingMultiWildcard",
                      "%a\\\\%",
                      "\\",
                      {"xa\\y", "xa\\", "xa"},
                      {true, true, false}},
    like_pattern_case{"EscapeCharacterIsMultiWildcard",
                      "%w%%",
                      "%",
                      {"w%", "xw%", "w%x", "w", ""},
                      {true, false, false, false, false}},
    like_pattern_case{
      "Utf8Prefix", "é%", "", {"é", "è", "e", "éx", ""}, {true, false, false, true, false}},
    like_pattern_case{
      "Utf8Suffix", "%é", "", {"é", "xè", "xé", "è", ""}, {true, false, true, false, false}},
    like_pattern_case{"Utf8NoWildcard", "éé", "", {"éé", "éè", "é"}, {true, false, false}}),
  [](auto const& info) { return info.param.name; });

// Run the shift-and kernel with several thread blocks, on a sliced column, and with middle patterns
// that need the 16, 32 and 64 bit masks
TEST(LikeExpressionTest, ScalarLikeManyRowsAndSlicedColumn)
{
  std::vector<char const*> const base_rows = {"Widget",
                                              "widget",
                                              "gizmo",
                                              "WidgetX",
                                              nullptr,
                                              "aba",
                                              "the quick brown fox",
                                              "the quick brown fox jumps over",
                                              "",
                                              "the quick brown fox jumps over the lazy dog"};
  // Each pattern with its expected results for `base_rows`
  std::vector<std::pair<std::string, std::vector<bool>>> const patterns = {
    {"w_dget", {false, true, false, false, false, false, false, false, false, false}},
    {"%dget", {true, true, false, false, false, false, false, false, false, false}},
    {"the quick brown fox", {false, false, false, false, false, false, true, false, false, false}},
    // Middle pattern of 11, 21 and 35 bytes
    {"%quick brown%", {false, false, false, false, false, false, true, true, false, true}},
    {"the%quick brown fox jumps%",
     {false, false, false, false, false, false, false, true, false, true}},
    {"%quick brown fox jumps over the lazy%",
     {false, false, false, false, false, false, false, false, false, true}}};

  // 1000 rows, so 4 blocks of 256 threads, and a slice that starts inside the first copy
  constexpr int num_copies              = 100;
  constexpr cudf::size_type slice_begin = 3;
  constexpr cudf::size_type slice_end   = 703;

  std::vector<std::string> values;
  std::vector<bool> valids;
  for (int i = 0; i < num_copies; ++i) {
    for (auto const* row : base_rows) {
      values.emplace_back(row != nullptr ? row : "");
      valids.push_back(row != nullptr);
    }
  }
  auto c_0 = cudf::test::strings_column_wrapper(values.begin(), values.end(), valids.begin());
  auto const sliced = cudf::slice(c_0, {slice_begin, slice_end}).front();
  cudf::string_scalar const escape_char{""};

  for (auto const& [pattern, base_expected] : patterns) {
    SCOPED_TRACE("LIKE '" + pattern + "'");
    std::vector<bool> expected;
    for (int i = 0; i < num_copies; ++i) {
      expected.insert(expected.end(), base_expected.begin(), base_expected.end());
    }
    auto const expected_full =
      column_wrapper<bool>(expected.begin(), expected.end(), valids.begin());
    auto const expected_sliced = column_wrapper<bool>(
      expected.begin() + slice_begin, expected.begin() + slice_end, valids.begin() + slice_begin);

    for (auto const& [input_column, expected_column] :
         {std::pair{cudf::column_view{c_0}, cudf::column_view{expected_full}},
          std::pair{sliced, cudf::column_view{expected_sliced}}}) {
      cudf::strings_column_view const input{input_column};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column,
                                     gqe::like(input, pattern, escape_char)->view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column,
                                     gqe::like_utf8_bytewise(input, pattern, escape_char)->view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column,
                                     gqe::like_utf8(input, pattern, escape_char)->view());
    }
  }
}
