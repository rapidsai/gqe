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

#include <gqe/storage/zone_map.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/json_formatter.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>

#include <gtest/gtest.h>

#include <cudf_test/column_wrapper.hpp>

#include <absl/strings/str_format.h>
#include <gqe/expression/unary_op.hpp>

static constexpr cudf::size_type INPUT_COLUMN        = 1;
static constexpr cudf::size_type ZONE_MAP_MIN_COLUMN = INPUT_COLUMN * 2;
static constexpr cudf::size_type ZONE_MAP_MAX_COLUMN = INPUT_COLUMN * 2 + 1;

using expr_ptr = std::shared_ptr<gqe::expression>;

struct test_parameters {
  std::string_view name;
  expr_ptr input;
  expr_ptr expected = nullptr;
  static std::string to_string(const testing::TestParamInfo<test_parameters>& info)
  {
    return std::string{info.param.name};
  }
};

class ZoneMapExpressionTransformationTest : public testing::TestWithParam<test_parameters> {};

TEST_P(ZoneMapExpressionTransformationTest, TestTransformation)
{
  auto const params = GetParam();
  assert(params.input != nullptr);
  assert(params.expected != nullptr);
  auto actual = gqe::zone_map_expression_transformer::transform(*params.input);
  assert(actual);
  GQE_LOG_DEBUG("Test: {}\nInput:\n{}\nExpected:\n{}\nActual:\n{}",
                ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                gqe::expression_json_formatter::to_json(*params.input),
                gqe::expression_json_formatter::to_json(*params.expected),
                gqe::expression_json_formatter::to_json(*actual));
  EXPECT_TRUE(*params.expected == *actual);
}

// Test general transformations of comparisons and boolean operations
INSTANTIATE_TEST_SUITE_P(
  TransformationTests,
  ZoneMapExpressionTransformationTest,
  testing::Values(
    // column < value → MIN < value
    test_parameters{.name  = "lessLhsColumn",
                    .input = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0)),
                    .expected = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // value < column → value < MAX
    test_parameters{.name  = "lessRhsColumn",
                    .input = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN)),
                    .expected = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN))},
    // column <= value → MIN <= value
    test_parameters{.name  = "lessEqualLhsColumn",
                    .input = std::make_shared<gqe::less_equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0)),
                    .expected = std::make_shared<gqe::less_equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // value <= column → value <= MAX
    test_parameters{.name  = "lessEqualRhsColumn",
                    .input = std::make_shared<gqe::less_equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN)),
                    .expected = std::make_shared<gqe::less_equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN))},
    // column > value → MAX > value
    test_parameters{.name  = "greaterLhsColumn",
                    .input = std::make_shared<gqe::greater_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0)),
                    .expected = std::make_shared<gqe::greater_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // value > column → value > MIN
    test_parameters{.name  = "greaterRhsColumn",
                    .input = std::make_shared<gqe::greater_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN)),
                    .expected = std::make_shared<gqe::greater_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN))},
    // column >= value → MAX >= value
    test_parameters{.name  = "greaterEqualLhsColumn",
                    .input = std::make_shared<gqe::greater_equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0)),
                    .expected = std::make_shared<gqe::greater_equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // value >= column → value >= MIN
    test_parameters{.name  = "greaterEqualColumnRhs",
                    .input = std::make_shared<gqe::greater_equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN)),
                    .expected = std::make_shared<gqe::greater_equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN))},
    // column == value → MIN <= value AND MAX => value
    test_parameters{.name  = "equalLhsColumn",
                    .input = std::make_shared<gqe::equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0)),
                    .expected = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::less_equal_expression>(
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::greater_equal_expression>(
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)))},
    // value == column → value <= MAX AND value >= MIN
    test_parameters{.name  = "equalRhsColumn",
                    .input = std::make_shared<gqe::equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN)),
                    .expected = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::less_equal_expression>(
                        std::make_shared<gqe::literal_expression<double>>(1.0),
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN)),
                      std::make_shared<gqe::greater_equal_expression>(
                        std::make_shared<gqe::literal_expression<double>>(1.0),
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN)))},
    // lhs AND rhs
    // lhs: column < value → MIN < value
    // rhs: value < column → value < MAX
    test_parameters{.name  = "logicalAnd",
                    .input = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::literal_expression<double>>(2.0),
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN))),
                    .expected = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::literal_expression<double>>(2.0),
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN)))},
    // lhs OR rhs
    // lhs: column < value → MIN < value
    // rhs: value < column → value < MAX
    test_parameters{.name  = "logicalOr",
                    .input = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::literal_expression<double>>(2.0),
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN))),
                    .expected = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::literal_expression<double>>(2.0),
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN)))},
    // NOT (child) → NOT (transformed child with negation)
    // NOT (column < value) → NOT (MAX < value)
    test_parameters{
      .name     = "not",
      .input    = std::make_shared<gqe::not_expression>(std::make_shared<gqe::less_expression>(
        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
        std::make_shared<gqe::literal_expression<double>>(1.0))),
      .expected = std::make_shared<gqe::not_expression>(std::make_shared<gqe::less_expression>(
        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
        std::make_shared<gqe::literal_expression<double>>(1.0)))},
    // NOT (NOT (child)) → NOT (NOT (transformed child without negation))
    // NOT (NOT (column < value)) → NOT (NOT (MIN < value))
    test_parameters{.name  = "notNot",
                    .input = std::make_shared<gqe::not_expression>(
                      std::make_shared<gqe::not_expression>(std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)))),
                    .expected = std::make_shared<gqe::not_expression>(
                      std::make_shared<gqe::not_expression>(std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0))))},
    // The child of substr is transformed. Testing this by using the substr expression inside a
    // comparison. substr(column) < value → substr(MIN) < value
    test_parameters{
      .name  = "substr",
      .input = std::make_shared<gqe::less_expression>(
        std::make_shared<gqe::substr_expression>(
          std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN), 1, 2),
        std::make_shared<gqe::literal_expression<std::string>>("value")),
      .expected = std::make_shared<gqe::less_expression>(
        std::make_shared<gqe::substr_expression>(
          std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN), 1, 2),
        std::make_shared<gqe::literal_expression<std::string>>("value"))}),
  test_parameters::to_string);

// If any of the children of an AND expression are not supported, the other child can still be used
// to filter out partitions.
INSTANTIATE_TEST_SUITE_P(
  BinaryOperationsWithUnsupportedChild,
  ZoneMapExpressionTransformationTest,
  testing::Values(
    // lhs AND unsupported → lhs
    test_parameters{.name  = "andRhsUnsupported",
                    .input = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character")),
                    .expected = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // unsupported AND rhs → rhs
    test_parameters{.name  = "andLhsUnsupported",
                    .input = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0))),
                    .expected = std::make_shared<gqe::less_expression>(
                      std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))}),
  test_parameters::to_string);

INSTANTIATE_TEST_SUITE_P(
  RegressionTests,
  ZoneMapExpressionTransformationTest,
  testing::Values(
    // Q12 contains a predicate l_shipmode in ('MAIL', 'SHIP')
    // This is represented in the query plan as l_shipmode == 'MAIL' OR l_shipmode == 'SHIP'
    // The transformed expression looks like this:
    // MIN_l_shipmode <= 'MAIL' AND MAX_l_shipmode >= 'MAIL' OR MIN_l_shipmode <= 'MAIL'
    // I.e., the LHS of the OR predicate is not transformed correctly.
    test_parameters{.name  = "q12_IN_predicate",
                    .input = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::equal_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<std::string>>("MAIL")),
                      std::make_shared<gqe::equal_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<std::string>>("SHIP"))),
                    .expected = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::logical_and_expression>(
                        std::make_shared<gqe::less_equal_expression>(
                          std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                          std::make_shared<gqe::literal_expression<std::string>>("MAIL")),
                        std::make_shared<gqe::greater_equal_expression>(
                          std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
                          std::make_shared<gqe::literal_expression<std::string>>("MAIL"))),
                      std::make_shared<gqe::logical_and_expression>(
                        std::make_shared<gqe::less_equal_expression>(
                          std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MIN_COLUMN),
                          std::make_shared<gqe::literal_expression<std::string>>("SHIP")),
                        std::make_shared<gqe::greater_equal_expression>(
                          std::make_shared<gqe::column_reference_expression>(ZONE_MAP_MAX_COLUMN),
                          std::make_shared<gqe::literal_expression<std::string>>("SHIP"))))}),
  test_parameters::to_string);

class ZoneMapExpressionUnsupportedTransformationTest
  : public testing::TestWithParam<test_parameters> {};

TEST_P(ZoneMapExpressionUnsupportedTransformationTest, TestUnsupportedTranformation)
{
  auto const params = GetParam();
  assert(params.input != nullptr);
  assert(params.expected == nullptr);
  auto actual = gqe::zone_map_expression_transformer::transform(*params.input);
  EXPECT_FALSE(actual);
  if (actual) {
    GQE_LOG_DEBUG("Test: {}\nInput:\n{}\nExpected: nullptr\nActual: {}",
                  ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                  gqe::expression_json_formatter::to_json(*params.input),
                  gqe::expression_json_formatter::to_json(*actual));
  }
}

// Test transformations which remove an expression because it cannot be used for pruning
INSTANTIATE_TEST_SUITE_P(
  UnsupportedTransformationTests,
  ZoneMapExpressionUnsupportedTransformationTest,
  testing::Values(
    // column LIKE pattern
    test_parameters{.name  = "like",
                    .input = std::make_shared<gqe::like_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      "pattern",
                      "escape_character")},
    // unsupported AND unsupported → nullptr
    // If both children are unsupported return a nullptr for the AND expression. This should then
    // move up the binary logical expression tree until there is a binary expression with both
    // supported children.
    test_parameters{.name  = "andBothUnsupported",
                    .input = std::make_shared<gqe::logical_and_expression>(
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"),
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"))},
    // If any child of an OR expression is not supported, the entire expression is ignored
    // lhs OR unsupported → ignored
    test_parameters{.name  = "orRhsUnsupported",
                    .input = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)),
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"))},
    // unsupported OR rhs → ignored
    test_parameters{.name  = "orLhsUnsupported",
                    .input = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"),
                      std::make_shared<gqe::less_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        std::make_shared<gqe::literal_expression<double>>(1.0)))},
    // unsupported OR unsupported → ignored
    test_parameters{.name  = "orBothUnsupported",
                    .input = std::make_shared<gqe::logical_or_expression>(
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"),
                      std::make_shared<gqe::like_expression>(
                        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                        "pattern",
                        "escape_character"))},
    // Same logic for NOT expression
    test_parameters{
      .name  = "notChildUnsupported",
      .input = std::make_shared<gqe::not_expression>(std::make_shared<gqe::like_expression>(
        std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
        "pattern",
        "escape_character"))},
    // column <> value
    test_parameters{.name  = "notEqualLhsColumn",
                    .input = std::make_shared<gqe::not_equal_expression>(
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN),
                      std::make_shared<gqe::literal_expression<double>>(1.0))},
    // value <> column
    test_parameters{.name  = "notEqualRhsColumn",
                    .input = std::make_shared<gqe::not_equal_expression>(
                      std::make_shared<gqe::literal_expression<double>>(1.0),
                      std::make_shared<gqe::column_reference_expression>(INPUT_COLUMN))}),
  test_parameters::to_string);

namespace gqe {

// TODO Why doesn't this work?
// When a test fails it prints (instead of printing the partition in a human-readable form)
//
// Expected equality of these values:
//   partitions
//     Which is: { 12-byte object <2D-00 00-00 31-00 00-00 00-00 00-00>, 12-byte object <32-00 00-00
//     36-00 00-00 00-00 00-00>, 12-byte object <37-00 00-00 3B-00 00-00 00-00 00-00> }
// expected
//   Which is: { 12-byte object <2D-00 00-00 31-00 00-00 03-00 00-00>, 12-byte object <32-00 00-00
//   36-00 00-00 02-00 00-00>, 12-byte object <37-00 00-00 3B-00 00-00 03-00 00-00> }
template <typename Sink>
void AbslStringify(Sink& sink, const zone_map::partition& partition)
{
  absl::Format(&sink,
               "{.start = %d, .end = %d, .null_count = %d}",
               partition.start,
               partition.end,
               partition.null_counts);
}

}  // namespace gqe

// This test creates an input table, constructs a zone map on it, evaluates a partial filter on the
// zone map, and checks that the result are the expected partition boundaries and null counts.
TEST(ZoneMapTest, zoneMapWithTwoColumnsAndNullValues)
{
  // Create a table view consisting of two columns and 40 rows. The first column counts from 0
  // to 39. The second column also counts from 0 to 39 but every odd-indexed value is null: 0, null,
  // 2, null, 4, null, ..., 38, null
  constexpr cudf::size_type NUM_ROWS = 40;
  std::vector<int32_t> values(NUM_ROWS);
  std::iota(values.begin(), values.end(), 0);
  std::vector<bool> is_null(NUM_ROWS);
  std::transform(values.begin(), values.end(), is_null.begin(), [](auto i) { return i % 2 == 0; });
  cudf::test::fixed_width_column_wrapper<int32_t> col0(values.begin(), values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col1(
    values.begin(), values.end(), is_null.begin());
  cudf::table_view input_table{std::vector<cudf::column_view>{col0, col1}};

  // Create a zone map which partitions the table into zones of size 5.
  constexpr cudf::size_type PARTITION_SIZE = 5;
  gqe::zone_map zone_map{input_table, PARTITION_SIZE};

  // Create a filter expression to select values between 18 and 27 (inclusive). This should select
  // three consecutive partitions 15-20, 20-25, and 25-30 (start inclusive, end exclusive).
  constexpr cudf::size_type COL_INDEX = 0;
  expr_ptr partial_filter             = std::make_shared<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(COL_INDEX),
      std::make_shared<gqe::literal_expression<int32_t>>(18)),
    std::make_shared<gqe::less_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(COL_INDEX),
      std::make_shared<gqe::literal_expression<int32_t>>(27)));
  auto zone_map_filter = gqe::zone_map_expression_transformer::transform(*partial_filter);
  ASSERT_TRUE(zone_map_filter);
  partial_filter = zone_map_filter->clone();

  gqe::optimization_parameters const optimization_params{true};

  // Evaluate the filter expression on the zone map. The result should be 40/5 = 8 partitions.
  // The partitions described above are not pruned, the rest are pruned.
  // The null counts are {0, n}, i.e., the first column in each partition has no nulls and
  // the second column does have nulls (see below for number of nulls per partition).
  auto partitions = zone_map.evaluate(optimization_params, *partial_filter);
  std::vector<gqe::zone_map::partition> expected{
    {.pruned = true, .start = 0, .end = 5, .null_counts = {0, 2}},   // null values: 1, 3
    {.pruned = true, .start = 5, .end = 10, .null_counts = {0, 3}},  // null values: 5, 7, 9
    {.pruned = true, .start = 10, .end = 15, .null_counts = {0, 2}},
    {.pruned = false, .start = 15, .end = 20, .null_counts = {0, 3}},  // not pruned
    {.pruned = false, .start = 20, .end = 25, .null_counts = {0, 2}},  // not pruned
    {.pruned = false, .start = 25, .end = 30, .null_counts = {0, 3}},  // not pruned
    {.pruned = true, .start = 30, .end = 35, .null_counts = {0, 2}},
    {.pruned = true, .start = 35, .end = 40, .null_counts = {0, 3}}};
  EXPECT_EQ(partitions, expected);
}

TEST(ZoneMapTest, aggregateMaximallyCoveringPartition)
{
  // Input: 6 partitions; partitions 1, 3, and 4 are not pruned (counting from 0).
  std::vector<gqe::zone_map::partition> input = {
    {.pruned = true, .start = 0, .end = 5, .null_counts = {0, 2}},
    {.pruned = false, .start = 5, .end = 10, .null_counts = {0, 3}},
    {.pruned = true, .start = 10, .end = 15, .null_counts = {0, 2}},
    {.pruned = false, .start = 15, .end = 20, .null_counts = {0, 3}},
    {.pruned = false, .start = 20, .end = 25, .null_counts = {0, 2}},
    {.pruned = true, .start = 25, .end = 30, .null_counts = {0, 3}}};
  // Expected output: 1 partitions, representing input partition 1-4 (inclusive).
  gqe::zone_map::partition expected = {
    .pruned = false, .start = 5, .end = 25, .null_counts = {0, 10}};
  gqe::zone_map::partition actual = gqe::zone_map::consolidate_maximally_covering_partition(input);
  EXPECT_EQ(actual, expected);
}

TEST(ZoneMapTest, aggregatePartitions)
{
  // Input: 6 partitions; partitions 2, 4, and 5 are not pruned (counting from 0).
  std::vector<gqe::zone_map::partition> input = {
    {.pruned = true, .start = 0, .end = 5, .null_counts = {0, 2}},
    {.pruned = true, .start = 5, .end = 10, .null_counts = {0, 3}},
    {.pruned = false, .start = 10, .end = 15, .null_counts = {0, 2}},
    {.pruned = true, .start = 15, .end = 20, .null_counts = {0, 3}},
    {.pruned = false, .start = 20, .end = 25, .null_counts = {0, 2}},
    {.pruned = false, .start = 25, .end = 30, .null_counts = {0, 3}}};
  // Expected output: 4 partitions, alternating between pruned and unpruned
  std::vector<gqe::zone_map::partition> expected = {
    {.pruned = true, .start = 0, .end = 10, .null_counts = {0, 5}},
    {.pruned = false, .start = 10, .end = 15, .null_counts = {0, 2}},
    {.pruned = true, .start = 15, .end = 20, .null_counts = {0, 3}},
    {.pruned = false, .start = 20, .end = 30, .null_counts = {0, 5}}};
  std::vector<gqe::zone_map::partition> actual = gqe::zone_map::consolidate_partitions(input);
  EXPECT_EQ(actual, expected);
}

TEST(ZoneMapTest, aggregatePartitionsSinglePartition)
{
  // Input: 1 partition
  std::vector<gqe::zone_map::partition> input = {
    {.pruned = true, .start = 0, .end = 5, .null_counts = {0, 2}}};
  // Expected output: same partition
  std::vector<gqe::zone_map::partition> expected = {
    {.pruned = true, .start = 0, .end = 5, .null_counts = {0, 2}}};
  std::vector<gqe::zone_map::partition> actual = gqe::zone_map::consolidate_partitions(input);
  EXPECT_EQ(actual, expected);
}

TEST(ZoneMapTest, aggregatePartitionsEmptyPartition)
{
  // Input: no partitions
  std::vector<gqe::zone_map::partition> input = {};
  // Expected output: no partitions
  std::vector<gqe::zone_map::partition> expected = {};
  std::vector<gqe::zone_map::partition> actual   = gqe::zone_map::consolidate_partitions(input);
  EXPECT_EQ(actual, expected);
}
