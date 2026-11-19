/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include "join_task_test_util.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>

#include <cudf/sorting.hpp>

#include <limits>

namespace gqe_test {
namespace qep = gqe::qep;

/** @brief Logical join configuration shared by the implementation adapters. */
struct SemiAntiJoinTestOptions {
  gqe::join_type_type join_type = gqe::join_type_type::left_semi;
  std::vector<cudf::size_type> keys{0};
  cudf::null_equality null_equality = cudf::null_equality::UNEQUAL;
  std::vector<cudf::size_type> projections{1};
  std::vector<cudf::data_type> output_types{cudf::data_type{cudf::type_id::INT64}};
  std::vector<cudf::size_type> predicate_columns;
};

/**
 * @brief Shared numeric-key semi/anti expectations expressed in logical left/right terms.
 *
 * Adapters provide `execute(left, right, options)` and return materialized projected rows or
 * a row-count state when projections are empty. Operator lifecycles stay in the test `.cpp` files.
 *
 * @tparam Adapter A QEP join fixture exposing the common execution interface.
 */
template <typename Adapter>
class QepSemiAntiJoinContractTest : public Adapter {
 protected:
  void expect_output(qep::state_container const& output, cudf::table_view expected)
  {
    auto table = qep::to_table_view(output);
    ASSERT_TRUE(table.has_value());
    auto sorted          = cudf::sort(*table);
    auto sorted_expected = cudf::sort(expected);
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), sorted_expected->view());
  }
};

TYPED_TEST_SUITE_P(QepSemiAntiJoinContractTest);

TYPED_TEST_P(QepSemiAntiJoinContractTest, DuplicateMatchesAndLeftOnlyProjections)
{
  auto left       = two_int64_columns({1, 1, 2, 3}, {100, 110, 200, 300});
  auto right      = two_int64_columns({1, 1, 3}, {10, 11, 30});
  auto const type = cudf::data_type{cudf::type_id::INT64};
  SemiAntiJoinTestOptions options;
  options.projections  = {1, 0, 1};
  options.output_types = {type, type, type};
  int64_column_wrapper semi_values{100, 110, 300};
  int64_column_wrapper semi_keys{1, 1, 3};
  this->expect_output(this->execute(left, right, options),
                      cudf::table_view{{semi_values, semi_keys, semi_values}});

  options.join_type    = gqe::join_type_type::left_anti;
  options.projections  = {1, 0};
  options.output_types = {type, type};
  int64_column_wrapper anti_values{200};
  int64_column_wrapper anti_keys{2};
  this->expect_output(this->execute(left, right, options),
                      cudf::table_view{{anti_values, anti_keys}});
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, CompositeKeysHonorNullEqualityPolicy)
{
  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.push_back(int64_column_wrapper{{1, 2}, {true, false}}.release());
  right_columns.push_back(int32_column_wrapper{5, 6}.release());
  right_columns.push_back(int64_column_wrapper{10, 20}.release());
  auto right = table_state(std::move(right_columns));
  std::vector<std::unique_ptr<cudf::column>> left_columns;
  left_columns.push_back(int64_column_wrapper{{1, 9, 3}, {true, false, true}}.release());
  left_columns.push_back(int32_column_wrapper{5, 6, 7}.release());
  left_columns.push_back(int64_column_wrapper{100, 200, 300}.release());
  auto left = table_state(std::move(left_columns));

  SemiAntiJoinTestOptions options;
  options.keys        = {0, 1};
  options.projections = {2};
  for (auto nulls : {cudf::null_equality::UNEQUAL, cudf::null_equality::EQUAL}) {
    SCOPED_TRACE(nulls == cudf::null_equality::EQUAL);
    options.null_equality = nulls;
    auto expected         = nulls == cudf::null_equality::EQUAL ? int64_column_wrapper{100, 200}
                                                                : int64_column_wrapper{100};
    this->expect_output(this->execute(left, right, options), cudf::table_view{{expected}});
  }
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, MasksDiscardFalseAndNullRowsOnBothSides)
{
  auto right = masked_two_int64_columns(
    bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(),
    {1, 2, 3, 4},
    {10, 20, 30, 40});
  auto left = masked_two_int64_columns(
    bool_column_wrapper{{true, true, false, true}, {true, true, true, false}}.release(),
    {1, 2, 3, 4},
    {100, 200, 300, 400});
  SemiAntiJoinTestOptions options;
  int64_column_wrapper semi{100};
  this->expect_output(this->execute(left, right, options), cudf::table_view{{semi}});
  options.join_type = gqe::join_type_type::left_anti;
  int64_column_wrapper anti{200};
  this->expect_output(this->execute(left, right, options), cudf::table_view{{anti}});
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, EmptyLogicalInputs)
{
  auto input = two_int64_columns({1, 2}, {100, 200});
  auto empty = two_int64_columns({}, {});
  int64_column_wrapper empty_values{};
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(mode == gqe::join_type_type::left_anti);
    SemiAntiJoinTestOptions options;
    options.join_type = mode;
    auto expected     = mode == gqe::join_type_type::left_anti ? int64_column_wrapper{100, 200}
                                                               : int64_column_wrapper{};
    this->expect_output(this->execute(input, empty, options), cudf::table_view{{expected}});
    this->expect_output(this->execute(empty, input, options), cudf::table_view{{empty_values}});
    this->expect_output(this->execute(empty, empty, options), cudf::table_view{{empty_values}});
    options.projections.clear();
    options.output_types.clear();
    EXPECT_EQ(qep::try_row_count(this->execute(input, empty, options)),
              mode == gqe::join_type_type::left_anti ? 2 : 0);
    EXPECT_EQ(qep::try_row_count(this->execute(empty, input, options)), 0);
    EXPECT_EQ(qep::try_row_count(this->execute(empty, empty, options)), 0);
  }
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, CountOnlyReturnsMatchAndNonMatchCounts)
{
  auto left  = two_int64_columns({2, 3, 1, 1}, {200, 300, 100, 110});
  auto right = two_int64_columns({1, 2, 1}, {10, 20, 11});
  SemiAntiJoinTestOptions options;
  options.projections.clear();
  options.output_types.clear();
  EXPECT_EQ(qep::try_row_count(this->execute(left, right, options)), 3);
  options.join_type = gqe::join_type_type::left_anti;
  EXPECT_EQ(qep::try_row_count(this->execute(left, right, options)), 1);
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, ProjectedNoMatchOutputIsEmpty)
{
  auto left  = two_int64_columns({9}, {900});
  auto right = two_int64_columns({1, 2}, {10, 20});
  int64_column_wrapper expected{};
  this->expect_output(this->execute(left, right, SemiAntiJoinTestOptions{}),
                      cudf::table_view{{expected}});
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, CountOnlyExcludesMaskedRows)
{
  auto right = two_int64_columns({1, 3}, {10, 30});
  auto left  = masked_two_int64_columns(
    bool_column_wrapper{true, false, true, false}.release(), {1, 3, 2, 4}, {100, 300, 200, 400});
  // The mask removes one matching and one nonmatching row. Each mode retains one row.
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(mode == gqe::join_type_type::left_anti);
    SemiAntiJoinTestOptions options;
    options.join_type = mode;
    options.projections.clear();
    options.output_types.clear();
    EXPECT_EQ(qep::try_row_count(this->execute(left, right, options)), 1);
  }
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, CompositeNullableKeysHonorBothNullPolicies)
{
  std::vector<std::unique_ptr<cudf::column>> left_columns;
  left_columns.push_back(
    int64_column_wrapper{{99, 88, 1, 1, 1, 99}, {false, false, true, true, true, false}}.release());
  left_columns.push_back(int64_column_wrapper{10, 20, 30, 40, 50, 60}.release());
  left_columns.push_back(
    int32_column_wrapper{{2, 77, 2, 88, 3, 2}, {true, false, true, false, true, true}}.release());
  auto left = table_state(std::move(left_columns));
  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.push_back(
    int64_column_wrapper{{123, 456, 1, 1, 7}, {false, false, true, true, true}}.release());
  right_columns.push_back(int64_column_wrapper{0, 0, 0, 0, 0}.release());
  right_columns.push_back(
    int32_column_wrapper{{2, 999, 2, 888, 2}, {true, false, true, false, true}}.release());
  auto right = table_state(std::move(right_columns));
  for (auto nulls : {cudf::null_equality::UNEQUAL, cudf::null_equality::EQUAL}) {
    SCOPED_TRACE(nulls == cudf::null_equality::EQUAL);
    SemiAntiJoinTestOptions options;
    options.null_equality = nulls;
    options.keys          = {0, 2};
    auto semi             = this->execute(left, right, options);
    options.join_type     = gqe::join_type_type::left_anti;
    auto anti             = this->execute(left, right, options);
    if (nulls == cudf::null_equality::EQUAL) {
      this->expect_output(semi, cudf::table_view{{int64_column_wrapper{10, 20, 30, 40, 60}}});
      this->expect_output(anti, cudf::table_view{{int64_column_wrapper{50}}});
    } else {
      this->expect_output(semi, cudf::table_view{{int64_column_wrapper{30}}});
      this->expect_output(anti, cudf::table_view{{int64_column_wrapper{10, 20, 40, 50, 60}}});
    }
  }
}

TYPED_TEST_P(QepSemiAntiJoinContractTest, SignedZerosAndOneSidedNaNs)
{
  using doubles  = cudf::test::fixed_width_column_wrapper<double>;
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  // One-sided NaNs cannot match finite keys in any implementation. NaN-vs-NaN
  // equality differs between mark and cuDF-backed joins.
  for (bool nan_on_left : {false, true}) {
    SCOPED_TRACE(nan_on_left);
    std::vector<std::unique_ptr<cudf::column>> left_columns;
    left_columns.push_back(
      doubles{0.0, -0.0, nan_on_left ? nan : 2.0, nan_on_left ? nan : 3.0, 1.0}.release());
    left_columns.push_back(int64_column_wrapper{10, 20, 30, 40, 50}.release());
    auto left = table_state(std::move(left_columns));
    std::vector<std::unique_ptr<cudf::column>> right_columns;
    right_columns.push_back(
      doubles{-0.0, nan_on_left ? 4.0 : nan, nan_on_left ? 5.0 : nan}.release());
    right_columns.push_back(int64_column_wrapper{0, 0, 0}.release());
    auto right = table_state(std::move(right_columns));
    SemiAntiJoinTestOptions options;
    int64_column_wrapper semi{10, 20};
    this->expect_output(this->execute(left, right, options), cudf::table_view{{semi}});
    options.join_type = gqe::join_type_type::left_anti;
    int64_column_wrapper anti{30, 40, 50};
    this->expect_output(this->execute(left, right, options), cudf::table_view{{anti}});
  }
}

REGISTER_TYPED_TEST_SUITE_P(QepSemiAntiJoinContractTest,
                            DuplicateMatchesAndLeftOnlyProjections,
                            CompositeKeysHonorNullEqualityPolicy,
                            MasksDiscardFalseAndNullRowsOnBothSides,
                            EmptyLogicalInputs,
                            CountOnlyReturnsMatchAndNonMatchCounts,
                            ProjectedNoMatchOutputIsEmpty,
                            CountOnlyExcludesMaskedRows,
                            CompositeNullableKeysHonorBothNullPolicies,
                            SignedZerosAndOneSidedNaNs);

/**
 * @brief Shared residual-predicate expectations for implementations supporting mixed joins.
 * @tparam Adapter A join fixture whose execution helper also accepts a residual expression.
 */
template <typename Adapter>
class QepSemiAntiJoinResidualTest : public QepSemiAntiJoinContractTest<Adapter> {};

TYPED_TEST_SUITE_P(QepSemiAntiJoinResidualTest);

TYPED_TEST_P(QepSemiAntiJoinResidualTest, ResidualFalseAndNullDoNotMatch)
{
  auto left = two_int64_columns({1, 1, 2, 3}, {10, 20, 30, 40});
  std::vector<std::unique_ptr<cudf::column>> right_columns;
  right_columns.push_back(int64_column_wrapper{1, 1, 2, 3}.release());
  right_columns.push_back(
    int64_column_wrapper{{15, 5, 0, 0}, {true, true, false, false}}.release());
  auto right     = table_state(std::move(right_columns));
  auto predicate = [] {
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::column_reference_expression>(3));
  };
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(mode == gqe::join_type_type::left_anti);
    SemiAntiJoinTestOptions options;
    options.join_type         = mode;
    options.predicate_columns = {1};
    auto expected             = mode == gqe::join_type_type::left_semi ? int64_column_wrapper{10}
                                                                       : int64_column_wrapper{20, 30, 40};
    this->expect_output(this->execute(left, right, options, predicate()),
                        cudf::table_view{{expected}});
    options.projections.clear();
    options.output_types.clear();
    EXPECT_EQ(qep::try_row_count(this->execute(left, right, options, predicate())),
              mode == gqe::join_type_type::left_semi ? 1 : 3);
  }
}

REGISTER_TYPED_TEST_SUITE_P(QepSemiAntiJoinResidualTest, ResidualFalseAndNullDoNotMatch);

}  // namespace gqe_test
