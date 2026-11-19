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

#include "qep_join_integration_test_util.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>

#include <cudf/sorting.hpp>

using namespace gqe_test;
/** @brief Runs conditional semi/anti tasks through the partitioned QEP executor. */
class QepConditionalSemiAntiJoinIntegrationTest : public QepJoinIntegrationTestBase {};

TEST_F(QepConditionalSemiAntiJoinIntegrationTest, StreamedModesPreserveRepeatedLeftProjections)
{
  // Left values 100, 200, and 300 match a right value; only 400 survives anti selection.
  // Unequal cardinalities also detect a swapped mode in count-only execution.
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (bool count : {false, true}) {
      std::vector<std::unique_ptr<cudf::table>> right;
      right.push_back(make_table_from_columns(int64_column_wrapper{200, 200}.release(),
                                              int64_column_wrapper{0, 0}.release(),
                                              int64_column_wrapper{1, 1}.release()));
      right.push_back(make_table_from_columns(int64_column_wrapper{100, 350}.release(),
                                              int64_column_wrapper{0, 0}.release(),
                                              int64_column_wrapper{2, 4}.release()));
      auto condition = std::make_unique<gqe::less_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(3));
      auto result =
        execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                 .join_type = mode,
                                                 .is_perfect_hashing        = false,
                                                 .is_hash_map_cache_enabled = false},
                     "conditional_streamed",
                     qep_join_execution_options{
                       .right_partitions   = std::move(right),
                       .projection_indices = count ? std::vector<cudf::size_type>{}
                                                   : std::vector<cudf::size_type>{2, 0, 2},
                       .join_condition     = std::move(condition)});
      if (count) {
        EXPECT_EQ(qep::try_row_count(result), mode == gqe::join_type_type::left_semi ? 3 : 1);
        continue;
      }
      auto sorted   = cudf::sort(*qep::to_table_view(result));
      auto expected = mode == gqe::join_type_type::left_semi
                        ? make_table_from_columns(int64_column_wrapper{1000, 2000, 3000}.release(),
                                                  int64_column_wrapper{100, 200, 300}.release(),
                                                  int64_column_wrapper{1000, 2000, 3000}.release())
                        : make_table_from_columns(int64_column_wrapper{4000}.release(),
                                                  int64_column_wrapper{400}.release(),
                                                  int64_column_wrapper{4000}.release());
      CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
    }
  }
}

TEST_F(QepConditionalSemiAntiJoinIntegrationTest, ProjectedResidualOperandsAndFilteredLeftInput)
{
  auto condition = std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(3),
    std::make_shared<gqe::column_reference_expression>(7));
  auto filter =
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                           std::make_shared<gqe::literal_expression<int64_t>>(350));
  auto result =
    execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                             .join_type = gqe::join_type_type::left_semi,
                                             .is_perfect_hashing        = false,
                                             .is_hash_map_cache_enabled = false},
                 "conditional_project",
                 qep_join_execution_options{.has_input_sum_projection = true,
                                            .projection_indices       = {0},
                                            .left_filter              = std::move(filter),
                                            .join_condition           = std::move(condition)});
  auto sorted = cudf::sort(*qep::to_table_view(result));
  int64_column_wrapper expected{100, 200, 300};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(QepConditionalSemiAntiJoinIntegrationTest, AccountBalanceAboveStreamedAverageThenAggregate)
{
  // The scalar-average comparison in Q22 is independent of its mark/anti-join orientation.
  // Independent input relations avoid requiring diamond replay or mark-join support here.
  std::vector<std::unique_ptr<cudf::table>> left;
  left.push_back(make_table_from_columns(int64_column_wrapper{100, -10}.release(),
                                         int64_column_wrapper{13, 13}.release(),
                                         int64_column_wrapper{1, 2}.release()));
  left.push_back(make_table_from_columns(int64_column_wrapper{300, 500}.release(),
                                         int64_column_wrapper{31, 31}.release(),
                                         int64_column_wrapper{3, 4}.release()));
  std::vector<std::unique_ptr<cudf::table>> right;
  right.push_back(make_table_from_columns(int64_column_wrapper{100, 300}.release(),
                                          int64_column_wrapper{0, 0}.release(),
                                          int64_column_wrapper{1, 2}.release()));
  right.push_back(make_table_from_columns(int64_column_wrapper{500, 100}.release(),
                                          int64_column_wrapper{0, 0}.release(),
                                          int64_column_wrapper{3, 4}.release()));
  auto result = execute_join(
    qep_join_test_configuration{.policy                    = physical::broadcast_policy::right,
                                .join_type                 = gqe::join_type_type::left_semi,
                                .is_perfect_hashing        = false,
                                .is_hash_map_cache_enabled = false},
    "conditional_account_balance",
    qep_join_execution_options{
      .left_partitions    = std::move(left),
      .right_partitions   = std::move(right),
      .projection_indices = {0},
      .transform_root     = [](auto, auto accounts, auto average_input) {
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> mean;
        mean.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(0));
        mean.emplace_back(cudf::aggregation::COUNT_VALID,
                          std::make_unique<gqe::column_reference_expression>(0));
        // Use the existing planner's SUM/COUNT decomposition of AVG.
        auto partials = std::make_shared<physical::concatenate_aggregate_relation>(
          std::move(average_input),
          std::vector<std::shared_ptr<physical::relation>>{},
          std::vector<std::unique_ptr<gqe::expression>>{},
          std::move(mean));
        std::vector<std::unique_ptr<gqe::expression>> quotient;
        quotient.push_back(std::make_unique<gqe::divide_expression>(
          std::make_shared<gqe::cast_expression>(
            std::make_shared<gqe::column_reference_expression>(0),
            cudf::data_type{cudf::type_id::FLOAT64}),
          std::make_shared<gqe::cast_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            cudf::data_type{cudf::type_id::FLOAT64})));
        auto average = std::make_shared<physical::project_relation>(
          std::move(partials),
          std::vector<std::shared_ptr<physical::relation>>{},
          std::move(quotient));
        std::vector<std::unique_ptr<gqe::expression>> expressions;
        for (cudf::size_type index = 0; index < 3; ++index) {
          expressions.push_back(std::make_unique<gqe::column_reference_expression>(index));
        }
        expressions.push_back(std::make_unique<gqe::cast_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          cudf::data_type{cudf::type_id::FLOAT64}));
        auto projected = std::make_shared<physical::project_relation>(
          std::move(accounts),
          std::vector<std::shared_ptr<physical::relation>>{},
          std::move(expressions));
        auto selected = std::make_shared<physical::broadcast_join_relation>(
          std::move(projected),
          std::move(average),
          std::vector<std::shared_ptr<physical::relation>>{},
          gqe::join_type_type::left_semi,
          std::make_unique<gqe::greater_expression>(
            std::make_shared<gqe::column_reference_expression>(3),
            std::make_shared<gqe::column_reference_expression>(4)),
          std::vector<cudf::size_type>{1, 0},
          physical::broadcast_policy::right);
        std::vector<std::unique_ptr<gqe::expression>> groups;
        groups.push_back(std::make_unique<gqe::column_reference_expression>(0));
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> totals;
        totals.emplace_back(cudf::aggregation::COUNT_ALL,
                            std::make_unique<gqe::column_reference_expression>(1));
        totals.emplace_back(cudf::aggregation::SUM,
                            std::make_unique<gqe::column_reference_expression>(1));
        return std::make_shared<physical::concatenate_aggregate_relation>(
          std::move(selected),
          std::vector<std::shared_ptr<physical::relation>>{},
          std::move(groups),
          std::move(totals));
      }});
  auto expected = make_table_from_columns(int64_column_wrapper{31}.release(),
                                          int64_column_wrapper{2}.release(),
                                          int64_column_wrapper{800}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*qep::to_table_view(result), expected->view());
}
