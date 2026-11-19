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

#include <gqe/expression/literal.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf/sorting.hpp>

using namespace gqe_test;

/** @brief Exercises streamed conditional inner lowering with either broadcast side. */
class QepConditionalInnerJoinIntegrationTest : public QepJoinIntegrationTestBase {};

TEST_F(QepConditionalInnerJoinIntegrationTest, StreamedDuplicatePairsAndRepeatedProjections)
{
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (bool count : {false, true}) {
      std::vector<std::unique_ptr<cudf::table>> right;
      right.push_back(make_table_from_columns(int64_column_wrapper{200, 200}.release(),
                                              int64_column_wrapper{11, 11}.release(),
                                              int64_column_wrapper{1, 1}.release()));
      right.push_back(make_table_from_columns(int64_column_wrapper{350}.release(),
                                              int64_column_wrapper{33}.release(),
                                              int64_column_wrapper{2}.release()));
      auto condition = std::make_unique<gqe::less_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(3));
      auto result =
        execute_join(qep_join_test_configuration{.policy             = policy,
                                                 .join_type          = gqe::join_type_type::inner,
                                                 .is_perfect_hashing = false,
                                                 .is_hash_map_cache_enabled = false},
                     "conditional_inner_streamed",
                     qep_join_execution_options{
                       .right_partitions   = std::move(right),
                       .projection_indices = count ? std::vector<cudf::size_type>{}
                                                   : std::vector<cudf::size_type>{4, 0, 4},
                       .join_condition     = std::move(condition)});
      if (count) {
        EXPECT_EQ(qep::try_row_count(result), 5);
      } else {
        auto sorted = cudf::sort(*qep::to_table_view(result));
        auto expected =
          make_table_from_columns(int64_column_wrapper{11, 11, 33, 33, 33}.release(),
                                  int64_column_wrapper{100, 100, 100, 200, 300}.release(),
                                  int64_column_wrapper{11, 11, 33, 33, 33}.release());
        CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
      }
    }
  }
}

TEST_F(QepConditionalInnerJoinIntegrationTest, ProjectedOperandsAndFilteredInputs)
{
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    auto condition = std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::column_reference_expression>(7));
    auto filter = std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::literal_expression<int64_t>>(250));
    auto result = execute_join(qep_join_test_configuration{.policy    = policy,
                                                           .join_type = gqe::join_type_type::inner,
                                                           .is_perfect_hashing        = false,
                                                           .is_hash_map_cache_enabled = false},
                               "conditional_inner_project",
                               qep_join_execution_options{.has_input_sum_projection = true,
                                                          .projection_indices       = {0},
                                                          .left_filter    = std::move(filter),
                                                          .join_condition = std::move(condition)});
    auto sorted = cudf::sort(*qep::to_table_view(result));
    int64_column_wrapper expected{100, 100, 100, 100, 200, 200, 200, 200};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
  }
}
