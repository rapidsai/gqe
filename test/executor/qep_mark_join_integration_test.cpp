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
#include <gqe/physical/aggregate.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/sorting.hpp>

namespace gqe_test {

/** @brief Executes normalized mark joins against partitioned inputs. */
class QepMarkJoinIntegrationTest : public QepJoinIntegrationTestBase {};

TEST_F(QepMarkJoinIntegrationTest, RuntimeMarkSelectionHonorsBroadcastSide)
{
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (bool const runtime_mark : {true, false}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(runtime_mark);
      get_query_ctx()->parameters.join_use_mark_join = runtime_mark;
      // Deliberately disagree with the runtime setting to cover handwritten/stale metadata.
      auto result = execute_join({.policy    = runtime_mark ? physical::broadcast_policy::left
                                                            : physical::broadcast_policy::right,
                                  .join_type = mode,
                                  .is_perfect_hashing        = false,
                                  .is_hash_map_cache_enabled = false,
                                  .is_mark_join              = !runtime_mark},
                                 "runtime_mark_selection",
                                 {.projection_indices = {0}});
      auto table  = qep::to_table_view(result);
      ASSERT_TRUE(table.has_value());
      auto sorted   = cudf::sort(*table);
      auto expected = mode == gqe::join_type_type::left_semi
                        ? int64_column_wrapper{100, 200, 400}.release()
                        : int64_column_wrapper{300}.release();
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected->view());
    }
    get_query_ctx()->parameters.join_use_mark_join = false;
    EXPECT_THROW(std::ignore = execute_join({.policy             = physical::broadcast_policy::left,
                                             .join_type          = mode,
                                             .is_perfect_hashing = false,
                                             .is_hash_map_cache_enabled = false,
                                             .is_mark_join              = true},
                                            "disabled_mark_left_build",
                                            {.projection_indices = {0}}),
                 std::logic_error);
  }
}

TEST_F(QepMarkJoinIntegrationTest, CountOnly)
{
  auto result = execute_join({.policy                    = physical::broadcast_policy::left,
                              .join_type                 = gqe::join_type_type::left_anti,
                              .is_perfect_hashing        = false,
                              .is_hash_map_cache_enabled = true,
                              .is_mark_join              = true},
                             "mark_count",
                             {.projection_indices = {}});
  EXPECT_EQ(qep::try_row_count(result), 1);
}

TEST_F(QepMarkJoinIntegrationTest, AllProbeRowsFilteredOut)
{
  auto false_filter =
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                           std::make_shared<gqe::column_reference_expression>(2));
  auto result = execute_join({.policy                    = physical::broadcast_policy::left,
                              .join_type                 = gqe::join_type_type::left_anti,
                              .is_perfect_hashing        = false,
                              .is_hash_map_cache_enabled = true,
                              .is_mark_join              = true},
                             "filtered_mark",
                             {.projection_indices = {0}, .right_filter = std::move(false_filter)});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted = cudf::sort(*table);
  int64_column_wrapper expected{100, 200, 300, 400};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(QepMarkJoinIntegrationTest, UpstreamFiltersApplyOnBothSides)
{
  for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto left_filter = std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(1));
    auto right_filter = std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::literal_expression<int64_t>>(20));
    auto result = execute_join({.policy                    = physical::broadcast_policy::left,
                                .join_type                 = type,
                                .is_perfect_hashing        = false,
                                .is_hash_map_cache_enabled = false,
                                .is_mark_join              = true},
                               "upstream_input_filters",
                               {.projection_indices = {0},
                                .left_filter        = std::move(left_filter),
                                .right_filter       = std::move(right_filter)});
    auto table  = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = type == gqe::join_type_type::left_semi
                      ? int64_column_wrapper{400}.release()
                      : int64_column_wrapper{200, 300}.release();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected->view());
  }
}

TEST_F(QepMarkJoinIntegrationTest, ConsumesMixedHashJoinOutputOnEitherSide)
{
  for (bool const indirect_build : {false, true}) {
    for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
      SCOPED_TRACE(indirect_build);
      SCOPED_TRACE(type == gqe::join_type_type::left_anti);
      auto condition = std::make_unique<gqe::logical_and_expression>(
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(1),
          std::make_shared<gqe::column_reference_expression>(5)),
        std::make_shared<gqe::greater_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::literal_expression<int64_t>>(150)));
      auto result = execute_join(
        {.policy                    = physical::broadcast_policy::right,
         .join_type                 = gqe::join_type_type::inner,
         .is_perfect_hashing        = false,
         .is_hash_map_cache_enabled = true},
        "mixed_hash_to_mark",
        {.projection_indices = {0, 1},
         .join_condition     = std::move(condition),
         .transform_root     = [=](std::shared_ptr<physical::relation> join,
                               std::shared_ptr<physical::relation>,
                               std::shared_ptr<physical::relation>) {
           auto other = make_join_read("left", {"left_value", "join_key", "left_payload"});
           auto left  = indirect_build ? join : other;
           auto right = indirect_build ? other : join;
           auto keys  = std::make_unique<gqe::equal_expression>(
             std::make_shared<gqe::column_reference_expression>(1),
             std::make_shared<gqe::column_reference_expression>(indirect_build ? 3 : 4));
           return std::make_shared<physical::broadcast_join_relation>(
             std::move(left),
             std::move(right),
             std::vector<std::shared_ptr<physical::relation>>{},
             type,
             std::move(keys),
             std::vector<cudf::size_type>{0},
             physical::broadcast_policy::left,
             gqe::unique_keys_policy::none,
             false,
             nullptr,
             nullptr,
             true,
             true);
         }});
      auto table = qep::to_table_view(result);
      ASSERT_TRUE(table.has_value());
      auto sorted   = cudf::sort(*table);
      auto expected = type == gqe::join_type_type::left_semi
                        ? int64_column_wrapper{200, 400}.release()
                        : (indirect_build ? int64_column_wrapper{}.release()
                                          : int64_column_wrapper{100, 300}.release());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected->view());
    }
  }
}

TEST_F(QepMarkJoinIntegrationTest, LoweringPreservesArithmeticAndDisjunction)
{
  auto condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(5)),
    std::make_shared<gqe::logical_or_expression>(
      std::make_shared<gqe::greater_expression>(
        std::make_shared<gqe::add_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(3)),
        std::make_shared<gqe::literal_expression<int64_t>>(225)),
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::literal_expression<int64_t>>(100))));
  auto result = execute_join({.policy                    = physical::broadcast_policy::left,
                              .join_type                 = gqe::join_type_type::left_semi,
                              .is_perfect_hashing        = false,
                              .is_hash_map_cache_enabled = true,
                              .is_mark_join              = true},
                             "arithmetic_or",
                             {.projection_indices = {0}, .join_condition = std::move(condition)});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted = cudf::sort(*table);
  int64_column_wrapper expected{100, 400};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(QepMarkJoinIntegrationTest, DownstreamReductionCombinesScanPartitions)
{
  for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(type == gqe::join_type_type::left_anti);
    auto result = execute_join(
      {.policy                    = physical::broadcast_policy::left,
       .join_type                 = type,
       .is_perfect_hashing        = false,
       .is_hash_map_cache_enabled = true,
       .is_mark_join              = true},
      "mark_reduce",
      {.projection_indices = {0},
       .transform_root     = [](std::shared_ptr<physical::relation> join,
                            std::shared_ptr<physical::relation>,
                            std::shared_ptr<physical::relation>) {
         std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
         values.emplace_back(cudf::aggregation::SUM,
                             std::make_unique<gqe::column_reference_expression>(0));
         values.emplace_back(cudf::aggregation::COUNT_ALL,
                             std::make_unique<gqe::column_reference_expression>(0));
         return std::make_shared<physical::concatenate_aggregate_relation>(
           join,
           std::vector<std::shared_ptr<physical::relation>>{},
           std::vector<std::unique_ptr<gqe::expression>>{},
           std::move(values));
       }});
    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    // Semi emits {100} and {200, 400}; anti emits {300} and an empty partition.
    bool const is_anti = type == gqe::join_type_type::left_anti;
    auto expected = make_table_from_columns(int64_column_wrapper{is_anti ? 300 : 700}.release(),
                                            int64_column_wrapper{is_anti ? 1 : 3}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
  }
}

}  // namespace gqe_test
