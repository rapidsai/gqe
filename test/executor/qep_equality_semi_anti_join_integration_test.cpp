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

#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe_test {

/** @brief Executes equality semi/anti joins over partitioned Parquet inputs. */
class QepEqualitySemiAntiJoinIntegrationTest : public QepJoinIntegrationTestBase {
 protected:
  qep::state_container execute_semi_anti(gqe::join_type_type join_type,
                                         std::string_view name,
                                         std::vector<cudf::size_type> projections)
  {
    return execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                    .join_type = join_type,
                                                    .is_perfect_hashing        = false,
                                                    .is_hash_map_cache_enabled = true},
                        name,
                        qep_join_execution_options{.projection_indices = std::move(projections)});
  }
};

TEST_F(QepEqualitySemiAntiJoinIntegrationTest, PartitionedLeftSemiProjectsOnlyLogicalLeft)
{
  auto result = execute_semi_anti(gqe::join_type_type::left_semi, "left_semi", {2, 0, 2});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(int64_column_wrapper{1000, 2000, 4000}.release(),
                                          int64_column_wrapper{100, 200, 400}.release(),
                                          int64_column_wrapper{1000, 2000, 4000}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
}

TEST_F(QepEqualitySemiAntiJoinIntegrationTest, PartitionedLeftAntiProjectsOnlyLogicalLeft)
{
  auto result = execute_semi_anti(gqe::join_type_type::left_anti, "left_anti", {2, 0});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = make_table_from_columns(int64_column_wrapper{3000}.release(),
                                          int64_column_wrapper{300}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
}

TEST_F(QepEqualitySemiAntiJoinIntegrationTest, CountOnlyOutputsSurviveTaskGraphBridge)
{
  auto semi = execute_semi_anti(gqe::join_type_type::left_semi, "semi_count", {});
  EXPECT_EQ(qep::try_row_count(semi), 3);
  auto anti = execute_semi_anti(gqe::join_type_type::left_anti, "anti_count", {});
  EXPECT_EQ(qep::try_row_count(anti), 1);
}

TEST_F(QepEqualitySemiAntiJoinIntegrationTest, MaskedChildrenFeedLogicalJoinInputs)
{
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{10, 20, 30}.release(),
                            int64_column_wrapper{1, 2, 3}.release(),
                            int64_column_wrapper{{100, 10, 0}, {true, true, false}}.release()));
  std::vector<std::unique_ptr<cudf::table>> right_partitions;
  right_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{10, 20, 30}.release(),
                            int64_column_wrapper{{100, 10, 0}, {true, true, false}}.release(),
                            int64_column_wrapper{1, 2, 3}.release()));

  auto result = execute_join(
    qep_join_test_configuration{.policy             = physical::broadcast_policy::right,
                                .join_type          = gqe::join_type_type::left_semi,
                                .is_perfect_hashing = false,
                                // Equality semi/anti lowering always builds a reusable filtered
                                // join and ignores this flag.
                                .is_hash_map_cache_enabled = true},
    "masked_semi",
    qep_join_execution_options{.left_partitions    = std::move(left_partitions),
                               .right_partitions   = std::move(right_partitions),
                               .projection_indices = {2, 0, 2},
                               .left_filter        = std::make_unique<gqe::less_expression>(
                                 std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(2)),
                               .right_filter = std::make_unique<gqe::less_expression>(
                                 std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(1))});
  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = make_table_from_columns(int64_column_wrapper{{100}, {true}}.release(),
                                          int64_column_wrapper{10}.release(),
                                          int64_column_wrapper{{100}, {true}}.release());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*table, expected->view());
}

}  // namespace gqe_test
