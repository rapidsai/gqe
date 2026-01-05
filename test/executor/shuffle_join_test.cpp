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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/partition.hpp>
#include <gqe/executor/partition_merge.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <vector>

using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

/**
 * @brief Test fixture for shuffle join functionality.
 *
 * This test suite validates the shuffle join operation which involves:
 * 1. Partitioning left and right input tables based on join keys
 * 2. Merging corresponding partitions from left and right sides
 * 3. Performing local joins on merged partitions
 */
class ShuffleJoinTest : public ::testing::Test {
 protected:
  ShuffleJoinTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  /**
   * @brief Execute shuffle join with partition task on two sides
   */
  void execute_shuffle_join_two_sides(int32_t num_partitions = 3)
  {
    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;
    // Left table: ID, Value
    int64_column_wrapper left_id({1, 2, 3, 4, 5, 6});
    int64_column_wrapper left_value({100, 200, 300, 400, 500, 600});

    std::vector<std::unique_ptr<cudf::column>> left_columns;
    left_columns.push_back(left_id.release());
    left_columns.push_back(left_value.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_columns));

    // Right table: ID, Salary
    int64_column_wrapper right_id({2, 3, 4, 7, 8, 9});
    int64_column_wrapper right_salary({50000, 75000, 65000, 55000, 70000, 45000});

    std::vector<std::unique_ptr<cudf::column>> right_columns;
    right_columns.push_back(right_id.release());
    right_columns.push_back(right_salary.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_columns));

    // Create input tasks
    auto left_input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table));
    auto right_input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, right_task_id, stage_id, std::move(right_table));

    // Create partition columns expression (partition on first column - ID)
    std::vector<std::unique_ptr<gqe::expression>> left_partition_cols;
    left_partition_cols.push_back(std::make_unique<gqe::column_reference_expression>(0));

    std::vector<std::unique_ptr<gqe::expression>> right_partition_cols;
    right_partition_cols.push_back(std::make_unique<gqe::column_reference_expression>(0));

    // Create partition tasks
    auto left_partition_task = std::make_shared<gqe::partition_task>(ctx_ref,
                                                                     left_task_id,
                                                                     stage_id,
                                                                     left_input_task,
                                                                     std::move(left_partition_cols),
                                                                     num_partitions);
    auto right_partition_task =
      std::make_shared<gqe::partition_task>(ctx_ref,
                                            right_task_id,
                                            stage_id,
                                            right_input_task,
                                            std::move(right_partition_cols),
                                            num_partitions);

    // Execute partition tasks
    left_partition_task->execute();
    right_partition_task->execute();

    auto join_condition = std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2));

    std::vector<cudf::table_view> join_results;
    for (int32_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
      // Create merge tasks for this partition
      std::vector<std::shared_ptr<gqe::task>> left_partition_inputs = {
        std::static_pointer_cast<gqe::task>(left_partition_task)};
      std::vector<std::shared_ptr<gqe::task>> right_partition_inputs = {
        std::static_pointer_cast<gqe::task>(right_partition_task)};

      auto left_merge_task = std::make_shared<gqe::partition_merge_task>(
        ctx_ref, left_task_id, stage_id, left_partition_inputs, partition_idx);
      auto right_merge_task = std::make_shared<gqe::partition_merge_task>(
        ctx_ref, right_task_id, stage_id, right_partition_inputs, partition_idx);
      left_merge_task->execute();
      right_merge_task->execute();

      std::vector<cudf::size_type> projection_indices = {0, 1, 3};
      auto join_task                                  = std::make_unique<gqe::join_task>(ctx_ref,
                                                        join_task_id,
                                                        stage_id,
                                                        left_merge_task,
                                                        right_merge_task,
                                                        gqe::join_type_type::inner,
                                                        join_condition->clone(),
                                                        projection_indices);
      join_tasks.push_back(std::move(join_task));
    }
  }

  /**
   * @brief Execute shuffle join with partition task on one side
   */
  void execute_shuffle_join_one_side()
  {
    int32_t num_partitions          = 3;
    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;
    // create 3 left tables with already cudf::hash_partitioned based on its ID: ID, Value
    // the first one is has 0 rows
    int64_column_wrapper left_id0({});
    int64_column_wrapper left_value0({});
    std::vector<std::unique_ptr<cudf::column>> left_columns0;
    left_columns0.push_back(left_id0.release());
    left_columns0.push_back(left_value0.release());
    auto left_table0 = std::make_unique<cudf::table>(std::move(left_columns0));
    // the second one has 5 values
    int64_column_wrapper left_id1({2, 3, 4, 5, 6});
    int64_column_wrapper left_value1({200, 300, 400, 500, 600});
    std::vector<std::unique_ptr<cudf::column>> left_columns1;
    left_columns1.push_back(left_id1.release());
    left_columns1.push_back(left_value1.release());
    auto left_table1 = std::make_unique<cudf::table>(std::move(left_columns1));
    // the third one has 1 value
    int64_column_wrapper left_id2({1});
    int64_column_wrapper left_value2({100});
    std::vector<std::unique_ptr<cudf::column>> left_columns2;
    left_columns2.push_back(left_id2.release());
    left_columns2.push_back(left_value2.release());
    auto left_table2 = std::make_unique<cudf::table>(std::move(left_columns2));

    // Right table: ID, Salary
    int64_column_wrapper right_id({2, 3, 4, 7, 8, 9});
    int64_column_wrapper right_salary({50000, 75000, 65000, 55000, 70000, 45000});
    std::vector<std::unique_ptr<cudf::column>> right_columns;
    right_columns.push_back(right_id.release());
    right_columns.push_back(right_salary.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_columns));

    // Create input tasks
    auto left_input_task0 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table0));
    auto left_input_task1 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table1));
    auto left_input_task2 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table2));

    auto right_input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, right_task_id, stage_id, std::move(right_table));

    // Create partition columns expression (partition on first column - ID)
    std::vector<std::unique_ptr<gqe::expression>> right_partition_cols;
    right_partition_cols.push_back(std::make_unique<gqe::column_reference_expression>(0));

    auto right_partition_task =
      std::make_shared<gqe::partition_task>(ctx_ref,
                                            right_task_id,
                                            stage_id,
                                            right_input_task,
                                            std::move(right_partition_cols),
                                            num_partitions);

    // Execute partition tasks
    right_partition_task->execute();

    auto join_condition = std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2));

    std::vector<std::shared_ptr<gqe::task>> left_partition_inputs = {
      std::static_pointer_cast<gqe::task>(left_input_task0),
      std::static_pointer_cast<gqe::task>(left_input_task1),
      std::static_pointer_cast<gqe::task>(left_input_task2)};
    std::vector<cudf::table_view> join_results;
    for (int32_t partition_idx = 0; partition_idx < num_partitions; partition_idx++) {
      // Create merge tasks for this partition
      std::vector<std::shared_ptr<gqe::task>> right_partition_inputs = {
        std::static_pointer_cast<gqe::task>(right_partition_task)};
      auto right_merge_task = std::make_shared<gqe::partition_merge_task>(
        ctx_ref, right_task_id, stage_id, right_partition_inputs, partition_idx);
      right_merge_task->execute();

      std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};
      auto join_task                                  = std::make_unique<gqe::join_task>(ctx_ref,
                                                        join_task_id,
                                                        stage_id,
                                                        left_partition_inputs[partition_idx],
                                                        right_merge_task,
                                                        gqe::join_type_type::inner,
                                                        join_condition->clone(),
                                                        projection_indices);
      join_tasks.push_back(std::move(join_task));
    }
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::vector<std::unique_ptr<gqe::join_task>> join_tasks;
};

TEST_F(ShuffleJoinTest, ShuffleJoinTwoSidesShuffle)
{
  execute_shuffle_join_two_sides(3);
  std::vector<cudf::table_view> join_results;
  for (auto& join_task : join_tasks) {
    join_task->execute();
    if (join_task->result().has_value() && join_task->result().value().num_rows() > 0) {
      join_results.push_back(join_task->result().value());
    }
  }
  auto result        = cudf::concatenate(join_results);
  auto result_sorted = cudf::sort(result->view());

  int64_column_wrapper ref_result_col0({2, 3, 4});
  int64_column_wrapper ref_result_col1({200, 300, 400});
  int64_column_wrapper ref_result_col2({50000, 75000, 65000});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  auto ref_result_table        = std::make_unique<cudf::table>(std::move(ref_result_columns));
  auto ref_result_table_sorted = cudf::sort(ref_result_table->view());

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(result_sorted->view(), ref_result_table_sorted->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result_sorted->view(), ref_result_table_sorted->view());
}

TEST_F(ShuffleJoinTest, ShuffleJoinOneSideShuffle)
{
  execute_shuffle_join_one_side();
  std::vector<cudf::table_view> join_results;
  for (auto& join_task : join_tasks) {
    join_task->execute();
    if (join_task->result().has_value() && join_task->result().value().num_rows() > 0) {
      join_results.push_back(join_task->result().value());
    }
  }
  auto result        = cudf::concatenate(join_results);
  auto result_sorted = cudf::sort(result->view());

  int64_column_wrapper ref_result_col0({2, 3, 4});
  int64_column_wrapper ref_result_col1({200, 300, 400});
  int64_column_wrapper ref_result_col2({2, 3, 4});
  int64_column_wrapper ref_result_col3({50000, 75000, 65000});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  ref_result_columns.push_back(ref_result_col3.release());
  auto ref_result_table        = std::make_unique<cudf::table>(std::move(ref_result_columns));
  auto ref_result_table_sorted = cudf::sort(ref_result_table->view());

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(result_sorted->view(), ref_result_table_sorted->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result_sorted->view(), ref_result_table_sorted->view());
}
