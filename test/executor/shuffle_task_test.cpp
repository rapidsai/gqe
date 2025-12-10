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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/partition.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

class ShuffleTaskTest : public ::testing::Test {
 protected:
  ShuffleTaskTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  void construct_input_task(int32_t const stage_id, gqe::context_reference& ctx_ref)
  {
    constexpr int32_t input_task_id = 0;

    int64_column_wrapper input_col_0({1, 1, 3, 4, 9, 3, 7, 3, 9, 4});
    int64_column_wrapper input_col_1(
      {0, 12, 13, 0, 12, 16, 0, 16, 13, 20},
      {false, true, true, false, true, true, false, true, true, true});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());

    input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::test::executed_task> input_task;
};

TEST_F(ShuffleTaskTest, ShuffleOnNonNullColumns)
{
  constexpr int32_t stage_id = 0;

  construct_input_task(stage_id, ctx_ref);

  constexpr int32_t shuffle_task_id = 1;

  std::vector<std::unique_ptr<gqe::expression>> shuffle_cols;
  shuffle_cols.push_back(std::make_unique<gqe::column_reference_expression>(0));

  auto partition_task = std::make_unique<gqe::partition_task>(
    ctx_ref, shuffle_task_id, stage_id, std::move(input_task), std::move(shuffle_cols), 4);

  partition_task->execute();
  auto partition_result = partition_task->result();
  ASSERT_EQ(partition_result.has_value(), true);

  /**
   * partition 0 : start_idx=0, end_idx=6
   * partition 1 : start_idx=6, end_idx=8
   * partition 2 : start_idx=8, end_idx=8
   * partition 3 : start_idx=8, end_idx=10
   */
  int64_column_wrapper ref_result_col0({3, 9, 3, 7, 3, 9, 1, 1, 4, 4});
  int64_column_wrapper ref_result_col1(
    {13, 12, 16, 0, 16, 13, 0, 12, 0, 20},
    {true, true, true, false, true, true, false, true, false, true});
  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(partition_result.value(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(partition_result.value(), ref_result_table->view());
}

TEST_F(ShuffleTaskTest, ShuffleOnNullableColumns)
{
  constexpr int32_t stage_id = 0;

  construct_input_task(stage_id, ctx_ref);

  constexpr int32_t shuffle_task_id = 1;

  std::vector<std::unique_ptr<gqe::expression>> shuffle_cols;
  shuffle_cols.push_back(std::make_unique<gqe::column_reference_expression>(1));

  auto partition_task = std::make_unique<gqe::partition_task>(
    ctx_ref, shuffle_task_id, stage_id, std::move(input_task), std::move(shuffle_cols), 3);

  partition_task->execute();
  auto partition_result = partition_task->result();
  ASSERT_EQ(partition_result.has_value(), true);

  /**
   * partition 0 : start_idx=0, end_idx=2
   * partition 1 : start_idx=2, end_idx=6
   * partition 2 : start_idx=6, end_idx=10
   */
  int64_column_wrapper ref_result_col0({3, 9, 1, 9, 3, 3, 1, 4, 7, 4});
  int64_column_wrapper ref_result_col1(
    {13, 13, 12, 12, 16, 16, 0, 0, 0, 20},
    {true, true, true, true, true, true, false, false, false, true});
  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(partition_result.value(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(partition_result.value(), ref_result_table->view());
}
