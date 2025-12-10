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
#include <gqe/executor/sort.hpp>
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

class SingleKeyColumnSortTest : public ::testing::Test {
 protected:
  SingleKeyColumnSortTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_sort_task(cudf::order column_order, cudf::null_order null_precedence)
  {
    constexpr int32_t stage_id      = 0;
    constexpr int32_t input_task_id = 0;
    constexpr int32_t sort_task_id  = 1;

    int64_column_wrapper input_col_0({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    int64_column_wrapper input_col_1({5, 3, 4, 8, 1, 9, 6, 7, 2, 0},
                                     {true, true, true, true, true, true, true, false, true, true});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    std::vector<std::unique_ptr<gqe::expression>> keys;
    keys.push_back(std::make_unique<gqe::column_reference_expression>(1));

    sort_task = std::make_unique<gqe::sort_task>(ctx_ref,
                                                 sort_task_id,
                                                 stage_id,
                                                 std::move(input_task),
                                                 std::move(keys),
                                                 std::vector<cudf::order>({column_order}),
                                                 std::vector<cudf::null_order>({null_precedence}));
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::unique_ptr<gqe::sort_task> sort_task;
};

TEST_F(SingleKeyColumnSortTest, AscendingNullLast)
{
  construct_sort_task(cudf::order::ASCENDING, cudf::null_order::AFTER);

  sort_task->execute();
  auto sort_result = sort_task->result();
  ASSERT_EQ(sort_result.has_value(), true);

  int64_column_wrapper ref_col_0({9, 4, 8, 1, 2, 0, 6, 3, 5, 7});
  int64_column_wrapper ref_col_1({0, 1, 2, 3, 4, 5, 6, 8, 9, 7},
                                 {true, true, true, true, true, true, true, true, true, false});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(sort_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_result.value(), ref_table.view());
}

TEST_F(SingleKeyColumnSortTest, AscendingNullFirst)
{
  construct_sort_task(cudf::order::ASCENDING, cudf::null_order::BEFORE);

  sort_task->execute();
  auto sort_result = sort_task->result();
  ASSERT_EQ(sort_result.has_value(), true);

  int64_column_wrapper ref_col_0({7, 9, 4, 8, 1, 2, 0, 6, 3, 5});
  int64_column_wrapper ref_col_1({7, 0, 1, 2, 3, 4, 5, 6, 8, 9},
                                 {false, true, true, true, true, true, true, true, true, true});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(sort_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_result.value(), ref_table.view());
}

TEST_F(SingleKeyColumnSortTest, DescendingNullLast)
{
  construct_sort_task(cudf::order::DESCENDING, cudf::null_order::AFTER);

  sort_task->execute();
  auto sort_result = sort_task->result();
  ASSERT_EQ(sort_result.has_value(), true);

  int64_column_wrapper ref_col_0({7, 5, 3, 6, 0, 2, 1, 8, 4, 9});
  int64_column_wrapper ref_col_1({7, 9, 8, 6, 5, 4, 3, 2, 1, 0},
                                 {false, true, true, true, true, true, true, true, true, true});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(sort_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_result.value(), ref_table.view());
}
