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
#include <gqe/executor/fetch.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

class FetchTest : public ::testing::Test {
 protected:
  FetchTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  void construct_fetch_task(int32_t offset, int32_t count)
  {
    constexpr int32_t stage_id      = 0;
    constexpr int32_t input_task_id = 0;
    constexpr int32_t fetch_task_id = 1;

    int64_column_wrapper input_col_0({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    int64_column_wrapper input_col_1({5, 3, 4, 8, 1, 9, 6, 7, 2, 0});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());

    auto input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));

    fetch_task = std::make_unique<gqe::fetch_task>(
      ctx_ref, fetch_task_id, stage_id, std::move(input_task), offset, count);
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::unique_ptr<gqe::fetch_task> fetch_task;
};

TEST_F(FetchTest, NormalFetch)
{
  construct_fetch_task(1, 4);

  fetch_task->execute();
  auto fetch_result = fetch_task->result();
  ASSERT_EQ(fetch_result.has_value(), true);

  int64_column_wrapper ref_col_0({1, 2, 3, 4});
  int64_column_wrapper ref_col_1({3, 4, 8, 1});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(fetch_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(fetch_result.value(), ref_table.view());
}

TEST_F(FetchTest, FetchToEnd)
{
  construct_fetch_task(1, 100);

  fetch_task->execute();
  auto fetch_result = fetch_task->result();
  ASSERT_EQ(fetch_result.has_value(), true);

  int64_column_wrapper ref_col_0({1, 2, 3, 4, 5, 6, 7, 8, 9});
  int64_column_wrapper ref_col_1({3, 4, 8, 1, 9, 6, 7, 2, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(fetch_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(fetch_result.value(), ref_table.view());
}

TEST_F(FetchTest, InvalidFetch)
{
  construct_fetch_task(100, 1);

  fetch_task->execute();
  auto fetch_result = fetch_task->result();
  ASSERT_EQ(fetch_result.has_value(), true);

  int64_column_wrapper ref_col_0({});
  int64_column_wrapper ref_col_1({});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(fetch_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(fetch_result.value(), ref_table.view());
}

TEST_F(FetchTest, OverflowFetch)
{
  construct_fetch_task(std::numeric_limits<cudf::size_type>::max() - 10, 20);

  EXPECT_THROW(fetch_task->execute(), std::overflow_error);
}
