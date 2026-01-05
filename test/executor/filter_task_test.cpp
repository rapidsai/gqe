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
#include <gqe/executor/filter.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

class FilterTest : public ::testing::Test {
 protected:
  FilterTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  void construct_input_task(int32_t const stage_id, gqe::context_reference& ctx_ref)
  {
    constexpr int32_t input_task_id = 0;

    int64_column_wrapper input_col_0({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    int64_column_wrapper input_col_1({11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    bool_column_wrapper input_col_2(
      {true, false, true, false, true, false, true, false, true, false});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());
    input_columns.push_back(input_col_2.release());

    input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(std::move(input_columns)));
  }

  void construct_filter_task(std::vector<cudf::size_type> projection_indices)
  {
    constexpr int32_t stage_id       = 0;
    constexpr int32_t filter_task_id = 1;

    construct_input_task(stage_id, ctx_ref);

    std::unique_ptr<gqe::expression> condition{
      std::make_unique<gqe::column_reference_expression>(2)};

    filter_task = std::make_unique<gqe::filter_task>(ctx_ref,
                                                     filter_task_id,
                                                     stage_id,
                                                     std::move(input_task),
                                                     std::move(condition),
                                                     projection_indices);
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::test::executed_task> input_task;
  std::unique_ptr<gqe::filter_task> filter_task;
};

TEST_F(FilterTest, FilterOddsOnly)
{
  std::vector<cudf::size_type> projection_indices{0, 1, 2};
  construct_filter_task(projection_indices);

  filter_task->execute();
  auto filter_result = filter_task->result();
  ASSERT_EQ(filter_result.has_value(), true);

  int64_column_wrapper ref_col_0({1, 3, 5, 7, 9});
  int64_column_wrapper ref_col_1({11, 13, 15, 17, 19});
  bool_column_wrapper ref_col_2({true, true, true, true, true});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(filter_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(filter_result.value(), ref_table.view());
}

TEST_F(FilterTest, MaterializeSubsetOfColumns)
{
  std::vector<cudf::size_type> projection_indices{1};
  construct_filter_task(projection_indices);

  filter_task->execute();
  auto filter_result = filter_task->result();
  ASSERT_EQ(filter_result.has_value(), true);

  int64_column_wrapper ref_col_0({11, 13, 15, 17, 19});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());

  cudf::table ref_table(std::move(ref_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(filter_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(filter_result.value(), ref_table.view());
}

TEST_F(FilterTest, EmptyProjectionIndices)
{
  std::vector<cudf::size_type> projection_indices{};
  construct_filter_task(projection_indices);

  filter_task->execute();
  auto filter_result = filter_task->result();
  ASSERT_EQ(filter_result.has_value(), true);

  cudf::table ref_table;

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(filter_result.value(), ref_table.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(filter_result.value(), ref_table.view());
}
