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
#include <gqe/executor/concatenate.hpp>
#include <gqe/executor/optimization_parameters.hpp>
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

class ConcatenateTaskTest : public ::testing::Test {
 protected:
  ConcatenateTaskTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
};

TEST_F(ConcatenateTaskTest, MixTypes)
{
  // This unit test creates two input tables. Each of them has 3 columns with types int64, int32 and
  // string. All columns have hand-coded values. Then, a concatenate task with these input tables is
  // created and executed. The correctness is verified by comparing against the hand-coded reference
  // result.
  constexpr int32_t stage_id = 0;

  cudf::test::fixed_width_column_wrapper<int64_t> table_0_col_0({1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> table_0_col_1({6, 5});
  cudf::test::strings_column_wrapper table_0_col_2({"apple", "orange"});

  std::vector<std::unique_ptr<cudf::column>> table_0_columns;
  table_0_columns.push_back(table_0_col_0.release());
  table_0_columns.push_back(table_0_col_1.release());
  table_0_columns.push_back(table_0_col_2.release());
  auto table_0                      = std::make_unique<cudf::table>(std::move(table_0_columns));
  constexpr int32_t table_0_task_id = 0;
  auto table_0_task                 = std::make_shared<gqe::test::executed_task>(
    ctx_ref, table_0_task_id, stage_id, std::move(table_0));

  cudf::test::fixed_width_column_wrapper<int64_t> table_1_col_0({3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> table_1_col_1({5, 6});
  cudf::test::strings_column_wrapper table_1_col_2({"potato", "DGXA100"});

  std::vector<std::unique_ptr<cudf::column>> table_1_columns;
  table_1_columns.push_back(table_1_col_0.release());
  table_1_columns.push_back(table_1_col_1.release());
  table_1_columns.push_back(table_1_col_2.release());
  auto table_1                      = std::make_unique<cudf::table>(std::move(table_1_columns));
  constexpr int32_t table_1_task_id = 1;
  auto table_1_task                 = std::make_shared<gqe::test::executed_task>(
    ctx_ref, table_1_task_id, stage_id, std::move(table_1));

  std::vector<std::shared_ptr<gqe::task>> inputs;
  inputs.push_back(std::move(table_0_task));
  inputs.push_back(std::move(table_1_task));

  constexpr int32_t concatenate_task_id = 2;
  auto concatenate_task                 = std::make_unique<gqe::concatenate_task>(
    ctx_ref, concatenate_task_id, stage_id, std::move(inputs));
  concatenate_task->execute();

  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_0({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_col_1({6, 5, 5, 6});
  cudf::test::strings_column_wrapper ref_col_2({"apple", "orange", "potato", "DGXA100"});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  auto concatenate_task_result = concatenate_task->result();
  ASSERT_EQ(concatenate_task_result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(concatenate_task_result.value(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(concatenate_task_result.value(), ref_table->view());
}
