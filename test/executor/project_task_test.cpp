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
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

class ProjectTest : public ::testing::Test {
 protected:
  ProjectTest()
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

TEST_F(ProjectTest, ReorderColumns)
{
  constexpr int32_t stage_id        = 0;
  constexpr int32_t input_task_id   = 0;
  constexpr int32_t project_task_id = 1;

  int64_column_wrapper ref_result_col0({0, 1, 2, 3, 4, 5, 6});
  int64_column_wrapper ref_result_col1({10, 11, 12, 13, 14, 15, 16});
  int64_column_wrapper ref_result_col2({20, 21, 22, 23, 24, 25, 26});

  std::vector<std::unique_ptr<cudf::column>> input_table_columns;
  input_table_columns.push_back(ref_result_col0.release());
  input_table_columns.push_back(ref_result_col1.release());
  input_table_columns.push_back(ref_result_col2.release());

  cudf::table_view input_table({input_table_columns[0]->view(),
                                input_table_columns[1]->view(),
                                input_table_columns[2]->view()});

  auto input_task = std::make_shared<gqe::test::executed_task>(
    ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(input_table));

  std::vector<std::unique_ptr<gqe::expression>> project_expressions;
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(2));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(2));

  auto project_task = std::make_unique<gqe::project_task>(
    ctx_ref, project_task_id, stage_id, std::move(input_task), std::move(project_expressions));

  project_task->execute();
  auto project_result = project_task->result();
  ASSERT_EQ(project_result.has_value(), true);

  cudf::table_view ref_table({input_table_columns[2]->view(),
                              input_table_columns[0]->view(),
                              input_table_columns[1]->view(),
                              input_table_columns[2]->view()});

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(project_result.value(), ref_table);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(project_result.value(), ref_table);
}
