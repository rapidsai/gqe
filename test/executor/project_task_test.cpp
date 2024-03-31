/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "utilities.hpp"

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

TEST(ProjectTaskTest, ReorderColumns)
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

  gqe::query_context qctx(gqe::optimization_parameters(true));

  auto input_task = std::make_shared<gqe::test::executed_task>(
    &qctx, input_task_id, stage_id, std::make_unique<cudf::table>(input_table));

  std::vector<std::unique_ptr<gqe::expression>> project_expressions;
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(2));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(2));

  auto project_task = std::make_unique<gqe::project_task>(
    &qctx, project_task_id, stage_id, std::move(input_task), std::move(project_expressions));

  project_task->execute();
  auto project_result = project_task->result();
  ASSERT_EQ(project_result.has_value(), true);

  cudf::table_view ref_table({input_table_columns[2]->view(),
                              input_table_columns[0]->view(),
                              input_table_columns[1]->view(),
                              input_table_columns[2]->view()});

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(project_result.value(), ref_table);
}
