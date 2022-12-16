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

#include <gqe/catalog.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

TEST(LogicalToExecution, HardcodePlanAndData)
{
  // Write two test Parquet files to the disk
  cudf::test::strings_column_wrapper table_0_col_0({"apple", "orange", "duck", "orange"});
  cudf::test::fixed_width_column_wrapper<int64_t> table_0_col_1({0, 1, 2, 3});

  std::vector<std::unique_ptr<cudf::column>> table_0_columns;
  table_0_columns.push_back(table_0_col_0.release());
  table_0_columns.push_back(table_0_col_1.release());
  auto table_0 = std::make_unique<cudf::table>(std::move(table_0_columns));

  cudf::io::table_input_metadata table_0_metadata(table_0->view());
  table_0_metadata.column_metadata[0].set_name("table_0_col_0");
  table_0_metadata.column_metadata[1].set_name("table_0_col_1");

  auto table_0_filepath = temp_env->get_temp_filepath("table_0.parquet");
  auto table_0_options  = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(table_0_filepath), table_0->view());
  table_0_options.metadata(&table_0_metadata);
  cudf::io::write_parquet(table_0_options);

  cudf::test::strings_column_wrapper table_1_col_0({"duck", "apple", "orange", "apple"});
  cudf::test::fixed_width_column_wrapper<int32_t> table_1_col_1({0, 1, 2, 3});

  std::vector<std::unique_ptr<cudf::column>> table_1_columns;
  table_1_columns.push_back(table_1_col_0.release());
  table_1_columns.push_back(table_1_col_1.release());
  auto table_1 = std::make_unique<cudf::table>(std::move(table_1_columns));

  cudf::io::table_input_metadata table_1_metadata(table_1->view());
  table_1_metadata.column_metadata[0].set_name("table_1_col_0");
  table_1_metadata.column_metadata[1].set_name("table_1_col_1");

  auto table_1_filepath = temp_env->get_temp_filepath("table_1.parquet");
  auto table_1_options  = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(table_1_filepath), table_1->view());
  table_1_options.metadata(&table_1_metadata);
  cudf::io::write_parquet(table_1_options);

  // Register the input tables
  gqe::catalog catalog;
  catalog.register_table("table_0",
                         {{"table_0_col_0", cudf::data_type(cudf::type_id::STRING)},
                          {"table_0_col_1", cudf::data_type(cudf::type_id::INT64)}},
                         {table_0_filepath},
                         gqe::file_format_type::parquet);
  catalog.register_table("table_1",
                         {{"table_1_col_0", cudf::data_type(cudf::type_id::STRING)},
                          {"table_1_col_1", cudf::data_type(cudf::type_id::INT32)}},
                         {table_1_filepath},
                         gqe::file_format_type::parquet);

  // Hand-code the logical plan
  auto read_relation_0 = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::string>({"table_0_col_0", "table_0_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::STRING), cudf::data_type(cudf::type_id::INT64)}),
    "table_0",
    nullptr);  // partial_filter

  auto read_relation_1 = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::string>({"table_1_col_0", "table_1_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::STRING), cudf::data_type(cudf::type_id::INT32)}),
    "table_1",
    nullptr);  // partial_filter

  auto join_condition =
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2));
  auto join_relation = std::make_shared<gqe::logical::join_relation>(
    read_relation_0,
    read_relation_1,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(join_condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({0, 1, 3}));

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(join_relation.get());

  // Generate the task graph and execute on a single GPU
  gqe::task_graph_builder graph_builder(&catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(task_graph.get());

  // Verify the execution result
  cudf::test::strings_column_wrapper ref_col_0({"apple", "apple", "duck", "orange", "orange"});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_1({0, 0, 2, 1, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_col_2({1, 3, 0, 2, 2});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  ref_columns.push_back(ref_col_1.release());
  ref_columns.push_back(ref_col_2.release());

  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  ASSERT_EQ(task_graph->root_tasks.size(), 1);
  auto execute_result = task_graph->root_tasks[0]->result();
  ASSERT_EQ(execute_result.has_value(), true);
  auto execute_result_sorted = cudf::sort(*execute_result);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(execute_result_sorted->view(), ref_table->view());
}
