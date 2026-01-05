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

#include <gqe/catalog.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class LogicalToExecution : public ::testing::Test {
 protected:
  LogicalToExecution()
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

TEST_F(LogicalToExecution, HardcodePlanAndData)
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
  table_0_options.metadata(table_0_metadata);
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
  table_1_options.metadata(table_1_metadata);
  cudf::io::write_parquet(table_1_options);

  // Register the input tables
  gqe::catalog catalog{&task_manager_ctx};
  catalog.register_table("table_0",
                         {{"table_0_col_0", cudf::data_type(cudf::type_id::STRING)},
                          {"table_0_col_1", cudf::data_type(cudf::type_id::INT64)}},
                         gqe::storage_kind::parquet_file{{table_0_filepath}},
                         gqe::partitioning_schema_kind::automatic{});
  catalog.register_table("table_1",
                         {{"table_1_col_0", cudf::data_type(cudf::type_id::STRING)},
                          {"table_1_col_1", cudf::data_type(cudf::type_id::INT32)}},
                         gqe::storage_kind::parquet_file{{table_1_filepath}},
                         gqe::partitioning_schema_kind::automatic{});

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
  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

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

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(execute_result_sorted->view(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(execute_result_sorted->view(), ref_table->view());
}

TEST_F(LogicalToExecution, ApplyConcatApply)
{
  auto generate_parquet_file = [](std::string file_name, std::vector<double> values) {
    cudf::test::fixed_width_column_wrapper<double> partition_values(values.begin(), values.end());

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(partition_values.release());
    auto partition_table = std::make_unique<cudf::table>(std::move(columns));

    cudf::io::table_input_metadata partition_metadata(partition_table->view());
    partition_metadata.column_metadata[0].set_name("values");

    auto partition_filepath = temp_env->get_temp_filepath(file_name);
    auto partition_options  = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(partition_filepath), partition_table->view());
    partition_options.metadata(partition_metadata);
    cudf::io::write_parquet(partition_options);
  };

  generate_parquet_file("partition_0.parquet", std::vector<double>({1.0, 2.0, 3.0}));
  generate_parquet_file("partition_1.parquet", std::vector<double>({4.0, 5.0}));

  // Register the input tables
  gqe::catalog catalog{&task_manager_ctx};
  catalog.register_table(
    "input",
    {{"values", cudf::data_type(cudf::type_id::FLOAT64)}},
    gqe::storage_kind::parquet_file{{temp_env->get_temp_filepath("partition_0.parquet"),
                                     temp_env->get_temp_filepath("partition_1.parquet")}},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  auto read_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::string>({"values"}),
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::FLOAT64)}),
    "input",
    nullptr  // partial_filter
  );

  std::vector<std::pair<cudf::aggregation::Kind, double>> tests = {{cudf::aggregation::MEAN, 3.0},
                                                                   {cudf::aggregation::SUM, 15.0},
                                                                   {cudf::aggregation::MIN, 1.0},
                                                                   {cudf::aggregation::MAX, 5.0}};
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  std::vector<std::unique_ptr<cudf::column>> ref_columns;

  for (auto [measure, result] : tests) {
    measures.emplace_back(measure, std::make_unique<gqe::column_reference_expression>(0));
    ref_columns.push_back(cudf::test::fixed_width_column_wrapper<double>({result}).release());
  }

  auto aggregate_relation = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(read_relation),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::unique_ptr<gqe::expression>>(),         // keys
    std::move(measures));

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(aggregate_relation.get());

  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Compare against reference result
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  ASSERT_EQ(task_graph->root_tasks.size(), 1);
  auto execute_result = task_graph->root_tasks[0]->result();
  ASSERT_EQ(execute_result.has_value(), true);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(*execute_result, ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*execute_result, ref_table->view());
}

TEST_F(LogicalToExecution, Window)
{
  auto generate_parquet_file = [](std::string file_name) {
    std::vector<int> arg_col{1, 2, 3};
    std::vector<int> partition_col0{1, 2, 1};
    std::vector<int> partition_col1{0, 0, 0};
    cudf::test::fixed_width_column_wrapper<int> arg_values(arg_col.begin(), arg_col.end());
    cudf::test::fixed_width_column_wrapper<int> partition_values0(partition_col0.begin(),
                                                                  partition_col0.end());
    cudf::test::fixed_width_column_wrapper<int> partition_values1(partition_col1.begin(),
                                                                  partition_col1.end());

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(partition_values0.release());
    columns.push_back(partition_values1.release());
    columns.push_back(arg_values.release());
    auto partition_table = std::make_unique<cudf::table>(std::move(columns));

    cudf::io::table_input_metadata partition_metadata(partition_table->view());
    partition_metadata.column_metadata[0].set_name("keys0");
    partition_metadata.column_metadata[1].set_name("keys1");
    partition_metadata.column_metadata[2].set_name("values");

    auto partition_filepath = temp_env->get_temp_filepath(file_name);
    auto partition_options  = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(partition_filepath), partition_table->view());
    partition_options.metadata(partition_metadata);
    cudf::io::write_parquet(partition_options);
  };

  generate_parquet_file("partition_0.parquet");

  // Register the input tables
  gqe::catalog catalog{&task_manager_ctx};
  catalog.register_table(
    "input",
    {{"keys0", cudf::data_type(cudf::type_id::INT32)},
     {"keys1", cudf::data_type(cudf::type_id::INT32)},
     {"values", cudf::data_type(cudf::type_id::INT32)}},
    gqe::storage_kind::parquet_file{{temp_env->get_temp_filepath("partition_0.parquet")}},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  auto read_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::string>({"keys0", "keys1", "values"}),
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::INT32),
                                  cudf::data_type(cudf::type_id::INT32),
                                  cudf::data_type(cudf::type_id::INT32)}),
    "input",
    nullptr  // partial_filter
  );

  std::vector<std::unique_ptr<gqe::expression>> arguments;
  arguments.emplace_back(std::make_unique<gqe::column_reference_expression>(2));

  std::vector<std::unique_ptr<gqe::expression>> partition_by;
  partition_by.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
  partition_by.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

  gqe::window_frame_bound::unbounded window_lower_bound;
  gqe::window_frame_bound::unbounded window_upper_bound;

  auto window_relation = std::make_shared<gqe::logical::window_relation>(
    std::move(read_relation),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    cudf::aggregation::SUM,
    std::move(arguments),
    std::vector<std::unique_ptr<gqe::expression>>(),
    std::move(partition_by),
    std::vector<cudf::order>(),
    window_lower_bound,
    window_upper_bound);

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(window_relation.get());

  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Compare against reference result
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c0({1, 1, 2});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c1({0, 0, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c2({1, 3, 2});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_c3({4, 4, 2});
  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_c0.release());
  ref_columns.push_back(ref_c1.release());
  ref_columns.push_back(ref_c2.release());
  ref_columns.push_back(ref_c3.release());
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  ASSERT_EQ(task_graph->root_tasks.size(), 1);
  auto execute_result = task_graph->root_tasks[0]->result();
  ASSERT_EQ(execute_result.has_value(), true);
  auto execute_result_sorted = cudf::sort(*execute_result);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(execute_result_sorted->view(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(execute_result_sorted->view(), ref_table->view());
}

TEST_F(LogicalToExecution, WindowWithOrderBy)
{
  auto generate_parquet_file = [](std::string file_name) {
    std::vector<int> arg_col{1, 2, 3, 4, 5, 6};
    std::vector<int> partition_col0{1, 2, 1, 1, 2, 3};
    std::vector<int> order_col0{6, 5, 4, 3, 2, 1};
    cudf::test::fixed_width_column_wrapper<int> arg_values(arg_col.begin(), arg_col.end());
    cudf::test::fixed_width_column_wrapper<int> partition_values0(partition_col0.begin(),
                                                                  partition_col0.end());
    cudf::test::fixed_width_column_wrapper<int> order_values0(order_col0.begin(), order_col0.end());

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(partition_values0.release());
    columns.push_back(order_values0.release());
    columns.push_back(arg_values.release());
    auto partition_table = std::make_unique<cudf::table>(std::move(columns));

    cudf::io::table_input_metadata partition_metadata(partition_table->view());
    partition_metadata.column_metadata[0].set_name("keys0");
    partition_metadata.column_metadata[1].set_name("keys1");
    partition_metadata.column_metadata[2].set_name("values");

    auto partition_filepath = temp_env->get_temp_filepath(file_name);
    auto partition_options  = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(partition_filepath), partition_table->view());
    partition_options.metadata(partition_metadata);
    cudf::io::write_parquet(partition_options);
  };

  generate_parquet_file("partition_0.parquet");

  // Register the input tables
  gqe::catalog catalog{&task_manager_ctx};
  catalog.register_table(
    "input",
    {{"keys0", cudf::data_type(cudf::type_id::INT32)},
     {"keys1", cudf::data_type(cudf::type_id::INT32)},
     {"values", cudf::data_type(cudf::type_id::INT32)}},
    gqe::storage_kind::parquet_file{{temp_env->get_temp_filepath("partition_0.parquet")}},
    gqe::partitioning_schema_kind::automatic{});

  // Hand-code the logical plan
  auto read_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::vector<std::string>({"keys0", "keys1", "values"}),
    std::vector<cudf::data_type>({cudf::data_type(cudf::type_id::INT32),
                                  cudf::data_type(cudf::type_id::INT32),
                                  cudf::data_type(cudf::type_id::INT32)}),
    "input",
    nullptr  // partial_filter
  );

  std::vector<std::unique_ptr<gqe::expression>> arguments;
  arguments.emplace_back(std::make_unique<gqe::column_reference_expression>(2));

  std::vector<std::unique_ptr<gqe::expression>> partition_by;
  partition_by.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

  std::vector<std::unique_ptr<gqe::expression>> order_by;
  order_by.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

  gqe::window_frame_bound::unbounded window_lower_bound;
  gqe::window_frame_bound::bounded window_upper_bound(0);

  auto window_relation = std::make_shared<gqe::logical::window_relation>(
    std::move(read_relation),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    cudf::aggregation::SUM,
    std::move(arguments),
    std::move(order_by),
    std::move(partition_by),
    std::vector<cudf::order>({cudf::order::ASCENDING}),
    window_lower_bound,
    window_upper_bound);

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(window_relation.get());

  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Compare against reference result
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c0({1, 2, 1, 1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c1({6, 5, 4, 3, 2, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> ref_c2({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int64_t> ref_c3({8, 7, 7, 4, 5, 6});
  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_c0.release());
  ref_columns.push_back(ref_c1.release());
  ref_columns.push_back(ref_c2.release());
  ref_columns.push_back(ref_c3.release());
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  ASSERT_EQ(task_graph->root_tasks.size(), 1);
  auto execute_result = task_graph->root_tasks[0]->result();
  ASSERT_EQ(execute_result.has_value(), true);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(*execute_result, ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*execute_result, ref_table->view());
}

TEST_F(LogicalToExecution, UniqueKeyJoin)
{
  /**
   * Hand-code logical plan and generate task graph. The test then inspects the
   * `_unique_keys_policy` enum flag of the root tasks of the task graph to check that unique
   * key join will be invoked.
   */

  // Write two test Parquet files to the disk
  cudf::test::fixed_width_column_wrapper<int64_t> table_0_col_0({0, 1, 2, 3, 2, 0, 1});
  cudf::test::fixed_width_column_wrapper<int64_t> table_0_col_1({2, 2, 1, 3, 0, 3, 1});

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
  table_0_options.metadata(table_0_metadata);
  cudf::io::write_parquet(table_0_options);

  cudf::test::strings_column_wrapper table_1_col_0({"apple", "cat", "duck", "orange"});
  cudf::test::fixed_width_column_wrapper<int64_t> table_1_col_1({0, 1, 2, 3});

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
  table_1_options.metadata(table_1_metadata);
  cudf::io::write_parquet(table_1_options);

  // Register the input tables
  gqe::catalog catalog{&task_manager_ctx};
  catalog.register_table(
    "table_0",
    {gqe::column_traits{"table_0_col_0", cudf::data_type(cudf::type_id::INT64), {}},
     gqe::column_traits{"table_0_col_1", cudf::data_type(cudf::type_id::INT64), {}}},
    gqe::storage_kind::parquet_file{{table_0_filepath}},
    gqe::partitioning_schema_kind::automatic{});
  catalog.register_table(
    "table_1",
    {gqe::column_traits{"table_1_col_0", cudf::data_type(cudf::type_id::STRING), {}},
     gqe::column_traits{"table_1_col_1",
                        cudf::data_type(cudf::type_id::INT64),
                        {gqe::column_traits::column_property::unique}}},
    gqe::storage_kind::parquet_file{{table_1_filepath}},
    gqe::partitioning_schema_kind::automatic{});

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
      {cudf::data_type(cudf::type_id::STRING), cudf::data_type(cudf::type_id::INT64)}),
    "table_1",
    nullptr);  // partial_filter

  auto join_condition_part_1 =
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3));
  auto join_condition_part_2 =
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(3));
  auto join_condition =
    std::make_unique<gqe::logical_and_expression>(join_condition_part_1, join_condition_part_2);
  auto join_relation = std::make_shared<gqe::logical::join_relation>(
    read_relation_0,
    read_relation_1,
    std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
    std::move(join_condition),
    gqe::join_type_type::inner,
    std::vector<cudf::size_type>({2}));

  // optimize logical plan
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys},
    {});
  gqe::optimizer::logical_optimizer optimizer(&logical_rule_config, &catalog);
  auto opt_logical_plan = optimizer.optimize(join_relation);

  // transform to physical plan
  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(opt_logical_plan.get());

  // Generate the task graph
  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  // inspect _unique_keys_policy of join task
  int unique_keys_left{}, unique_keys_right{}, unique_keys_none{};
  for (auto curr_task : task_graph->root_tasks) {
    gqe::join_task* curr_join_task = dynamic_cast<gqe::join_task*>(curr_task.get());
    if (curr_join_task) {
      switch (curr_join_task->unique_keys_policy()) {
        case gqe::unique_keys_policy::left: {
          ++unique_keys_left;
          break;
        }
        case gqe::unique_keys_policy::right: {
          ++unique_keys_right;
          break;
        }
        case gqe::unique_keys_policy::none: {
          ++unique_keys_none;
          break;
        }
        default: break;
      }
    }
  }

  ASSERT_EQ(unique_keys_left, 0);
  ASSERT_EQ(unique_keys_right, 1);
  ASSERT_EQ(unique_keys_none, 0);

  // Execute on a single GPU
  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Verify the execution result
  cudf::test::strings_column_wrapper ref_col_0({"cat", "orange"});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());

  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  ASSERT_EQ(task_graph->root_tasks.size(), 1);
  auto execute_result = task_graph->root_tasks[0]->result();
  ASSERT_EQ(execute_result.has_value(), true);
  auto execute_result_sorted = cudf::sort(*execute_result);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(execute_result_sorted->view(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(execute_result_sorted->view(), ref_table->view());
}
