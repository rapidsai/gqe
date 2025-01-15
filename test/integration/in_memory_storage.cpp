/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>

#include <cuda_runtime_api.h>
#include <cudf/io/parquet.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/cuda_device.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

rmm::cuda_device_id current_cuda_device_id()
{
  int id{};
  GQE_CUDA_TRY(cudaGetDevice(&id));
  return rmm::cuda_device_id{id};
}

TEST(InMemoryStorage, CopyTable)
{
  // Load a cuDF table into an in-memory table
  auto rand_gen = std::default_random_engine();
  auto dist     = std::uniform_int_distribution<int64_t>();
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_col_0(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_col_1(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});

  std::vector<std::unique_ptr<cudf::column>> in_table_columns;
  in_table_columns.push_back(in_table_col_0.release());
  in_table_columns.push_back(in_table_col_1.release());
  auto in_table = std::make_unique<cudf::table>(std::move(in_table_columns));

  cudf::io::table_input_metadata in_table_metadata(in_table->view());
  in_table_metadata.column_metadata[0].set_name("in_table_col_0");
  in_table_metadata.column_metadata[1].set_name("in_table_col_1");

  auto in_table_filepath = temp_env->get_temp_filepath("in_table.parquet");
  auto in_table_options  = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(in_table_filepath), in_table->view());
  in_table_options.metadata(in_table_metadata);
  cudf::io::write_parquet(in_table_options);

  // Register the input and output tables
  gqe::catalog catalog;
  catalog.register_table("in_table",
                         {{"in_table_col_0", cudf::data_type(cudf::type_id::INT64)},
                          {"in_table_col_1", cudf::data_type(cudf::type_id::INT64)}},
                         gqe::storage_kind::parquet_file{{in_table_filepath}},
                         gqe::partitioning_schema_kind::none{});

  catalog.register_table("out_table",
                         {{"out_table_col_0", cudf::data_type(cudf::type_id::INT64)},
                          {"out_table_col_1", cudf::data_type(cudf::type_id::INT64)}},
                         gqe::storage_kind::device_memory{current_cuda_device_id()},
                         gqe::partitioning_schema_kind::none{});

  // Hand-code the logical plan
  auto read_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::string>({"in_table_col_0", "in_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "in_table",
    nullptr);

  auto write_relation = std::make_shared<gqe::logical::write_relation>(
    read_relation,
    std::vector<std::string>({"out_table_col_0", "out_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "out_table");

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(write_relation.get());

  gqe::task_manager_context dbctx{};
  gqe::query_context qctx(gqe::optimization_parameters(true));
  gqe::context_reference ctx_ref{&dbctx, &qctx};

  // Generate the task graph and execute on a single GPU
  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Read the in-memory table
  auto read_im_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::string>({"out_table_col_0", "out_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "out_table",
    nullptr);

  gqe::physical_plan_builder plan_builder_check(&catalog);
  auto physical_plan_check = plan_builder_check.build(read_im_relation.get());
  gqe::task_graph_builder graph_builder_check(ctx_ref, &catalog);
  auto task_graph_check = graph_builder_check.build(physical_plan_check.get());
  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph_check.get());

  // Verify the execution result
  auto result_table       = task_graph_check->root_tasks.at(0)->result().value();
  auto statistics_manager = catalog.statistics("out_table");

  ASSERT_EQ(result_table.num_columns(), 2);
  ASSERT_EQ(result_table.num_rows(), 4);
  ASSERT_EQ((statistics_manager->statistics()).num_rows, 4);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(in_table->view(), result_table);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_table->view(), result_table);
}

TEST(InMemoryStorage, CopyTableParallel)
{
  // Write a test Parquet file to disk
  auto rand_gen = std::default_random_engine();
  auto dist     = std::uniform_int_distribution<int64_t>();
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_part_0_col_0(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_part_0_col_1(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_part_1_col_0(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});
  cudf::test::fixed_width_column_wrapper<int64_t> in_table_part_1_col_1(
    {dist(rand_gen), dist(rand_gen), dist(rand_gen), dist(rand_gen)});

  std::vector<std::unique_ptr<cudf::column>> in_table_part_0_columns;
  std::vector<std::unique_ptr<cudf::column>> in_table_part_1_columns;
  in_table_part_0_columns.push_back(in_table_part_0_col_0.release());
  in_table_part_0_columns.push_back(in_table_part_0_col_1.release());
  in_table_part_1_columns.push_back(in_table_part_1_col_0.release());
  in_table_part_1_columns.push_back(in_table_part_1_col_1.release());
  auto in_table_part_0 = std::make_unique<cudf::table>(std::move(in_table_part_0_columns));
  auto in_table_part_1 = std::make_unique<cudf::table>(std::move(in_table_part_1_columns));

  cudf::io::table_input_metadata in_table_part_0_metadata(in_table_part_0->view());
  cudf::io::table_input_metadata in_table_part_1_metadata(in_table_part_1->view());
  in_table_part_0_metadata.column_metadata[0].set_name("in_table_col_0");
  in_table_part_0_metadata.column_metadata[1].set_name("in_table_col_1");
  in_table_part_1_metadata.column_metadata[0].set_name("in_table_col_0");
  in_table_part_1_metadata.column_metadata[1].set_name("in_table_col_1");

  auto in_table_part_0_filepath = temp_env->get_temp_filepath("in_table_part_0.parquet");
  auto in_table_part_1_filepath = temp_env->get_temp_filepath("in_table_part_1.parquet");
  auto in_table_part_0_options  = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(in_table_part_0_filepath), in_table_part_0->view());
  auto in_table_part_1_options = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(in_table_part_1_filepath), in_table_part_1->view());
  in_table_part_0_options.metadata(in_table_part_0_metadata);
  in_table_part_1_options.metadata(in_table_part_1_metadata);
  cudf::io::write_parquet(in_table_part_0_options);
  cudf::io::write_parquet(in_table_part_1_options);

  // Register the input and output tables
  gqe::catalog catalog;
  catalog.register_table(
    "in_table",
    {{"in_table_col_0", cudf::data_type(cudf::type_id::INT64)},
     {"in_table_col_1", cudf::data_type(cudf::type_id::INT64)}},
    gqe::storage_kind::parquet_file{{in_table_part_0_filepath, in_table_part_1_filepath}},
    gqe::partitioning_schema_kind::none{});

  catalog.register_table("out_table",
                         {{"out_table_col_0", cudf::data_type(cudf::type_id::INT64)},
                          {"out_table_col_1", cudf::data_type(cudf::type_id::INT64)}},
                         gqe::storage_kind::device_memory{current_cuda_device_id()},
                         gqe::partitioning_schema_kind::none{});

  // Hand-code the logical plan
  auto read_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::string>({"in_table_col_0", "in_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "in_table",
    nullptr);

  auto write_relation = std::make_shared<gqe::logical::write_relation>(
    read_relation,
    std::vector<std::string>({"out_table_col_0", "out_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "out_table");

  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(write_relation.get());

  gqe::task_manager_context dbctx{};
  gqe::query_context qctx(gqe::optimization_parameters(true));
  gqe::context_reference ctx_ref{&dbctx, &qctx};

  // Generate the task graph and execute on a single GPU
  gqe::task_graph_builder graph_builder(ctx_ref, &catalog);
  auto task_graph = graph_builder.build(physical_plan.get());

  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph.get());

  // Read the in-memory table
  auto read_im_relation = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::string>({"out_table_col_0", "out_table_col_1"}),
    std::vector<cudf::data_type>(
      {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT64)}),
    "out_table",
    nullptr);

  gqe::physical_plan_builder plan_builder_check(&catalog);
  auto physical_plan_check = plan_builder_check.build(read_im_relation.get());

  gqe::task_graph_builder graph_builder_check(ctx_ref, &catalog);
  auto task_graph_check = graph_builder_check.build(physical_plan_check.get());
  gqe::execute_task_graph_single_gpu(ctx_ref, task_graph_check.get());

  // Verify the execution result
  auto result_table_part_0 = task_graph_check->root_tasks.at(0)->result().value();
  auto result_table_part_1 = task_graph_check->root_tasks.at(1)->result().value();
  auto statistics_manager  = catalog.statistics("out_table");

  ASSERT_EQ(result_table_part_0.num_columns(), 2);
  ASSERT_EQ(result_table_part_1.num_columns(), 2);
  ASSERT_EQ(result_table_part_0.num_rows(), 4);
  ASSERT_EQ(result_table_part_1.num_rows(), 4);
  ASSERT_EQ((statistics_manager->statistics()).num_rows, 8);

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(in_table_part_0->view(), result_table_part_0);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_table_part_0->view(), result_table_part_0);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(in_table_part_1->view(), result_table_part_1);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_table_part_1->view(), result_table_part_1);
}