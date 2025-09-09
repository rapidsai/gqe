/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gqe/catalog.hpp>
#include <gqe/communicator.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/project.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/query_context.hpp>
#include <gqe/scheduler.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "utilities.hpp"

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

TEST(MultiProcessExecutionTest, commInitTest)
{
  auto comm = std::make_unique<gqe::nvshmem_communicator>(MPI_COMM_WORLD);
  comm->init();
  std::size_t constexpr allocation_alignment = 256;
  char* local_base_ptr                       = (char*)nvshmem_align(allocation_alignment, 1024);
  char* peer_base_ptr                        = (char*)nvshmem_align(allocation_alignment, 1024);
  nvshmem_char_put(peer_base_ptr, local_base_ptr, 1024, (comm->rank() + 1) % comm->world_size());
  nvshmem_quiet();
  nvshmem_free(local_base_ptr);
  nvshmem_free(peer_base_ptr);
  comm->finalize();
}

TEST(MultiProcessExecutionTest, SingleGPUExecution)
{
  auto task_manager_ctx = gqe::multi_process_task_manager_context::default_init(MPI_COMM_WORLD);
  gqe::query_context query_ctx(gqe::optimization_parameters(true));
  gqe::context_reference ctx_ref{task_manager_ctx.get(), &query_ctx};

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
  gqe::catalog catalog;
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
  gqe::execute_task_graph_multi_process(ctx_ref, task_graph.get());

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
  auto root_task       = task_graph->root_tasks[0];
  auto execution_ranks = task_manager_ctx->scheduler->get_execution_ranks(root_task.get());

  // We have 2 stages in this test, and both have 1 pipeline. Result is only available on one of the
  // ranks.
  auto executed_locally =
    execution_ranks.find(task_manager_ctx->comm->rank()) != execution_ranks.end();
  if (executed_locally) {
    auto execute_result = root_task->result();
    ASSERT_EQ(execute_result.has_value(), true);
    auto execute_result_sorted = cudf::sort(*execute_result);

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(execute_result_sorted->view(), ref_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(execute_result_sorted->view(), ref_table->view());
  }
  task_manager_ctx->finalize();
}

TEST(MultiProcessExecutionTest, TableMigration)
{
  auto comm = std::make_unique<gqe::nvshmem_communicator>(MPI_COMM_WORLD);
  comm->init();
  auto pool_size = gqe::utility::default_device_memory_pool_size();

  if (comm->num_ranks_per_device() > 1) {
    GQE_LOG_WARN("Node process count {} >= number of GPUs {}. Using MPG mode for NVSHMEM",
                 comm->world_size(),
                 comm->num_ranks_per_device());
    pool_size = rmm::align_down(pool_size / comm->num_ranks_per_device(), 256);
  }
  // PGAS memory resource has to have the same size on all ranks.
  MPI_Allreduce(&pool_size, &pool_size, 1, MPI_LONG_LONG, MPI_MIN, comm->mpi_comm());
  GQE_LOG_INFO("Setting pool size to {}", pool_size);
  auto upstream_mr       = std::make_unique<gqe::pgas_memory_resource>(pool_size);
  auto scheduler         = std::make_unique<gqe::explicit_scheduler>();
  auto migration_service = std::make_unique<gqe::task_migration_service>(comm->device_id());
  auto server            = gqe::rpc_server(std::vector<grpc::Service*>{migration_service.get()});
  auto migration_client  = std::make_unique<gqe::nvshmem_task_migration_client>(
    comm.get(), server, upstream_mr->get_local_base_ptr());
  auto task_manager_ctx = gqe::multi_process_task_manager_context(std::move(comm),
                                                                  std::move(scheduler),
                                                                  std::move(migration_client),
                                                                  std::move(migration_service),
                                                                  std::move(server),
                                                                  std::move(upstream_mr));

  gqe::query_context query_ctx(gqe::optimization_parameters(true));
  gqe::context_reference ctx_ref{&task_manager_ctx, &query_ctx};

  int first_rank = 0;
  int last_rank  = task_manager_ctx.comm->world_size() - 1;

  cudf::test::fixed_width_column_wrapper<int64_t> col_wrap_0({0, 1, 2, 3, 4, 5},
                                                             {1, 0, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> col_wrap_1({0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  cudf::test::strings_column_wrapper col_wrap_2(
    {"apple", "orange", "apple", "apple", "apple", "orange"},
    {true, false, true, true, false, true});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(col_wrap_0.release());
  columns.push_back(col_wrap_1.release());
  columns.push_back(col_wrap_2.release());
  cudf::table_view input_table({columns[0]->view(), columns[1]->view(), columns[2]->view()});

  std::shared_ptr<gqe::task> input_task;
  if (task_manager_ctx.comm->rank() == first_rank) {
    input_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, 0, 0, std::make_unique<cudf::table>(input_table));
    task_manager_ctx.migration_service->register_task(
      input_task.get());  // all ranks should know where a task is executed
  } else {
    input_task = std::make_shared<gqe::test::no_op_task>(ctx_ref, 0, 0);
  }
  dynamic_cast<gqe::explicit_scheduler*>(task_manager_ctx.scheduler.get())
    ->set_execution_ranks(input_task.get(), {first_rank});
  task_manager_ctx.comm->barrier_world();  // make sure ranks wait for the result to be set

  std::vector<std::unique_ptr<gqe::expression>> project_expressions;
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
  project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(2));
  auto project_task = std::make_unique<gqe::project_task>(
    ctx_ref, 0, 1, std::move(input_task), std::move(project_expressions));

  if (task_manager_ctx.comm->rank() == last_rank) {
    project_task->execute();
    auto project_result = project_task->result();
    ASSERT_EQ(project_result.has_value(), true);
    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(project_result.value(), input_table);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(project_result.value(), input_table);
  }
  task_manager_ctx.finalize();
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  GQE_MPI_TRY(MPI_Init(NULL, NULL));

  int result = RUN_ALL_TESTS();

  GQE_MPI_TRY(MPI_Finalize());
  return result;
}