/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/task_manager/plan_executor.hpp>

#include "../utility.hpp"

#include <gqe/physical/read.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/storage/descriptor.hpp>
#include <gqe/utility/uuid.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

std::shared_ptr<gqe::physical::read_relation> make_read_plan()
{
  return std::make_shared<gqe::physical::read_relation>(
    std::vector<std::shared_ptr<gqe::physical::relation>>{},
    std::vector<std::string>{"value"},
    "input",
    nullptr,
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}});
}

gqe::physical::identified_plan make_deserialized_plan(std::shared_ptr<gqe::physical::relation> root)
{
  auto identified_plan = gqe::physical::make_identified_plan(std::move(root));
  return gqe::rpc::deserialize_physical_plan(gqe::rpc::serialize_physical_plan(identified_plan));
}

gqe::optimization_parameters make_qep_parameters()
{
  auto parameters              = gqe::make_optimization_parameters(true);
  parameters.execution_backend = gqe::execution_backend_type::qep_task_graph;
  return parameters;
}

}  // namespace

/** @brief Test fixture for physical-plan execution through the task-manager boundary. */
class PlanExecutorTest : public gqe::test::BaseFixture {
 protected:
  PlanExecutorTest() { initialize_contexts(gqe::make_optimization_parameters(true)); }
};

TEST_F(PlanExecutorTest, ExecutesQepTaskGraphBackend)
{
  using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<std::int64_t>;

  auto input = make_table_from_columns(int64_column_wrapper{4, 8, 15, 16, 23, 42}.release());
  auto const filepath = temp_env->get_temp_filepath("plan_executor_qep.parquet");
  write_table_to_file(input->view(), {"value"}, filepath);

  auto const table_id = gqe::utility::uuid::from_string("550e8400-e29b-41d4-a716-446655440001");
  auto descriptor     = gqe::storage::parquet_file_descriptor{
        .table_name = "input", .table_id = table_id, .storage_revision = 0, .file_paths = {filepath}};
  gqe::task_manager::plan_executor executor(get_task_manager_ctx(), 0, false);

  auto result = executor(make_deserialized_plan(make_read_plan()),
                         gqe::utility::uuid::from_string("550e8400-e29b-41d4-a716-446655440002"),
                         {gqe::storage::descriptor{std::move(descriptor)}},
                         make_qep_parameters());

  ASSERT_FALSE(result.arrow_ipc.empty());
  auto buffer = arrow::Buffer::FromString(result.arrow_ipc);
  auto reader = arrow::ipc::RecordBatchStreamReader::Open(
                  std::make_shared<arrow::io::BufferReader>(std::move(buffer)))
                  .ValueOrDie();
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->num_columns(), 1);

  auto values = std::dynamic_pointer_cast<arrow::Int64Array>(batch->column(0));
  ASSERT_NE(values, nullptr);
  ASSERT_EQ(values->length(), 6);
  EXPECT_EQ(values->Value(0), 4);
  EXPECT_EQ(values->Value(1), 8);
  EXPECT_EQ(values->Value(2), 15);
  EXPECT_EQ(values->Value(3), 16);
  EXPECT_EQ(values->Value(4), 23);
  EXPECT_EQ(values->Value(5), 42);
}

TEST_F(PlanExecutorTest, RejectsQepTaskGraphBackendForMultipleProcesses)
{
  gqe::task_manager::plan_executor executor(get_task_manager_ctx(), 0, true);

  EXPECT_THROW(
    std::ignore = executor(
      make_deserialized_plan(make_read_plan()), gqe::utility::uuid{}, {}, make_qep_parameters()),
    std::invalid_argument);
}

TEST_F(PlanExecutorTest, ExecutesQepTaskGraphBackendForWritePlans)
{
  using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<std::int64_t>;

  auto input = make_table_from_columns(int64_column_wrapper{4, 8, 15, 16, 23, 42}.release());
  auto const input_filepath = temp_env->get_temp_filepath("plan_executor_qep_write_input.parquet");
  auto const output_filepath =
    temp_env->get_temp_filepath("plan_executor_qep_write_output.parquet");
  write_table_to_file(input->view(), {"value"}, input_filepath);

  auto plan = std::make_shared<gqe::physical::write_relation>(
    make_read_plan(), std::vector<std::string>{"value"}, "output");
  auto input_descriptor = gqe::storage::parquet_file_descriptor{
    .table_name       = "input",
    .table_id         = gqe::utility::uuid::from_string("550e8400-e29b-41d4-a716-446655440003"),
    .storage_revision = 0,
    .file_paths       = {input_filepath}};
  auto output_descriptor = gqe::storage::parquet_file_descriptor{
    .table_name       = "output",
    .table_id         = gqe::utility::uuid::from_string("550e8400-e29b-41d4-a716-446655440004"),
    .storage_revision = 0,
    .file_paths       = {output_filepath}};
  gqe::task_manager::plan_executor executor(get_task_manager_ctx(), 0, false);

  auto result = executor(make_deserialized_plan(plan),
                         gqe::utility::uuid::from_string("550e8400-e29b-41d4-a716-446655440005"),
                         {gqe::storage::descriptor{std::move(input_descriptor)},
                          gqe::storage::descriptor{std::move(output_descriptor)}},
                         make_qep_parameters());

  EXPECT_TRUE(result.arrow_ipc.empty());
  ASSERT_TRUE(result.has_write_stats);
  auto const statistics = gqe::rpc::deserialize_table_statistics(result.write_stats);
  EXPECT_EQ(statistics.num_rows, 6);
  EXPECT_EQ(statistics.num_row_groups, 1);
  EXPECT_EQ(statistics.num_columns, 1);

  auto const read_options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(output_filepath)).build();
  auto output = cudf::io::read_parquet(read_options).tbl;
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input->view(), output->view());
}
