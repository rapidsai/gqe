/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/chunked_parquet_writer.hpp>

#include "../utility.hpp"

#include <gqe/qep/shapes/write_statistics.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>

#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

std::unique_ptr<cudf::table> read_parquet(std::string const& filepath)
{
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath)).build();
  return std::move(cudf::io::read_parquet(options).tbl);
}

std::size_t read_parquet_row_group_count(std::string const& filepath)
{
  return static_cast<std::size_t>(
    cudf::io::read_parquet_metadata(cudf::io::source_info(filepath)).num_rowgroups());
}

gqe::table_statistics get_write_statistics(gqe::qep::state_container_view result)
{
  auto statistics = gqe::qep::try_write_statistics(result);
  if (!statistics) { throw std::logic_error("write task did not emit write statistics"); }
  return std::move(*statistics);
}

}  // namespace

class ChunkedParquetWriterTest : public gqe::test::BaseFixture {
 protected:
  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }
};

TEST_F(ChunkedParquetWriterTest, RejectsEmptyAndMismatchedLengthsInSchema)
{
  auto const filepath = temp_env->get_temp_filepath("invalid_constructor.parquet");

  EXPECT_THROW(gqe::storage::chunked_parquet_write_task(filepath, {}, {}), std::invalid_argument);

  EXPECT_THROW(gqe::storage::chunked_parquet_write_task(
                 filepath,
                 {"col_0"},
                 {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}),
               std::invalid_argument);
}

TEST_F(ChunkedParquetWriterTest, RejectsNestedTypesInSchema)
{
  auto const filepath = temp_env->get_temp_filepath("nested_constructor.parquet");

  EXPECT_THROW(gqe::storage::chunked_parquet_write_task(
                 filepath, {"list_col"}, {cudf::data_type{cudf::type_id::LIST}}),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_parquet_write_task(
                 filepath, {"struct_col"}, {cudf::data_type{cudf::type_id::STRUCT}}),
               std::invalid_argument);
}

TEST_F(ChunkedParquetWriterTest, WritesSingleChunkAndReturnsStatistics)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1({10, 20, 30});
  auto expected = make_table_from_columns(col_0.release(), col_1.release());

  auto const filepath = temp_env->get_temp_filepath("single_chunk.parquet");
  gqe::storage::chunked_parquet_write_task task{
    filepath,
    {"col_0", "col_1"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}};
  auto accumulator = task.initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int64_t> input_col_0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> input_col_1({10, 20, 30});
  auto table = make_table_from_columns(input_col_0.release(), input_col_1.release());
  auto input = gqe::qep::state_container_builder()
                 .add_state(std::move(*table), gqe::query_stream::default_stream())
                 .build();
  task.next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 3);
  EXPECT_EQ(statistics.num_row_groups, 1);
  EXPECT_EQ(statistics.num_columns, 2);

  auto output = read_parquet(filepath);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedParquetWriterTest, FinalizeWithoutNextProducesReadableEmptyFile)
{
  auto const filepath   = temp_env->get_temp_filepath("no_chunks.parquet");
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT32}};
  gqe::storage::chunked_parquet_write_task task{filepath, {"col_0", "col_1"}, data_types};
  auto accumulator = task.initialize(ctx_ref());

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 0);
  EXPECT_EQ(statistics.num_row_groups, read_parquet_row_group_count(filepath));
  EXPECT_EQ(statistics.num_columns, 2);

  auto expected       = gqe::qep::make_empty_state_container(data_types);
  auto expected_table = gqe::qep::to_table_view(expected);
  ASSERT_TRUE(expected_table.has_value());

  // Successful readback verifies that finalize emitted a complete Parquet footer, not just a file
  // header.
  auto output = read_parquet(filepath);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_table, output->view());
}

TEST_F(ChunkedParquetWriterTest, WritesZeroRowChunkAsReadableEmptyFile)
{
  auto const filepath   = temp_env->get_temp_filepath("zero_row_chunk.parquet");
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT32}};
  gqe::storage::chunked_parquet_write_task task{filepath, {"col_0", "col_1"}, data_types};
  auto accumulator = task.initialize(ctx_ref());

  auto input       = gqe::qep::make_empty_state_container(data_types);
  auto input_table = gqe::qep::to_table_view(input);
  ASSERT_TRUE(input_table.has_value());
  task.next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 0);
  EXPECT_EQ(statistics.num_row_groups, read_parquet_row_group_count(filepath));
  EXPECT_EQ(statistics.num_columns, 2);

  // A successful zero-row write must initialize enough schema metadata for a valid file.
  auto output = read_parquet(filepath);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*input_table, output->view());
}

TEST_F(ChunkedParquetWriterTest, WritesMultipleChunksAfterInputContainersAreReleased)
{
  auto const filepath = temp_env->get_temp_filepath("multiple_chunks.parquet");
  gqe::storage::chunked_parquet_write_task task{
    filepath,
    {"col_0", "col_1"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}};
  auto accumulator = task.initialize(ctx_ref());

  {
    cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2});
    cudf::test::fixed_width_column_wrapper<int32_t> col_1({10, 20});
    auto table = make_table_from_columns(col_0.release(), col_1.release());
    auto input = gqe::qep::state_container_builder()
                   .add_state(std::move(*table), gqe::query_stream::default_stream())
                   .build();
    task.next(gqe::qep::state_container_view(input),
              gqe::qep::state_container_view(accumulator),
              ctx_ref());
  }

  {
    cudf::test::fixed_width_column_wrapper<int64_t> col_0({3, 4, 5});
    cudf::test::fixed_width_column_wrapper<int32_t> col_1({30, 40, 50});
    auto table = make_table_from_columns(col_0.release(), col_1.release());
    auto input = gqe::qep::state_container_builder()
                   .add_state(std::move(*table), gqe::query_stream::default_stream())
                   .build();
    task.next(gqe::qep::state_container_view(input),
              gqe::qep::state_container_view(accumulator),
              ctx_ref());
  }

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 5);
  EXPECT_EQ(statistics.num_row_groups, read_parquet_row_group_count(filepath));

  cudf::test::fixed_width_column_wrapper<int64_t> expected_col_0({1, 2, 3, 4, 5});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col_1({10, 20, 30, 40, 50});
  auto expected = make_table_from_columns(expected_col_0.release(), expected_col_1.release());
  auto output   = read_parquet(filepath);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedParquetWriterTest, ConcurrentNextWritesAllRows)
{
  auto const filepath = temp_env->get_temp_filepath("concurrent_chunks.parquet");
  auto const reference_filepath =
    temp_env->get_temp_filepath("concurrent_chunks_reference.parquet");
  gqe::storage::chunked_parquet_write_task task{
    filepath,
    {"col_0", "col_1"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}};
  auto accumulator = task.initialize(ctx_ref());

  std::vector<int64_t> col_0_values;
  std::vector<int32_t> col_1_values;
  col_0_values.reserve(32);
  col_1_values.reserve(32);
  for (int chunk_idx = 0; chunk_idx < 16; ++chunk_idx) {
    auto const base = static_cast<int64_t>(chunk_idx * 10);
    col_0_values.push_back(base + 1);
    col_0_values.push_back(base + 2);
    col_1_values.push_back(static_cast<int32_t>(base + 101));
    col_1_values.push_back(static_cast<int32_t>(base + 102));
  }

  cudf::test::fixed_width_column_wrapper<int64_t> expected_col_0(col_0_values.begin(),
                                                                 col_0_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col_1(col_1_values.begin(),
                                                                 col_1_values.end());
  auto expected = make_table_from_columns(expected_col_0.release(), expected_col_1.release());
  write_table_to_file(expected->view(), {"col_0", "col_1"}, reference_filepath);
  auto reference = read_parquet(reference_filepath);

  std::vector<std::future<void>> workers;

  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&, worker_idx] {
      for (int local_chunk_idx = 0; local_chunk_idx < 4; ++local_chunk_idx) {
        auto const chunk_idx = worker_idx * 4 + local_chunk_idx;
        auto const start     = chunk_idx * 2;
        auto chunk_view      = cudf::slice(expected->view(), {start, start + 2}).front();
        auto input           = gqe::qep::state_container_builder().add_state(chunk_view).build();

        task.next(gqe::qep::state_container_view(input),
                  gqe::qep::state_container_view(accumulator),
                  ctx_ref());
      }
    }));
  }

  for (auto& worker : workers) {
    worker.get();
  }

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, reference->num_rows());
  EXPECT_EQ(statistics.num_row_groups, read_parquet_row_group_count(filepath));
  EXPECT_EQ(statistics.num_columns, 2);

  auto output          = read_parquet(filepath);
  auto sorted_expected = cudf::sort(reference->view());
  auto sorted_output   = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_expected->view(), sorted_output->view());
}

TEST_F(ChunkedParquetWriterTest, RejectsWrongInputColumnCount)
{
  auto const filepath = temp_env->get_temp_filepath("wrong_count.parquet");
  gqe::storage::chunked_parquet_write_task task{
    filepath,
    {"col_0", "col_1"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}};
  auto accumulator = task.initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3});
  auto table = make_table_from_columns(col_0.release());
  auto input = gqe::qep::state_container_builder()
                 .add_state(std::move(*table), gqe::query_stream::default_stream())
                 .build();

  EXPECT_THROW(task.next(gqe::qep::state_container_view(input),
                         gqe::qep::state_container_view(accumulator),
                         ctx_ref()),
               std::invalid_argument);
}

TEST_F(ChunkedParquetWriterTest, RejectsWrongInputColumnType)
{
  auto const filepath = temp_env->get_temp_filepath("wrong_type.parquet");
  gqe::storage::chunked_parquet_write_task task{
    filepath, {"col_0"}, {cudf::data_type{cudf::type_id::INT64}}};
  auto accumulator = task.initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> col_0({1, 2, 3});
  auto table = make_table_from_columns(col_0.release());
  auto input = gqe::qep::state_container_builder()
                 .add_state(std::move(*table), gqe::query_stream::default_stream())
                 .build();

  EXPECT_THROW(task.next(gqe::qep::state_container_view(input),
                         gqe::qep::state_container_view(accumulator),
                         ctx_ref()),
               std::invalid_argument);
}

TEST_F(ChunkedParquetWriterTest, WriteableViewProvidesFoldTask)
{
  auto const filepath = temp_env->get_temp_filepath("view_fold.parquet");
  gqe::storage::parquet_table table{{filepath}};

  auto task =
    table.writeable_view()->get_fold_task({"col_0"}, {cudf::data_type{cudf::type_id::INT64}});
  ASSERT_NE(task, nullptr);

  auto accumulator = task->initialize(ctx_ref());
  cudf::test::fixed_width_column_wrapper<int64_t> input_col({1, 2, 3});
  auto chunk = make_table_from_columns(input_col.release());
  auto input = gqe::qep::state_container_builder()
                 .add_state(std::move(*chunk), gqe::query_stream::default_stream())
                 .build();
  task->next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result = task->finalize(std::move(accumulator), ctx_ref());
  EXPECT_EQ(get_write_statistics(result).num_rows, 3);

  cudf::test::fixed_width_column_wrapper<int64_t> expected_col({1, 2, 3});
  auto expected = make_table_from_columns(expected_col.release());
  auto output   = read_parquet(filepath);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedParquetWriterTest, WriteableViewRejectsMultiFileTables)
{
  gqe::storage::parquet_table table{{"shard_0.parquet", "shard_1.parquet"}};

  // A QEP fold task writes its entire input stream to one sink, so multi-file tables are
  // rejected.
  EXPECT_THROW(
    table.writeable_view()->get_fold_task({"col_0"}, {cudf::data_type{cudf::type_id::INT64}}),
    std::invalid_argument);
}
