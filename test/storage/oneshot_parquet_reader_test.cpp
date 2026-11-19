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

#include <gqe/storage/oneshot_parquet_reader.hpp>

#include "../utility.hpp"

#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

gqe::qep::state_container execute_once(gqe::qep::iterate_task const& task,
                                       gqe::qep::state_container_view inputs,
                                       gqe::context_reference ctx_ref)
{
  auto iterator = task.initialize(inputs, ctx_ref);
  auto output   = task.next(gqe::qep::state_container_view{iterator}, ctx_ref);
  if (!output.has_value()) { throw std::logic_error("one-shot task emitted no output"); }
  if (task.next(gqe::qep::state_container_view{iterator}, ctx_ref).has_value()) {
    throw std::logic_error("one-shot task emitted more than one output");
  }
  task.finalize(std::move(iterator), ctx_ref);
  return std::move(*output);
}

gqe::qep::state_container execute_once(gqe::qep::iterate_task const& task,
                                       gqe::qep::state_container_view inputs,
                                       gqe::context_reference ctx_ref,
                                       gqe::query_stream const& stream_handle,
                                       rmm::device_async_resource_ref mr)
{
  auto iterator = task.initialize(inputs, ctx_ref, stream_handle, mr);
  auto output   = task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream_handle, mr);
  if (!output.has_value()) { throw std::logic_error("one-shot task emitted no output"); }
  if (task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream_handle, mr).has_value()) {
    throw std::logic_error("one-shot task emitted more than one output");
  }
  task.finalize(std::move(iterator), ctx_ref, stream_handle, mr);
  return std::move(*output);
}

}  // namespace

class OneshotParquetReaderTest : public gqe::test::BaseFixture {
 protected:
  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }
};

TEST_F(OneshotParquetReaderTest, ProjectsCastsAndEmitsOnce)
{
  cudf::test::fixed_width_column_wrapper<int32_t> projected({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> ignored({10, 20, 30, 40});
  auto input = make_table_from_columns(projected.release(), ignored.release());

  auto const filepath = temp_env->get_temp_filepath("oneshot_project_cast.parquet");
  write_table_to_file(input->view(), {"projected", "ignored"}, filepath);

  gqe::storage::oneshot_parquet_read_task task{
    {filepath}, {"projected"}, {cudf::data_type{cudf::type_id::INT64}}};
  auto result = execute_once(task, gqe::qep::state_container_view{}, ctx_ref());

  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_col({1, 2, 3, 4});
  auto expected = make_table_from_columns(expected_col.release());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), *result_table);
}

TEST_F(OneshotParquetReaderTest, ReadUsesExplicitStreamAndMemoryResource)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("oneshot_explicit_stream_mr.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::oneshot_parquet_read_task task{
    {filepath}, {"col_0"}, {cudf::data_type{cudf::type_id::INT64}}};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto result =
    execute_once(task, gqe::qep::state_container_view{}, ctx_ref(), stream_handle, result_mr);
  stream_handle.ref().sync();

  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input->view(), *result_table);
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(result_table->column(0).head())));
}

TEST_F(OneshotParquetReaderTest, CountOnlyEmitsRowCount)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("oneshot_count.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::oneshot_parquet_read_task task{{filepath}, {}, {}};
  auto result = execute_once(task, gqe::qep::state_container_view{}, ctx_ref());

  EXPECT_EQ(gqe::qep::get_row_count(result), 5);
  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  EXPECT_EQ(result_table->num_columns(), 0);
}

TEST_F(OneshotParquetReaderTest, EmptyCountOnlySplitEmitsZero)
{
  gqe::storage::oneshot_parquet_read_task task{std::vector<std::string>{}, {}, {}};
  ASSERT_EQ(task.work_unit_count(), 0);

  auto split  = task.split({0, 0});
  auto result = execute_once(*split, gqe::qep::state_container_view{}, ctx_ref());

  EXPECT_EQ(gqe::qep::get_row_count(result), 0);
}

TEST_F(OneshotParquetReaderTest, CountOnlyAcrossRealSplitsIncludesEmptyTail)
{
  std::vector<std::string> filepaths;
  for (cudf::size_type file_idx = 0; file_idx < 5; ++file_idx) {
    auto const row_count = file_idx + 1;
    std::vector<int64_t> values(static_cast<std::size_t>(row_count), file_idx);
    cudf::test::fixed_width_column_wrapper<int64_t> col(values.begin(), values.end());
    auto table = make_table_from_columns(col.release());

    auto filepath =
      temp_env->get_temp_filepath("oneshot_count_split_" + std::to_string(file_idx) + ".parquet");
    write_table_to_file(table->view(), {"col_0"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  gqe::storage::oneshot_parquet_read_task task{filepaths, {}, {}};
  ASSERT_EQ(task.work_unit_count(), 5);

  std::vector<cudf::size_type> split_counts;
  cudf::size_type total_count = 0;
  std::vector<gqe::qep::source_work_range> const ranges{{0, 2}, {2, 4}, {4, 5}, {5, 5}};
  for (cudf::size_type split_idx = 0; split_idx < 4; ++split_idx) {
    auto split  = task.split(ranges[static_cast<std::size_t>(split_idx)]);
    auto result = execute_once(*split, gqe::qep::state_container_view{}, ctx_ref());

    ASSERT_EQ(result.size(), 1);
    auto const count = gqe::qep::get_row_count(result);
    split_counts.push_back(count);
    total_count += count;
  }

  EXPECT_EQ(split_counts, (std::vector<cudf::size_type>{3, 7, 5, 0}));
  EXPECT_EQ(total_count, 15);
}

TEST_F(OneshotParquetReaderTest, ConcurrentNextEmitsExactlyOnce)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("oneshot_concurrent.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::oneshot_parquet_read_task task{
    {filepath}, {"col_0"}, {cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());

  std::vector<gqe::qep::state_container> outputs;
  std::mutex outputs_mutex;
  std::vector<std::future<void>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&] {
      if (auto output = task.next(gqe::qep::state_container_view{iterator}, ctx_ref())) {
        std::lock_guard<std::mutex> const lock{outputs_mutex};
        outputs.push_back(std::move(*output));
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_FALSE(task.next(gqe::qep::state_container_view{iterator}, ctx_ref()).has_value());
  auto result_table = gqe::qep::to_table_view(outputs.front());
  ASSERT_TRUE(result_table.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input->view(), *result_table);
  task.finalize(std::move(iterator), ctx_ref());
}

TEST_F(OneshotParquetReaderTest, HivePartitionPruningAndSynthesis)
{
  std::vector<std::string> filepaths;
  for (auto const partition_key : {1, 2, 3}) {
    cudf::test::fixed_width_column_wrapper<int32_t> payload_col(
      {partition_key * 10, partition_key * 10 + 1});
    auto table = make_table_from_columns(payload_col.release());

    auto const dir =
      temp_env->get_temp_dir() + "/oneshot_hive/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 3});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  gqe::storage::oneshot_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};
  auto result = execute_once(task, gqe::qep::state_container_view{haystack}, ctx_ref());

  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col({1, 1, 3, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_payload_col({10, 11, 30, 31});
  auto expected =
    make_table_from_columns(expected_part_col.release(), expected_payload_col.release());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), *result_table);
}

TEST_F(OneshotParquetReaderTest, HivePruningCanEmitSchemaCorrectEmptyTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> payload_col({10, 11});
  auto table = make_table_from_columns(payload_col.release());

  auto const dir = temp_env->get_temp_dir() + "/oneshot_empty_hive/part_col=1";
  std::filesystem::create_directories(dir);
  auto const filepath = dir + "/data.parquet";
  write_table_to_file(table->view(), {"payload"}, filepath);

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({2});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  gqe::storage::oneshot_parquet_read_task task{
    {filepath},
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};
  auto result = execute_once(task, gqe::qep::state_container_view{haystack}, ctx_ref());

  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  EXPECT_EQ(result_table->num_rows(), 0);
  ASSERT_EQ(result_table->num_columns(), 2);
  EXPECT_EQ(result_table->column(0).type(), cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(result_table->column(1).type(), cudf::data_type{cudf::type_id::INT32});
}

TEST_F(OneshotParquetReaderTest, HivePruningAndNonzeroSynthesisAcrossSplits)
{
  std::vector<std::string> filepaths;
  std::vector<int32_t> expected_before;
  std::vector<int64_t> expected_partition_keys;
  std::vector<int32_t> expected_after;

  for (int32_t partition_key = 1; partition_key <= 4; ++partition_key) {
    auto const row_count = partition_key;
    std::vector<int32_t> before_values;
    std::vector<int32_t> after_values;
    before_values.reserve(row_count);
    after_values.reserve(row_count);
    for (int32_t row_idx = 0; row_idx < row_count; ++row_idx) {
      before_values.push_back(partition_key * 100 + row_idx);
      after_values.push_back(partition_key * 1000 + row_idx);
    }

    cudf::test::fixed_width_column_wrapper<int32_t> before_col(before_values.begin(),
                                                               before_values.end());
    cudf::test::fixed_width_column_wrapper<int32_t> after_col(after_values.begin(),
                                                              after_values.end());
    auto table = make_table_from_columns(before_col.release(), after_col.release());

    auto const dir =
      temp_env->get_temp_dir() + "/oneshot_split_hive/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"before", "after"}, filepath);
    filepaths.push_back(std::move(filepath));

    if (partition_key != 2) {
      expected_before.insert(expected_before.end(), before_values.begin(), before_values.end());
      expected_partition_keys.insert(expected_partition_keys.end(), row_count, partition_key);
      expected_after.insert(expected_after.end(), after_values.begin(), after_values.end());
    }
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 3, 4});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  gqe::storage::oneshot_parquet_read_task task{filepaths,
                                               {"before", "part_col", "after"},
                                               {cudf::data_type{cudf::type_id::INT32},
                                                cudf::data_type{cudf::type_id::INT64},
                                                cudf::data_type{cudf::type_id::INT32}},
                                               gqe::storage::hive_pruner_options{1}};

  std::vector<gqe::qep::state_container> split_outputs;
  std::vector<cudf::size_type> split_sizes;
  std::vector<gqe::qep::source_work_range> const ranges{{0, 2}, {2, 4}, {4, 4}};
  for (cudf::size_type split_idx = 0; split_idx < 3; ++split_idx) {
    auto split = task.split(ranges[static_cast<std::size_t>(split_idx)]);
    split_outputs.push_back(
      execute_once(*split, gqe::qep::state_container_view{haystack}, ctx_ref()));

    auto table = gqe::qep::to_table_view(split_outputs.back());
    ASSERT_TRUE(table.has_value());
    ASSERT_EQ(table->num_columns(), 3);
    split_sizes.push_back(table->num_rows());
  }
  EXPECT_EQ(split_sizes, (std::vector<cudf::size_type>{1, 7, 0}));

  std::vector<cudf::table_view> split_tables;
  for (auto const& output : split_outputs) {
    split_tables.push_back(*gqe::qep::to_table_view(output));
  }
  auto combined = cudf::concatenate(split_tables);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_before_col(expected_before.begin(),
                                                                      expected_before.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_partition_col(
    expected_partition_keys.begin(), expected_partition_keys.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_after_col(expected_after.begin(),
                                                                     expected_after.end());
  auto expected = make_table_from_columns(
    expected_before_col.release(), expected_partition_col.release(), expected_after_col.release());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), combined->view());
}

TEST_F(OneshotParquetReaderTest, SplitsFilesIntoExhaustiveSlices)
{
  std::vector<std::string> filepaths;
  for (int64_t value = 0; value < 5; ++value) {
    cudf::test::fixed_width_column_wrapper<int64_t> col({value});
    auto table = make_table_from_columns(col.release());
    auto filepath =
      temp_env->get_temp_filepath("oneshot_split_" + std::to_string(value) + ".parquet");
    write_table_to_file(table->view(), {"col_0"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  gqe::storage::oneshot_parquet_read_task task{
    filepaths, {"col_0"}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_EQ(task.work_unit_count(), 5);

  std::vector<gqe::qep::state_container> split_outputs;
  std::vector<gqe::qep::source_work_range> const ranges{{0, 2}, {2, 4}, {4, 5}, {5, 5}};
  for (cudf::size_type split_idx = 0; split_idx < 4; ++split_idx) {
    auto split = task.split(ranges[static_cast<std::size_t>(split_idx)]);
    split_outputs.push_back(execute_once(*split, gqe::qep::state_container_view{}, ctx_ref()));
  }

  std::vector<cudf::table_view> split_tables;
  std::vector<cudf::size_type> split_sizes;
  for (auto const& output : split_outputs) {
    auto table = gqe::qep::to_table_view(output);
    ASSERT_TRUE(table.has_value());
    split_sizes.push_back(table->num_rows());
    split_tables.push_back(*table);
  }
  EXPECT_EQ(split_sizes, (std::vector<cudf::size_type>{2, 2, 1, 0}));

  auto combined = cudf::concatenate(split_tables);
  cudf::test::fixed_width_column_wrapper<int64_t> expected_col({0, 1, 2, 3, 4});
  auto expected = make_table_from_columns(expected_col.release());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), combined->view());

  EXPECT_THROW(std::ignore = task.split({0, 6}), std::invalid_argument);
}

TEST_F(OneshotParquetReaderTest, SourceSplitsCountSelectedFiles)
{
  std::vector<std::string> filepaths;
  for (int64_t value = 0; value < 5; ++value) {
    std::vector<int64_t> const values(static_cast<std::size_t>(value + 1), value);
    cudf::test::fixed_width_column_wrapper<int64_t> col(values.begin(), values.end());
    auto table = make_table_from_columns(col.release());
    auto filepath =
      temp_env->get_temp_filepath("oneshot_range_partition_" + std::to_string(value) + ".parquet");
    write_table_to_file(table->view(), {"col_0"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  gqe::storage::oneshot_parquet_read_task task{filepaths, {}, {}};
  EXPECT_EQ(task.work_unit_count(), 5);

  std::vector<gqe::qep::source_work_range> const ranges{{0, 2}, {2, 3}, {3, 4}, {4, 5}};
  std::vector<cudf::size_type> const expected_rows{3, 3, 4, 5};
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    auto split_reader = task.split(ranges[i]);
    ASSERT_NE(dynamic_cast<gqe::storage::oneshot_parquet_read_task*>(split_reader.get()), nullptr);
    auto output = execute_once(*split_reader, gqe::qep::state_container_view{}, ctx_ref());
    EXPECT_EQ(gqe::qep::get_row_count(output), expected_rows[i]);
  }
}

TEST_F(OneshotParquetReaderTest, SourceSplitRejectsInvalidRanges)
{
  gqe::storage::oneshot_parquet_read_task task{
    {}, {"col_0"}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_EQ(task.work_unit_count(), 0);

  EXPECT_THROW(std::ignore = task.split({1, 0}), std::invalid_argument);
  EXPECT_THROW(std::ignore = task.split({0, 1}), std::invalid_argument);
}

TEST_F(OneshotParquetReaderTest, ValidatesSchemaAndPrunerIndex)
{
  EXPECT_THROW(
    std::ignore = gqe::storage::oneshot_parquet_read_task({"unused.parquet"}, {"col_0"}, {}),
    std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::storage::oneshot_parquet_read_task({"unused.parquet"},
                                                          {"col_0"},
                                                          {cudf::data_type{cudf::type_id::INT64}},
                                                          gqe::storage::hive_pruner_options{1}),
    std::invalid_argument);
}

TEST_F(OneshotParquetReaderTest, SourceSplitPreservesProjectedValues)
{
  std::vector<std::string> filepaths;
  for (int64_t file_index = 0; file_index < 4; ++file_index) {
    cudf::test::fixed_width_column_wrapper<int64_t> unused({-1, -2});
    cudf::test::fixed_width_column_wrapper<int64_t> ids({file_index * 10, file_index * 10 + 1});
    cudf::test::fixed_width_column_wrapper<int32_t> payload(
      {static_cast<int32_t>(file_index * 100), static_cast<int32_t>(file_index * 100 + 1)});
    auto input    = make_table_from_columns(unused.release(), ids.release(), payload.release());
    auto filepath = temp_env->get_temp_filepath("oneshot_partition_projection_" +
                                                std::to_string(file_index) + ".parquet");
    write_table_to_file(input->view(), {"unused", "id", "payload"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  gqe::storage::oneshot_parquet_read_task oneshot{
    filepaths,
    {"payload", "id"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  auto split_reader = oneshot.split({1, 3});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  std::vector<gqe::qep::state_container> chunks;
  std::vector<cudf::table_view> views;
  while (auto chunk = split_reader->next(iterator, ctx_ref())) {
    chunks.push_back(std::move(*chunk));
    auto table = gqe::qep::to_table_view(chunks.back());
    ASSERT_TRUE(table);
    views.push_back(*table);
  }
  ASSERT_FALSE(views.empty());
  auto actual = cudf::concatenate(views);
  split_reader->finalize(std::move(iterator), ctx_ref());

  cudf::test::fixed_width_column_wrapper<int64_t> expected_payload({100, 101, 200, 201});
  cudf::test::fixed_width_column_wrapper<int64_t> expected_ids({10, 11, 20, 21});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT((cudf::table_view{{expected_payload, expected_ids}}),
                                     actual->view());
}

TEST_F(OneshotParquetReaderTest, ReadableViewProvidesIterateTask)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("view_iterate.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::parquet_table table{{filepath}};
  auto task = table.readable_view()
                ->get_read_pipeline(gqe::source_reader_type::oneshot,
                                    "t",
                                    nullptr,
                                    {"col_0"},
                                    {cudf::data_type{cudf::type_id::INT64}})
                .source;
  ASSERT_NE(task, nullptr);

  auto result       = execute_once(*task, gqe::qep::state_container_view{}, ctx_ref());
  auto result_table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(result_table.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input->view(), *result_table);
}

TEST_F(OneshotParquetReaderTest, ReadableViewForwardsPrunerIndex)
{
  gqe::storage::parquet_table table{{"unused.parquet"}};

  // An out-of-range partition column index throws at task construction, proving the index is
  // forwarded to the underlying read task's Hive-pruning configuration.
  EXPECT_THROW(
    std::ignore = table.readable_view()->get_read_pipeline(gqe::source_reader_type::oneshot,
                                                           "t",
                                                           nullptr,
                                                           {"col_0"},
                                                           {cudf::data_type{cudf::type_id::INT64}},
                                                           1),
    std::invalid_argument);
}

TEST_F(OneshotParquetReaderTest, EmptySourceSplitPreservesOutputShape)
{
  auto const type = cudf::data_type{cudf::type_id::INT32};
  gqe::storage::oneshot_parquet_read_task source{{}, {"value"}, {type}};
  EXPECT_EQ(source.work_unit_count(), 0);
  auto split_reader = source.split({0, 0});
  ASSERT_NE(dynamic_cast<gqe::storage::oneshot_parquet_read_task*>(split_reader.get()), nullptr);
  auto iterator = split_reader->initialize({}, ctx_ref());
  auto chunk    = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto table = gqe::qep::to_table_view(*chunk);
  ASSERT_TRUE(table);
  EXPECT_EQ(table->num_rows(), 0);
  ASSERT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->column(0).type(), type);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  EXPECT_THROW(std::ignore = source.split({1, 1}), std::invalid_argument);
}
