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

#include <gqe/storage/chunked_parquet_reader.hpp>

#include "../utility.hpp"

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <cudf/sorting.hpp>

#include <algorithm>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

/**
 * @brief Serially calls next() until end of stream.
 * Collects all chunks into a vector and builds the concatenated table using cudf table views.
 */
std::unique_ptr<cudf::table> drain_to_table(gqe::qep::iterate_task const& task,
                                            gqe::qep::state_container& iterator,
                                            gqe::context_reference ctx_ref)
{
  std::vector<gqe::qep::state_container> chunks;
  std::vector<cudf::table_view> chunk_views;

  while (auto chunk = task.next(gqe::qep::state_container_view(iterator), ctx_ref)) {
    chunks.emplace_back(std::move(*chunk));
    auto table = gqe::qep::to_table_view(chunks.back());
    EXPECT_TRUE(table.has_value());
    if (table.has_value()) { chunk_views.push_back(*table); }
  }

  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref).has_value());
  EXPECT_FALSE(chunk_views.empty());
  if (chunk_views.empty()) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }
  return cudf::concatenate(chunk_views);
}

}  // namespace

class ChunkedParquetReaderTest : public gqe::test::BaseFixture {
 protected:
  ChunkedParquetReaderTest()
  {
    auto params                    = gqe::make_optimization_parameters(true);
    params.parquet_chunk_read_size = 128;
    params.parquet_pass_read_size  = 1024 * 1024;
    initialize_contexts(params);
  }

  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }
};

TEST_F(ChunkedParquetReaderTest, ReportsHostInputResidency)
{
  gqe::storage::chunked_parquet_read_task task{
    {}, {"col"}, {cudf::data_type{cudf::type_id::INT32}}};
  auto const* trait = dynamic_cast<gqe::qep::input_residency_trait const*>(&task);

  ASSERT_NE(trait, nullptr);
  EXPECT_EQ(trait->input_residency().residency, gqe::qep::data_residency::host);
}

TEST_F(ChunkedParquetReaderTest, ReadsSingleFileAllColumns)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1({10, 20, 30, 40});
  auto expected = make_table_from_columns(col_0.release(), col_1.release());

  auto const filepath = temp_env->get_temp_filepath("chunked_single.parquet");
  auto const columns  = std::vector<std::string>{"col_0", "col_1"};
  write_table_to_file(expected->view(), columns, filepath);

  gqe::storage::chunked_parquet_read_task task{
    {filepath},
    columns,
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

TEST_F(ChunkedParquetReaderTest, ReadsMultipleFilesInOrder)
{
  cudf::test::fixed_width_column_wrapper<int64_t> file_0_col({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int64_t> file_1_col({4, 5});
  cudf::test::fixed_width_column_wrapper<int64_t> expected_col({1, 2, 3, 4, 5});

  auto file_0   = make_table_from_columns(file_0_col.release());
  auto file_1   = make_table_from_columns(file_1_col.release());
  auto expected = make_table_from_columns(expected_col.release());

  auto const path_0  = temp_env->get_temp_filepath("chunked_multi_0.parquet");
  auto const path_1  = temp_env->get_temp_filepath("chunked_multi_1.parquet");
  auto const columns = std::vector<std::string>{"col_0"};
  write_table_to_file(file_0->view(), columns, path_0);
  write_table_to_file(file_1->view(), columns, path_1);

  gqe::storage::chunked_parquet_read_task task{
    {path_0, path_1}, columns, {cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

/**
 * @brief Reader should only read the requested columns
 */
TEST_F(ChunkedParquetReaderTest, ProjectsColumns)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1({4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col({4, 5, 6});

  auto input    = make_table_from_columns(col_0.release(), col_1.release());
  auto expected = make_table_from_columns(expected_col.release());

  auto const filepath = temp_env->get_temp_filepath("chunked_project.parquet");
  write_table_to_file(input->view(), {"col_0", "col_1"}, filepath);

  gqe::storage::chunked_parquet_read_task task{
    {filepath}, {"col_1"}, {cudf::data_type{cudf::type_id::INT32}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

TEST_F(ChunkedParquetReaderTest, CastsToRequestedTypes)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int64_t> expected_col({1, 2, 3});

  auto input    = make_table_from_columns(col.release());
  auto expected = make_table_from_columns(expected_col.release());

  auto const filepath = temp_env->get_temp_filepath("chunked_cast.parquet");
  auto const columns  = std::vector<std::string>{"col_0"};
  write_table_to_file(input->view(), columns, filepath);

  gqe::storage::chunked_parquet_read_task task{
    {filepath}, columns, {cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

/**
 * @brief Chunked parquet reader in count only mode emits a single row count chunk and then signals
 * end of stream.
 */
TEST_F(ChunkedParquetReaderTest, CountOnlyEmitsRowCountOnce)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("chunked_count.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::chunked_parquet_read_task task{{filepath}, {}, {}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());

  auto count = task.next(gqe::qep::state_container_view(iterator), ctx_ref());
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(gqe::qep::get_row_count(*count), 5);
  auto table = gqe::qep::to_table_view(*count);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->num_columns(), 0);
  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref()).has_value());

  task.finalize(std::move(iterator), ctx_ref());
}

/**
 * @brief Concurrent count-only next() calls should emit exactly one row-count chunk.
 */
TEST_F(ChunkedParquetReaderTest, ConcurrentCountOnlyEmitsRowCountExactlyOnce)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5});
  auto input = make_table_from_columns(col.release());

  auto const filepath = temp_env->get_temp_filepath("chunked_count_concurrent.parquet");
  write_table_to_file(input->view(), {"col_0"}, filepath);

  gqe::storage::chunked_parquet_read_task task{{filepath}, {}, {}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());

  std::vector<cudf::size_type> counts;
  std::mutex counts_mutex;
  std::vector<std::future<void>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&] {
      while (auto chunk = task.next(gqe::qep::state_container_view(iterator), ctx_ref())) {
        auto const count = gqe::qep::get_row_count(*chunk);
        std::lock_guard<std::mutex> lock{counts_mutex};
        counts.push_back(count);
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }

  EXPECT_THAT(counts, ::testing::ElementsAre(5));
  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref()).has_value());

  task.finalize(std::move(iterator), ctx_ref());
}

/**
 * @brief Concurrent calls to next() should drain the requested chunks exactly once.
 */
TEST_F(ChunkedParquetReaderTest, ConcurrentNextDrainsExactlyOnce)
{
  constexpr auto num_files     = 8;
  constexpr auto rows_per_file = 16;
  std::vector<std::string> filepaths;
  filepaths.reserve(num_files);

  std::vector<int64_t> expected_values;
  expected_values.reserve(num_files * rows_per_file);
  auto const columns = std::vector<std::string>{"col_0"};

  for (int file_idx = 0; file_idx < num_files; ++file_idx) {
    std::vector<int64_t> values(rows_per_file);
    std::iota(values.begin(), values.end(), static_cast<int64_t>(file_idx * rows_per_file));
    expected_values.insert(expected_values.end(), values.begin(), values.end());

    cudf::test::fixed_width_column_wrapper<int64_t> col(values.begin(), values.end());
    auto table = make_table_from_columns(col.release());
    auto filepath =
      temp_env->get_temp_filepath("chunked_concurrent_" + std::to_string(file_idx) + ".parquet");
    write_table_to_file(table->view(), columns, filepath);
    filepaths.push_back(std::move(filepath));
  }

  cudf::test::fixed_width_column_wrapper<int64_t> expected_col(expected_values.begin(),
                                                               expected_values.end());
  auto expected = make_table_from_columns(expected_col.release());

  gqe::storage::chunked_parquet_read_task task{
    filepaths, columns, {cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref());

  std::vector<gqe::qep::state_container> chunks;
  std::mutex chunks_mutex;
  std::vector<std::future<void>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&] {
      while (auto chunk = task.next(gqe::qep::state_container_view(iterator), ctx_ref())) {
        std::lock_guard<std::mutex> lock{chunks_mutex};
        chunks.emplace_back(std::move(*chunk));
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }
  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref()).has_value());

  std::vector<cudf::table_view> chunk_views;
  chunk_views.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto table = gqe::qep::to_table_view(chunk);
    ASSERT_TRUE(table.has_value());
    chunk_views.push_back(*table);
  }
  ASSERT_FALSE(chunk_views.empty());
  auto result          = cudf::sort(cudf::concatenate(chunk_views)->view());
  auto expected_sorted = cudf::sort(expected->view());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), result->view());
}

/**
 * @brief Concurrent Hive reads should preserve the partition-key cursor alignment with chunks.
 */
TEST_F(ChunkedParquetReaderTest, ConcurrentHivePartitionSynthesisDrainsExactlyOnce)
{
  constexpr auto rows_per_file = 40;
  std::vector<std::string> filepaths;
  std::vector<int64_t> expected_part_values;
  std::vector<int64_t> expected_payload_values;

  for (auto const partition_key : {1, 2, 3}) {
    std::vector<int64_t> payload(rows_per_file);
    std::iota(payload.begin(), payload.end(), static_cast<int64_t>(partition_key * 1000));
    cudf::test::fixed_width_column_wrapper<int64_t> payload_col(payload.begin(), payload.end());
    auto table = make_table_from_columns(payload_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/concurrent_data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));

    expected_part_values.insert(expected_part_values.end(), rows_per_file, partition_key);
    expected_payload_values.insert(expected_payload_values.end(), payload.begin(), payload.end());
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 2, 3});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col(expected_part_values.begin(),
                                                                    expected_part_values.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_payload_col(
    expected_payload_values.begin(), expected_payload_values.end());
  auto expected =
    make_table_from_columns(expected_part_col.release(), expected_payload_col.release());

  gqe::storage::chunked_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}},
    gqe::storage::hive_pruner_options{0}};
  auto iterator = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());

  std::vector<gqe::qep::state_container> chunks;
  std::mutex chunks_mutex;
  std::vector<std::future<void>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&] {
      while (auto chunk = task.next(gqe::qep::state_container_view(iterator), ctx_ref())) {
        std::lock_guard<std::mutex> lock{chunks_mutex};
        chunks.emplace_back(std::move(*chunk));
      }
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }
  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref()).has_value());

  std::vector<cudf::table_view> chunk_views;
  chunk_views.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto table = gqe::qep::to_table_view(chunk);
    ASSERT_TRUE(table.has_value());
    chunk_views.push_back(*table);
  }
  ASSERT_FALSE(chunk_views.empty());
  auto result          = cudf::sort(cudf::concatenate(chunk_views)->view());
  auto expected_sorted = cudf::sort(expected->view());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), result->view());
}

TEST_F(ChunkedParquetReaderTest, HivePartitionPruningAndSynthesis)
{
  std::vector<std::string> filepaths;
  std::vector<int64_t> expected_partition_values;
  std::vector<int32_t> expected_payload_values;

  for (auto const& [partition_key, payload] : std::vector<std::pair<int64_t, std::vector<int32_t>>>{
         {1, {10, 11}}, {2, {20}}, {3, {30, 31}}}) {
    cudf::test::fixed_width_column_wrapper<int32_t> payload_col(payload.begin(), payload.end());
    auto table = make_table_from_columns(payload_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));

    if (partition_key == 1 || partition_key == 3) {
      expected_partition_values.insert(
        expected_partition_values.end(), payload.size(), partition_key);
      expected_payload_values.insert(expected_payload_values.end(), payload.begin(), payload.end());
    }
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 3});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col(
    expected_partition_values.begin(), expected_partition_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_payload_col(
    expected_payload_values.begin(), expected_payload_values.end());
  auto expected =
    make_table_from_columns(expected_part_col.release(), expected_payload_col.release());

  gqe::storage::chunked_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};
  auto split_reader = task.split({0, 3});
  auto clone        = gqe::qep::clone_qep_task_as<gqe::qep::iterate_task>(*split_reader);
  for (auto const* source : {static_cast<gqe::qep::iterate_task const*>(&task),
                             static_cast<gqe::qep::iterate_task const*>(clone.get())}) {
    auto iterator = source->initialize(gqe::qep::state_container_view(haystack), ctx_ref());
    auto result   = drain_to_table(*source, iterator, ctx_ref());
    source->finalize(std::move(iterator), ctx_ref());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
  }
  std::vector<std::unique_ptr<cudf::table>> split_tables;
  std::vector<cudf::table_view> split_views;
  for (std::size_t file = 0; file < filepaths.size(); ++file) {
    auto split_reader = task.split({file, file + 1});
    auto iterator     = split_reader->initialize(haystack, ctx_ref());
    auto result       = drain_to_table(*split_reader, iterator, ctx_ref());
    split_reader->finalize(std::move(iterator), ctx_ref());
    EXPECT_EQ(result->num_rows(), file == 1 ? 0 : 2);
    split_tables.push_back(std::move(result));
    split_views.push_back(split_tables.back()->view());
  }
  auto combined = cudf::concatenate(split_views);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), combined->view());
}

/**
 * @brief Exercises the "splice the synthesized partition column back at its original index" logic
 * with a partition column that is *not* at index 0. With idx == 0 the erase/insert at `begin()` are
 * degenerate, so a column-ordering bug in the splice would go undetected.
 */
TEST_F(ChunkedParquetReaderTest, HivePartitionColumnAtNonZeroIndex)
{
  struct file_spec {
    int64_t partition_key;
    std::vector<int32_t> before;
    std::vector<int32_t> after;
  };

  std::vector<std::string> filepaths;
  std::vector<int32_t> expected_before_values;
  std::vector<int64_t> expected_part_values;
  std::vector<int32_t> expected_after_values;

  for (auto const& [partition_key, before, after] : std::vector<file_spec>{
         {1, {100, 101}, {1000, 1001}}, {2, {200}, {2000}}, {3, {300, 301}, {3000, 3001}}}) {
    cudf::test::fixed_width_column_wrapper<int32_t> before_col(before.begin(), before.end());
    cudf::test::fixed_width_column_wrapper<int32_t> after_col(after.begin(), after.end());
    auto table = make_table_from_columns(before_col.release(), after_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"before", "after"}, filepath);
    filepaths.push_back(std::move(filepath));

    if (partition_key == 1 || partition_key == 3) {
      expected_before_values.insert(expected_before_values.end(), before.begin(), before.end());
      expected_part_values.insert(expected_part_values.end(), before.size(), partition_key);
      expected_after_values.insert(expected_after_values.end(), after.begin(), after.end());
    }
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 3});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  cudf::test::fixed_width_column_wrapper<int32_t> expected_before_col(
    expected_before_values.begin(), expected_before_values.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col(expected_part_values.begin(),
                                                                    expected_part_values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_after_col(expected_after_values.begin(),
                                                                     expected_after_values.end());
  auto expected = make_table_from_columns(
    expected_before_col.release(), expected_part_col.release(), expected_after_col.release());

  gqe::storage::chunked_parquet_read_task task{filepaths,
                                               {"before", "part_col", "after"},
                                               {cudf::data_type{cudf::type_id::INT32},
                                                cudf::data_type{cudf::type_id::INT64},
                                                cudf::data_type{cudf::type_id::INT32}},
                                               gqe::storage::hive_pruner_options{1}};
  auto iterator = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

/**
 * @brief When the haystack prunes away every file, `initialize` leaves the reader null and `next`
 * emits one schema-carrying empty table before signaling end-of-stream.
 */
TEST_F(ChunkedParquetReaderTest, HiveAllFilesPrunedEmitsSchemaCorrectEmptyChunk)
{
  std::vector<std::string> filepaths;
  for (auto const partition_key : {1, 2, 3}) {
    cudf::test::fixed_width_column_wrapper<int32_t> payload_col({10, 20});
    auto table = make_table_from_columns(payload_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  // Haystack matches none of the partition keys.
  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({97, 98, 99});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  gqe::storage::chunked_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};
  auto iterator = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());

  auto empty = task.next(iterator, ctx_ref());
  ASSERT_TRUE(empty);
  auto table = gqe::qep::to_table_view(*empty);
  ASSERT_TRUE(table);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
  // Idempotent end-of-stream.
  EXPECT_FALSE(task.next(gqe::qep::state_container_view(iterator), ctx_ref()).has_value());

  task.finalize(std::move(iterator), ctx_ref());
}

/**
 * @brief Files larger than `parquet_chunk_read_size` force the reader to split a single partition
 * across multiple chunks, and to span a file boundary within one chunk. This exercises the per-file
 * cursor in `compute_hive_chunk_segments` (partial-file carry-over across `next` calls).
 */
TEST_F(ChunkedParquetReaderTest, HivePartitionSynthesisAcrossChunkBoundaries)
{
  constexpr auto rows_per_file = 40;  // 40 * 8 bytes = 320B > 128B chunk_read_size
  std::vector<std::string> filepaths;
  std::vector<int64_t> expected_part_values;
  std::vector<int64_t> expected_payload_values;

  for (auto const partition_key : {1, 2}) {
    std::vector<int64_t> payload(rows_per_file);
    std::iota(payload.begin(), payload.end(), static_cast<int64_t>(partition_key * 1000));
    cudf::test::fixed_width_column_wrapper<int64_t> payload_col(payload.begin(), payload.end());
    auto table = make_table_from_columns(payload_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));

    expected_part_values.insert(expected_part_values.end(), rows_per_file, partition_key);
    expected_payload_values.insert(expected_payload_values.end(), payload.begin(), payload.end());
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 2});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col(expected_part_values.begin(),
                                                                    expected_part_values.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected_payload_col(
    expected_payload_values.begin(), expected_payload_values.end());
  auto expected =
    make_table_from_columns(expected_part_col.release(), expected_payload_col.release());

  gqe::storage::chunked_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}},
    gqe::storage::hive_pruner_options{0}};
  auto iterator = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

/**
 * @brief Null haystack entries must be dropped before matching. The null here masks the value `2`,
 * so the `part_col=2` file must be pruned; only `part_col=3` survives.
 */
TEST_F(ChunkedParquetReaderTest, HiveHaystackNullsAreDropped)
{
  std::vector<std::string> filepaths;
  for (auto const partition_key : {2, 3}) {
    cudf::test::fixed_width_column_wrapper<int32_t> payload_col({partition_key * 10});
    auto table = make_table_from_columns(payload_col.release());

    auto const dir = temp_env->get_temp_dir() + "/part_col=" + std::to_string(partition_key);
    std::filesystem::create_directories(dir);
    auto filepath = dir + "/data.parquet";
    write_table_to_file(table->view(), {"payload"}, filepath);
    filepaths.push_back(std::move(filepath));
  }

  // First entry's underlying value is 2 but is null; it must not keep the `part_col=2` file.
  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({2, 3}, {0, 1});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  cudf::test::fixed_width_column_wrapper<int64_t> expected_part_col({3});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_payload_col({30});
  auto expected =
    make_table_from_columns(expected_part_col.release(), expected_payload_col.release());

  gqe::storage::chunked_parquet_read_task task{
    filepaths,
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};
  auto iterator = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());
  auto result   = drain_to_table(task, iterator, ctx_ref());
  task.finalize(std::move(iterator), ctx_ref());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result->view());
}

/**
 * @brief Strict mode: a file that does not encode the partition key in its path must make
 * `initialize` throw rather than silently reading a mixed dataset.
 */
TEST_F(ChunkedParquetReaderTest, HiveNonMatchingFilePathThrows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> payload_col({10, 20});
  auto table = make_table_from_columns(payload_col.release());
  // Path intentionally lacks the `part_col=<value>` component.
  auto const filepath = temp_env->get_temp_filepath("chunked_hive_unpartitioned.parquet");
  write_table_to_file(table->view(), {"payload"}, filepath);

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 2, 3});
  auto haystack_table = make_table_from_columns(haystack_col.release());
  auto haystack       = gqe::qep::state_container_builder()
                    .add_state(std::move(*haystack_table), gqe::query_stream::default_stream())
                    .build();

  gqe::storage::chunked_parquet_read_task task{
    {filepath},
    {"part_col", "payload"},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}},
    gqe::storage::hive_pruner_options{0}};

  EXPECT_THROW(
    {
      auto unused = task.initialize(gqe::qep::state_container_view(haystack), ctx_ref());
      (void)unused;
    },
    std::runtime_error);
}

TEST_F(ChunkedParquetReaderTest, SourceSplitsCoverProjectedValuesAndCountOnly)
{
  std::vector<std::string> paths;
  for (int64_t value = 0; value < 3; ++value) {
    cudf::test::fixed_width_column_wrapper<int64_t> first({value});
    cudf::test::fixed_width_column_wrapper<int32_t> second({static_cast<int32_t>(value * 10)});
    auto table = make_table_from_columns(first.release(), second.release());
    auto path =
      temp_env->get_temp_filepath("partition_chunked_" + std::to_string(value) + ".parquet");
    write_table_to_file(table->view(), {"col_0", "col_1"}, path);
    paths.push_back(std::move(path));
  }
  gqe::storage::chunked_parquet_read_task task{
    paths, {"col_1"}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_EQ(task.work_unit_count(), 3);
  for (auto const range : std::vector<gqe::qep::source_work_range>{{0, 1}, {1, 3}}) {
    auto split_reader = task.split(range);
    auto iterator     = split_reader->initialize({}, ctx_ref());
    auto result       = drain_to_table(*split_reader, iterator, ctx_ref());
    split_reader->finalize(std::move(iterator), ctx_ref());
    std::vector<int64_t> values;
    for (auto index = range.begin; index < range.end; ++index) {
      values.push_back(index * 10);
    }
    cudf::test::fixed_width_column_wrapper<int64_t> expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT((cudf::table_view{{expected}}), result->view());
  }
  gqe::storage::chunked_parquet_read_task count{paths, {}, {}};
  auto split_reader = count.split({1, 3});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  auto chunk        = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 2);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  auto empty_split    = count.split({3, 3});
  auto empty_iterator = empty_split->initialize({}, ctx_ref());
  auto empty_chunk    = empty_split->next(empty_iterator, ctx_ref());
  ASSERT_TRUE(empty_chunk);
  EXPECT_EQ(gqe::qep::get_row_count(*empty_chunk), 0);
  EXPECT_FALSE(empty_split->next(empty_iterator, ctx_ref()));
  empty_split->finalize(std::move(empty_iterator), ctx_ref());
  EXPECT_THROW(std::ignore = task.split({2, 1}), std::invalid_argument);
  EXPECT_THROW(std::ignore = task.split({0, 4}), std::invalid_argument);
  gqe::storage::chunked_parquet_read_task empty{{}, {}, {}};
  EXPECT_EQ(empty.work_unit_count(), 0);
  EXPECT_THROW(std::ignore = empty.split({0, 1}), std::invalid_argument);
}

TEST_F(ChunkedParquetReaderTest, EmptySourceSplitPreservesOutputShape)
{
  auto const type = cudf::data_type{cudf::type_id::INT32};
  gqe::storage::chunked_parquet_read_task source{{}, {"value"}, {type}};
  EXPECT_EQ(source.work_unit_count(), 0);
  auto split_reader = source.split({0, 0});
  ASSERT_NE(dynamic_cast<gqe::storage::chunked_parquet_read_task*>(split_reader.get()), nullptr);
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

TEST_F(ChunkedParquetReaderTest, EmptyParquetFileSplitPreservesOutputShape)
{
  cudf::test::fixed_width_column_wrapper<int32_t> values{};
  auto input = make_table_from_columns(values.release());
  auto path  = temp_env->get_temp_filepath("split_empty_file.parquet");
  write_table_to_file(input->view(), {"value"}, path);
  gqe::storage::chunked_parquet_read_task source{
    {path}, {"value"}, {cudf::data_type{cudf::type_id::INT32}}};
  auto split_reader = source.split({0, 1});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  auto chunk        = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto table = gqe::qep::to_table_view(*chunk);
  ASSERT_TRUE(table);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 1);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
}
