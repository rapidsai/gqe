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

#include <gqe/storage/chunked_in_memory_writer.hpp>

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/node_manager/context.hpp>
#include <gqe/qep/shapes/write_statistics.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/oneshot_in_memory_reader.hpp>
#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <boost/interprocess/shared_memory_object.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <barrier>
#include <future>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr int num_workers       = 4;
constexpr int chunks_per_worker = 4;
constexpr int num_chunks        = num_workers * chunks_per_worker;

gqe::qep::state_container execute_once(
  gqe::qep::iterate_task const& task,
  gqe::context_reference ctx_ref,
  gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
{
  auto iterator = task.initialize(gqe::qep::state_container_view{}, ctx_ref, stream_handle, mr);
  auto output   = task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream_handle, mr);
  if (!output) { throw std::logic_error("one-shot task emitted no output"); }
  if (task.next(gqe::qep::state_container_view{iterator}, ctx_ref, stream_handle, mr)) {
    throw std::logic_error("one-shot task emitted more than one output");
  }
  task.finalize(std::move(iterator), ctx_ref, stream_handle, mr);
  return std::move(*output);
}

std::unique_ptr<cudf::table> read_table(gqe::storage::in_memory_table_interface const& table,
                                        std::vector<cudf::data_type> const& data_types,
                                        gqe::context_reference ctx_ref)
{
  std::vector<gqe::storage::row_group_interface const*> row_groups;
  row_groups.reserve(table.num_row_groups());
  for (std::size_t i = 0; i < table.num_row_groups(); ++i) {
    row_groups.push_back(table.row_group_at(i));
  }

  std::vector<cudf::size_type> column_indexes(data_types.size());
  std::iota(column_indexes.begin(), column_indexes.end(), cudf::size_type{0});
  gqe::storage::oneshot_in_memory_read_task read_task{
    std::move(row_groups), std::move(column_indexes), data_types};
  auto output       = execute_once(read_task, ctx_ref);
  auto output_table = gqe::qep::to_table_view(output);
  if (!output_table) { throw std::logic_error("in-memory reader did not emit a table"); }
  return std::make_unique<cudf::table>(*output_table);
}

std::vector<gqe::qep::state_container> make_single_value_inputs(int count)
{
  std::vector<gqe::qep::state_container> inputs;
  inputs.reserve(count);
  for (int32_t value = 0; value < count; ++value) {
    cudf::test::fixed_width_column_wrapper<int32_t> input_col({value});
    auto input_table = make_table_from_columns(input_col.release());
    inputs.push_back(gqe::qep::state_container_builder()
                       .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                       .build());
  }
  gqe::query_stream::default_stream().ref().sync();
  return inputs;
}

gqe::table_statistics get_write_statistics(gqe::qep::state_container_view result)
{
  auto statistics = gqe::qep::try_write_statistics(result);
  if (!statistics) { throw std::logic_error("write task did not emit write statistics"); }
  return std::move(*statistics);
}

}  // namespace

class ChunkedInMemoryWriterTest : public gqe::test::BaseFixture {
 protected:
  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  std::unique_ptr<gqe::storage::in_memory_table> make_table(std::vector<std::string> column_names,
                                                            std::vector<cudf::data_type> data_types)
  {
    return std::make_unique<gqe::storage::in_memory_table>(
      gqe::memory_kind::device{rmm::get_current_cuda_device()},
      column_names,
      data_types,
      get_task_manager_ctx());
  }
};

TEST_F(ChunkedInMemoryWriterTest, RejectsInvalidDestinationPermutationAndSchema)
{
  auto table = make_table(
    {"a", "b"}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}});

  EXPECT_THROW(gqe::storage::chunked_in_memory_write_task(
                 table->get_row_group_appender(), table->get_memory_kind(), {}, {}),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_write_task(
                 table->get_row_group_appender(),
                 table->get_memory_kind(),
                 {0},
                 {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_write_task(
                 table->get_row_group_appender(),
                 table->get_memory_kind(),
                 {0, 0},
                 {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_write_task(
                 table->get_row_group_appender(),
                 table->get_memory_kind(),
                 {0, 2},
                 {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_write_task(table->get_row_group_appender(),
                                                          table->get_memory_kind(),
                                                          {0},
                                                          {cudf::data_type{cudf::type_id::LIST}}),
               std::invalid_argument);
}

TEST_F(ChunkedInMemoryWriterTest, WritesSingleChunkAndReturnsStatistics)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT32}};
  auto table            = make_table({"a", "b"}, data_types);
  gqe::storage::chunked_in_memory_write_task task{
    table->get_row_group_appender(), table->get_memory_kind(), {0, 1}, data_types};
  auto accumulator = task.initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int64_t> input_a({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> input_b({10, 20, 30});
  auto input_table = make_table_from_columns(input_a.release(), input_b.release());
  auto input       = gqe::qep::state_container_builder()
                 .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                 .build();
  task.next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 3);
  EXPECT_EQ(statistics.num_row_groups, 1);
  EXPECT_EQ(statistics.num_columns, 2);
  ASSERT_EQ(table->num_row_groups(), 1);

  cudf::test::fixed_width_column_wrapper<int64_t> expected_a({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_b({10, 20, 30});
  auto expected = make_table_from_columns(expected_a.release(), expected_b.release());
  auto output   = read_table(*table, data_types, ctx_ref());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedInMemoryWriterTest, DistinguishesNoChunksFromZeroRowChunk)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}};

  {
    auto table = make_table({"a"}, data_types);
    gqe::storage::chunked_in_memory_write_task task{
      table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
    auto accumulator      = task.initialize(ctx_ref());
    auto result           = task.finalize(std::move(accumulator), ctx_ref());
    auto const statistics = get_write_statistics(result);
    EXPECT_EQ(statistics.num_rows, 0);
    EXPECT_EQ(statistics.num_row_groups, 0);
    EXPECT_EQ(table->num_row_groups(), 0);
  }

  {
    auto table = make_table({"a"}, data_types);
    gqe::storage::chunked_in_memory_write_task task{
      table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
    auto accumulator = task.initialize(ctx_ref());
    auto input       = gqe::qep::make_empty_state_container(data_types);
    task.next(gqe::qep::state_container_view(input),
              gqe::qep::state_container_view(accumulator),
              ctx_ref());
    auto result           = task.finalize(std::move(accumulator), ctx_ref());
    auto const statistics = get_write_statistics(result);
    EXPECT_EQ(statistics.num_rows, 0);
    EXPECT_EQ(statistics.num_row_groups, 1);
    ASSERT_EQ(table->num_row_groups(), 1);
    EXPECT_EQ(table->row_group_at(0)->size(), 0);
  }
}

TEST_F(ChunkedInMemoryWriterTest, WritesZeroRowStringChunk)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING}};
  auto table            = make_table({"a"}, data_types);
  gqe::storage::chunked_in_memory_write_task task{
    table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
  auto accumulator = task.initialize(ctx_ref());
  auto input       = gqe::qep::make_empty_state_container(data_types);

  task.next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 0);
  EXPECT_EQ(statistics.num_row_groups, 1);
  ASSERT_EQ(statistics.column_stats.size(), 1);
  auto const& string_statistics =
    std::get<gqe::string_compression_statistics>(statistics.column_stats.front().compression_stats);
  EXPECT_EQ(string_statistics.offsets_stats.uncompressed_size, 0);
  EXPECT_EQ(string_statistics.chars_stats.uncompressed_size, 0);

  ASSERT_EQ(table->num_row_groups(), 1);
  auto output = read_table(*table, data_types, ctx_ref());
  EXPECT_EQ(output->num_rows(), 0);
  ASSERT_EQ(output->num_columns(), 1);
  EXPECT_EQ(output->view().column(0).type(), data_types.front());
}

TEST_F(ChunkedInMemoryWriterTest, WritesMultipleChunksAfterInputsAreReleased)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32}};
  auto table            = make_table({"a"}, data_types);
  gqe::storage::chunked_in_memory_write_task task{
    table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
  auto accumulator = task.initialize(ctx_ref());

  for (auto values : {std::vector<int32_t>{1, 2}, std::vector<int32_t>{3, 4, 5}}) {
    cudf::test::fixed_width_column_wrapper<int32_t> input_col(values.begin(), values.end());
    auto input_table = make_table_from_columns(input_col.release());
    auto input       = gqe::qep::state_container_builder()
                   .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                   .build();
    task.next(gqe::qep::state_container_view(input),
              gqe::qep::state_container_view(accumulator),
              ctx_ref());
  }

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 5);
  EXPECT_EQ(statistics.num_row_groups, 2);
  EXPECT_EQ(table->num_row_groups(), 2);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_col({1, 2, 3, 4, 5});
  auto expected = make_table_from_columns(expected_col.release());
  auto output   = read_table(*table, data_types, ctx_ref());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedInMemoryWriterTest, WriteableViewMapsInputColumnsToDestinationOrder)
{
  get_query_ctx()->parameters.zone_map_partition_size = 0;
  auto const destination_types                        = std::vector<cudf::data_type>{
    cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}};
  auto table = make_table({"a", "b"}, destination_types);
  auto task  = table->writeable_view()->get_fold_task(
    {"b", "a"}, {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}});
  auto accumulator = task->initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> input_b({10, 20, 30});
  cudf::test::fixed_width_column_wrapper<int64_t> input_a({1, 2, 3});
  auto input_table = make_table_from_columns(input_b.release(), input_a.release());
  auto input       = gqe::qep::state_container_builder()
                 .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                 .build();
  task->next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result = task->finalize(std::move(accumulator), ctx_ref());
  EXPECT_EQ(get_write_statistics(result).num_rows, 3);

  cudf::test::fixed_width_column_wrapper<int64_t> expected_a({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_b({10, 20, 30});
  auto expected = make_table_from_columns(expected_a.release(), expected_b.release());
  auto output   = read_table(*table, destination_types, ctx_ref());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(ChunkedInMemoryWriterTest, WriteableViewRejectsPartialTableWrite)
{
  auto table = make_table(
    {"a", "b"}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}});

  EXPECT_THROW(
    table->writeable_view()->get_fold_task({"a"}, {cudf::data_type{cudf::type_id::INT64}}),
    std::invalid_argument);
}

TEST_F(ChunkedInMemoryWriterTest, WriteableViewRejectsDestinationTypeMismatch)
{
  auto table = make_table(
    {"a", "b"}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}});

  EXPECT_THROW(
    table->writeable_view()->get_fold_task(
      {"b", "a"}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}}),
    std::invalid_argument);
}

TEST_F(ChunkedInMemoryWriterTest, RejectsWrongInputSchema)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}};
  auto table            = make_table({"a"}, data_types);
  gqe::storage::chunked_in_memory_write_task task{
    table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
  auto accumulator = task.initialize(ctx_ref());

  auto empty_input = gqe::qep::state_container{};
  EXPECT_THROW(task.next(gqe::qep::state_container_view(empty_input),
                         gqe::qep::state_container_view(accumulator),
                         ctx_ref()),
               std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<int32_t> wrong_type({1, 2, 3});
  auto input_table = make_table_from_columns(wrong_type.release());
  auto input       = gqe::qep::state_container_builder()
                 .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                 .build();
  EXPECT_THROW(task.next(gqe::qep::state_container_view(input),
                         gqe::qep::state_container_view(accumulator),
                         ctx_ref()),
               std::invalid_argument);
}

TEST_F(ChunkedInMemoryWriterTest, ConcurrentNextWritesAllRows)
{
  auto const data_types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32}};
  auto table            = make_table({"a"}, data_types);
  gqe::storage::chunked_in_memory_write_task task{
    table->get_row_group_appender(), table->get_memory_kind(), {0}, data_types};
  auto accumulator = task.initialize(ctx_ref());
  auto inputs      = make_single_value_inputs(num_chunks);

  std::barrier start{num_workers};
  std::vector<std::future<void>> workers;
  auto* query_context = get_query_ctx();
  for (int worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&, worker_idx] {
      auto const stream_handle = query_context->create_stream("chunked-writer-worker");
      start.arrive_and_wait();
      for (int chunk_idx = 0; chunk_idx < chunks_per_worker; ++chunk_idx) {
        auto const value = worker_idx * chunks_per_worker + chunk_idx;
        task.next(gqe::qep::state_container_view(inputs.at(value)),
                  gqe::qep::state_container_view(accumulator),
                  ctx_ref(),
                  stream_handle);
      }
      stream_handle.ref().sync();
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }

  auto result           = task.finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, num_chunks);
  EXPECT_EQ(statistics.num_row_groups, num_chunks);
  EXPECT_EQ(table->num_row_groups(), num_chunks);

  std::vector<int32_t> expected_values(num_chunks);
  std::iota(expected_values.begin(), expected_values.end(), int32_t{0});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col(expected_values.begin(),
                                                               expected_values.end());
  auto expected      = make_table_from_columns(expected_col.release());
  auto output        = read_table(*table, data_types, ctx_ref());
  auto sorted_output = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), sorted_output->view());
}

class SharedChunkedInMemoryWriterTest : public gqe::test::BaseFixture {
 protected:
  static constexpr char table_name[] = "chunked_writer_shared_table";

  void SetUp() override
  {
    boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
    _segment_owner =
      std::make_unique<gqe::shared_segment>(gqe::create_only,
                                            gqe::node_manager::shared_memory_name,
                                            64 << 20,
                                            gqe::segment_layout_policy_type::single_arena_unplaced);
    _catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());
  }

  void TearDown() override
  {
    if (_catalog && _catalog->has_table(table_name)) { _catalog->unregister_table(table_name); }
    _catalog.reset();
    _segment_owner.reset();
    boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
  }

  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  std::shared_ptr<gqe::storage::shared_in_memory_table> register_table(
    std::vector<std::string> const& column_names, std::vector<cudf::data_type> const& data_types)
  {
    std::vector<gqe::column_traits> columns;
    columns.reserve(column_names.size());
    for (std::size_t i = 0; i < column_names.size(); ++i) {
      columns.emplace_back(column_names[i], data_types[i]);
    }
    _catalog->register_table(table_name,
                             columns,
                             gqe::storage_kind::shared_segment_memory{},
                             gqe::partitioning_schema_kind::none{});
    auto table = std::dynamic_pointer_cast<gqe::storage::shared_in_memory_table>(
      _catalog->table_storage(table_name));
    if (!table) { throw std::logic_error("catalog did not create a shared in-memory table"); }
    return table;
  }

  std::unique_ptr<gqe::storage::shared_in_memory_table> reattach(
    gqe::storage::shared_in_memory_table const& table,
    std::vector<std::string> const& column_names,
    std::vector<cudf::data_type> const& data_types)
  {
    return std::make_unique<gqe::storage::shared_in_memory_table>(
      &get_task_manager_ctx()->shared_memory_segment(),
      column_names,
      data_types,
      table.shared_table_handle());
  }

 private:
  std::unique_ptr<gqe::shared_segment> _segment_owner;
  std::unique_ptr<gqe::catalog> _catalog;
};

TEST_F(SharedChunkedInMemoryWriterTest, SharedTableWritesSingleChunkAndReturnsStatistics)
{
  std::vector<std::string> const column_names{"a"};
  std::vector<cudf::data_type> const data_types{cudf::data_type{cudf::type_id::INT32}};
  auto table       = register_table(column_names, data_types);
  auto task        = table->writeable_view()->get_fold_task(column_names, data_types);
  auto accumulator = task->initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> input_col({1, 2, 3});
  auto input_table = make_table_from_columns(input_col.release());
  auto input       = gqe::qep::state_container_builder()
                 .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                 .build();
  task->next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());

  auto result           = task->finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 3);
  EXPECT_EQ(statistics.num_row_groups, 1);
  EXPECT_EQ(table->num_row_groups(), 0);

  auto attached = reattach(*table, column_names, data_types);
  ASSERT_EQ(attached->num_row_groups(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col({1, 2, 3});
  auto expected = make_table_from_columns(expected_col.release());
  auto output   = read_table(*attached, data_types, ctx_ref());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(SharedChunkedInMemoryWriterTest, SharedTableWritesCompressedChunk)
{
  get_query_ctx()->parameters.compression_format = gqe::compression_format::ans;
  std::vector<std::string> const column_names{"a"};
  std::vector<cudf::data_type> const data_types{cudf::data_type{cudf::type_id::INT32}};
  auto table       = register_table(column_names, data_types);
  auto task        = table->writeable_view()->get_fold_task(column_names, data_types);
  auto accumulator = task->initialize(ctx_ref());

  cudf::test::fixed_width_column_wrapper<int32_t> input_col({1, 2, 3, 4});
  auto input_table = make_table_from_columns(input_col.release());
  auto input       = gqe::qep::state_container_builder()
                 .add_state(std::move(*input_table), gqe::query_stream::default_stream())
                 .build();
  task->next(
    gqe::qep::state_container_view(input), gqe::qep::state_container_view(accumulator), ctx_ref());
  auto result           = task->finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, 4);
  EXPECT_EQ(statistics.num_row_groups, 1);
  ASSERT_EQ(statistics.column_stats.size(), 1);
  auto const& compression_statistics = std::get<gqe::fixed_width_compression_statistics>(
    statistics.column_stats.front().compression_stats);
  EXPECT_EQ(compression_statistics.uncompressed_size, 4 * sizeof(int32_t));

  auto attached = reattach(*table, column_names, data_types);
  ASSERT_EQ(attached->num_row_groups(), 1);
  EXPECT_EQ(attached->row_group_at(0)->get_column(0).layout(),
            gqe::storage::column_layout::compressed_sliced);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col({1, 2, 3, 4});
  auto expected = make_table_from_columns(expected_col.release());
  auto output   = read_table(*attached, data_types, ctx_ref());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), output->view());
}

TEST_F(SharedChunkedInMemoryWriterTest, SharedTableConcurrentNextWritesAllRows)
{
  std::vector<std::string> const column_names{"a"};
  std::vector<cudf::data_type> const data_types{cudf::data_type{cudf::type_id::INT32}};
  auto table       = register_table(column_names, data_types);
  auto task        = table->writeable_view()->get_fold_task(column_names, data_types);
  auto accumulator = task->initialize(ctx_ref());
  auto inputs      = make_single_value_inputs(num_chunks);

  std::barrier start{num_workers};
  std::vector<std::future<void>> workers;
  auto* query_context = get_query_ctx();
  for (int worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
    workers.push_back(std::async(std::launch::async, [&, worker_idx] {
      auto const stream_handle = query_context->create_stream("shared-chunked-writer-worker");
      start.arrive_and_wait();
      for (int chunk_idx = 0; chunk_idx < chunks_per_worker; ++chunk_idx) {
        auto const value = worker_idx * chunks_per_worker + chunk_idx;
        task->next(gqe::qep::state_container_view(inputs.at(value)),
                   gqe::qep::state_container_view(accumulator),
                   ctx_ref(),
                   stream_handle);
      }
      stream_handle.ref().sync();
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }

  auto result           = task->finalize(std::move(accumulator), ctx_ref());
  auto const statistics = get_write_statistics(result);
  EXPECT_EQ(statistics.num_rows, num_chunks);
  EXPECT_EQ(statistics.num_row_groups, num_chunks);
  auto attached = reattach(*table, column_names, data_types);
  EXPECT_EQ(attached->num_row_groups(), num_chunks);

  std::vector<int32_t> expected_values(num_chunks);
  std::iota(expected_values.begin(), expected_values.end(), int32_t{0});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_col(expected_values.begin(),
                                                               expected_values.end());
  auto expected      = make_table_from_columns(expected_col.release());
  auto output        = read_table(*attached, data_types, ctx_ref());
  auto sorted_output = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), sorted_output->view());
}
