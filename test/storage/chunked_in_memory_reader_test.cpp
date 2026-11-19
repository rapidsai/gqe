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

#include <gqe/storage/chunked_in_memory_reader.hpp>

#include "../utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <cudf/sorting.hpp>
#include <rmm/cuda_stream.hpp>

#include <barrier>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

class ChunkedInMemoryReaderTest : public gqe::test::BaseFixture {
 protected:
  using row_group_storage = std::deque<gqe::storage::row_group>;

  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  static gqe::qep::source_morsel_metadata device_source_metadata()
  {
    return {.residency = gqe::qep::data_residency::device};
  }

  static std::vector<gqe::storage::row_group_interface const*> row_group_refs(
    row_group_storage const& row_groups)
  {
    std::vector<gqe::storage::row_group_interface const*> refs;
    refs.reserve(row_groups.size());
    for (auto const& row_group : row_groups) {
      refs.push_back(&row_group);
    }
    return refs;
  }

  static void append_row_group(
    row_group_storage& row_groups,
    std::vector<std::vector<int32_t>> const& values,
    std::optional<cudf::size_type> zone_map_partition_size = std::nullopt)
  {
    if (values.empty()) { throw std::invalid_argument("row group must have at least one column"); }

    std::vector<std::unique_ptr<cudf::column>> cudf_columns;
    cudf_columns.reserve(values.size());
    for (auto const& column : values) {
      cudf::test::fixed_width_column_wrapper<int32_t> wrapper(column.begin(), column.end());
      cudf_columns.push_back(wrapper.release());
    }

    std::vector<cudf::column_view> column_views;
    column_views.reserve(cudf_columns.size());
    for (auto const& column : cudf_columns) {
      column_views.push_back(column->view());
    }

    std::unique_ptr<gqe::zone_map> zone_map;
    if (zone_map_partition_size) {
      zone_map = std::make_unique<gqe::local_zone_map>(cudf::table_view{column_views},
                                                       *zone_map_partition_size);
    }

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.reserve(cudf_columns.size());
    for (auto& column : cudf_columns) {
      columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
    }
    row_groups.emplace_back(std::move(columns), std::move(zone_map));
  }

  static std::unique_ptr<gqe::expression> greater_than(int32_t value)
  {
    return std::make_unique<gqe::greater_expression>(
      std::make_unique<gqe::column_reference_expression>(0),
      std::make_unique<gqe::literal_expression<int32_t>>(value));
  }
};

TEST_F(ChunkedInMemoryReaderTest, RejectsInvalidConstructorArguments)
{
  gqe::storage::row_group const* null_row_group = nullptr;
  EXPECT_THROW(
    gqe::storage::chunked_in_memory_read_task(
      {null_row_group}, {0}, {cudf::data_type{cudf::type_id::INT32}}, device_source_metadata()),
    std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_read_task(
                 {},
                 {0},
                 {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
                 device_source_metadata()),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_read_task(
                 {}, {0}, {cudf::data_type{cudf::type_id::LIST}}, device_source_metadata()),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::chunked_in_memory_read_task(
                 {}, {}, {}, device_source_metadata(), greater_than(0)),
               std::invalid_argument);
}

TEST_F(ChunkedInMemoryReaderTest, ReportsHostInputResidency)
{
  gqe::storage::chunked_in_memory_read_task task{
    {}, {}, {}, {.residency = gqe::qep::data_residency::host}};
  auto const* trait = dynamic_cast<gqe::qep::input_residency_trait const*>(&task);

  ASSERT_NE(trait, nullptr);
  EXPECT_EQ(trait->input_residency().residency, gqe::qep::data_residency::host);
}

TEST_F(ChunkedInMemoryReaderTest, ReportsDeviceInputResidency)
{
  gqe::storage::chunked_in_memory_read_task task{{}, {}, {}, device_source_metadata()};
  auto const* trait = dynamic_cast<gqe::qep::input_residency_trait const*>(&task);

  ASSERT_NE(trait, nullptr);
  EXPECT_EQ(trait->input_residency().residency, gqe::qep::data_residency::device);
}

TEST_F(ChunkedInMemoryReaderTest, InitializeValidatesInputsContextsAndSchema)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}});

  gqe::storage::chunked_in_memory_read_task valid{row_group_refs(row_groups),
                                                  {0},
                                                  {cudf::data_type{cudf::type_id::INT32}},
                                                  device_source_metadata()};
  auto unexpected_input = gqe::qep::make_row_count_container(1);
  EXPECT_THROW(
    std::ignore = valid.initialize(gqe::qep::state_container_view{unexpected_input}, ctx_ref()),
    std::invalid_argument);
  EXPECT_THROW(std::ignore = valid.initialize({}, {nullptr, get_query_ctx()}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = valid.initialize({}, {get_task_manager_ctx(), nullptr}),
               std::invalid_argument);

  gqe::storage::chunked_in_memory_read_task bad_index{row_group_refs(row_groups),
                                                      {1},
                                                      {cudf::data_type{cudf::type_id::INT32}},
                                                      device_source_metadata()};
  EXPECT_THROW(std::ignore = bad_index.initialize({}, ctx_ref()), std::out_of_range);

  gqe::storage::chunked_in_memory_read_task bad_type{row_group_refs(row_groups),
                                                     {0},
                                                     {cudf::data_type{cudf::type_id::INT64}},
                                                     device_source_metadata()};
  EXPECT_THROW(std::ignore = bad_type.initialize({}, ctx_ref()), std::invalid_argument);
}

TEST_F(ChunkedInMemoryReaderTest, PreservesNullsForContiguousFixedWidthColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> nullable({1, 2, 3}, {true, false, true});
  std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
  columns.push_back(
    std::make_unique<gqe::storage::contiguous_column>(std::move(*nullable.release())));
  row_group_storage row_groups;
  row_groups.emplace_back(std::move(columns));

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {0},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());
  auto output   = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  ASSERT_EQ(output_table->num_columns(), 1);
  EXPECT_EQ(output_table->column(0).null_count(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2, 3}, {true, false, true});
  auto expected_table = make_table_from_columns(expected.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), *output_table);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
  task.finalize(std::move(iterator), ctx_ref());
}

TEST_F(ChunkedInMemoryReaderTest, PreservesNullsAfterPartitionPruning)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  cudf::test::fixed_width_column_wrapper<int32_t> keys({0, 1, 2, 3, 4, 5});
  cudf::test::fixed_width_column_wrapper<int32_t> values({10, 11, 12, 13, 14, 15},
                                                         {false, true, true, false, false, true});
  auto input    = make_table_from_columns(keys.release(), values.release());
  auto zone_map = std::make_unique<gqe::local_zone_map>(input->view(), 2);
  std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
  for (auto& column : input->release()) {
    columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
  }
  row_group_storage row_groups;
  row_groups.emplace_back(std::move(columns), std::move(zone_map));

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {1},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata(),
                                                 greater_than(1)};
  auto iterator = task.initialize({}, ctx_ref());
  auto output   = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({12, 13, 14, 15},
                                                           {true, false, false, true});
  auto expected_table = make_table_from_columns(expected.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), *output_table);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
  task.finalize(std::move(iterator), ctx_ref());
}

TEST_F(ChunkedInMemoryReaderTest, PreservesNullsAcrossDisjointRetainedRanges)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  // With key > 2, keep [0, 2) and [4, 6) but prune [2, 4), so the null masks must be
  // concatenated from two disjoint ranges within one row group.
  cudf::test::fixed_width_column_wrapper<int32_t> keys({3, 4, 0, 1, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> values({10, 11, 12, 13, 14, 15},
                                                         {false, true, true, false, false, true});
  auto input    = make_table_from_columns(keys.release(), values.release());
  auto zone_map = std::make_unique<gqe::local_zone_map>(input->view(), 2);
  std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
  for (auto& column : input->release()) {
    columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
  }
  row_group_storage row_groups;
  row_groups.emplace_back(std::move(columns), std::move(zone_map));

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {1},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata(),
                                                 greater_than(2)};
  auto iterator = task.initialize({}, ctx_ref());
  auto output   = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  EXPECT_EQ(output_table->column(0).null_count(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({10, 11, 14, 15},
                                                           {false, true, false, true});
  auto expected_table = make_table_from_columns(expected.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), *output_table);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
  task.finalize(std::move(iterator), ctx_ref());
}

TEST_F(ChunkedInMemoryReaderTest, EmitsOneOwnedChunkPerRowGroup)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2}, {10, 20}});
  append_row_group(row_groups, {{3, 4, 5}, {30, 40, 50}});

  gqe::storage::chunked_in_memory_read_task task{
    row_group_refs(row_groups),
    {1, 0},
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
    device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  auto first = task.next(iterator, ctx_ref());
  ASSERT_TRUE(first);
  auto first_table = gqe::qep::to_table_view(*first);
  ASSERT_TRUE(first_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_first_a({10, 20});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_first_b({1, 2});
  auto expected_first =
    make_table_from_columns(expected_first_a.release(), expected_first_b.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first->view(), *first_table);

  auto second = task.next(iterator, ctx_ref());
  ASSERT_TRUE(second);
  auto second_table = gqe::qep::to_table_view(*second);
  ASSERT_TRUE(second_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_second_a({30, 40, 50});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_second_b({3, 4, 5});
  auto expected_second =
    make_table_from_columns(expected_second_a.release(), expected_second_b.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_second->view(), *second_table);

  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, ZeroRowGroupsEachEmitAChunk)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{}});
  append_row_group(row_groups, {{}});

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {0},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    auto output = task.next(iterator, ctx_ref());
    ASSERT_TRUE(output);
    auto output_table = gqe::qep::to_table_view(*output);
    ASSERT_TRUE(output_table);
    EXPECT_EQ(output_table->num_rows(), 0);
    EXPECT_EQ(output_table->num_columns(), 1);
    EXPECT_EQ(output_table->column(0).type(), cudf::data_type{cudf::type_id::INT32});
  }
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, InitializePrunesRowGroupsAndNextPrunesPartitions)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  row_group_storage row_groups;
  append_row_group(row_groups, {{0, 1}}, 2);
  append_row_group(row_groups, {{10, 11}}, 2);
  append_row_group(row_groups, {{12, 13, 20, 21}}, 2);

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {0},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata(),
                                                 greater_than(15)};
  auto iterator = task.initialize({}, ctx_ref());

  auto output = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values({20, 21});
  auto expected = make_table_from_columns(expected_values.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), *output_table);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, EmitsOneEmptyChunkWhenNoRowGroupSurvives)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;

  row_group_storage row_groups;
  append_row_group(row_groups, {{0, 1}}, 2);
  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {0},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata(),
                                                 greater_than(10)};
  auto iterator = task.initialize({}, ctx_ref());

  auto output = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  EXPECT_EQ(output_table->num_rows(), 0);
  EXPECT_EQ(output_table->num_columns(), 1);
  EXPECT_EQ(output_table->column(0).type(), cudf::data_type{cudf::type_id::INT32});
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, EmptyInputEmitsOneSchemaCarryingChunk)
{
  gqe::storage::chunked_in_memory_read_task task{
    {}, {0}, {cudf::data_type{cudf::type_id::INT32}}, device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  auto output = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  auto output_table = gqe::qep::to_table_view(*output);
  ASSERT_TRUE(output_table);
  EXPECT_EQ(output_table->num_rows(), 0);
  EXPECT_EQ(output_table->num_columns(), 1);
  EXPECT_EQ(output_table->column(0).type(), cudf::data_type{cudf::type_id::INT32});
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, ConcurrentEmptyInputEmitsExactlyOneSchemaCarryingChunk)
{
  gqe::storage::chunked_in_memory_read_task task{
    {}, {0}, {cudf::data_type{cudf::type_id::INT32}}, device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  std::vector<std::future<std::optional<gqe::qep::state_container>>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(
      std::async(std::launch::async, [&] { return task.next(iterator, ctx_ref()); }));
  }

  std::size_t chunks_emitted = 0;
  for (auto& worker : workers) {
    auto output = worker.get();
    if (!output) { continue; }
    ++chunks_emitted;
    auto output_table = gqe::qep::to_table_view(*output);
    ASSERT_TRUE(output_table);
    EXPECT_EQ(output_table->num_rows(), 0);
    EXPECT_EQ(output_table->num_columns(), 1);
    EXPECT_EQ(output_table->column(0).type(), cudf::data_type{cudf::type_id::INT32});
  }
  EXPECT_EQ(chunks_emitted, 1);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, CountOnlyEmitsAggregateCountOnce)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2}});
  append_row_group(row_groups, {{3, 4, 5}});

  gqe::storage::chunked_in_memory_read_task task{
    row_group_refs(row_groups), {}, {}, device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  auto output = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  EXPECT_EQ(gqe::qep::get_row_count(*output), 5);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, ConcurrentCountOnlyEmitsAggregateCountExactlyOnce)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2}});
  append_row_group(row_groups, {{3, 4, 5}});

  gqe::storage::chunked_in_memory_read_task task{
    row_group_refs(row_groups), {}, {}, device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  std::vector<std::future<std::optional<gqe::qep::state_container>>> workers;
  for (int worker_idx = 0; worker_idx < 4; ++worker_idx) {
    workers.push_back(
      std::async(std::launch::async, [&] { return task.next(iterator, ctx_ref()); }));
  }

  std::vector<cudf::size_type> counts;
  for (auto& worker : workers) {
    auto output = worker.get();
    if (output) { counts.push_back(gqe::qep::get_row_count(*output)); }
  }
  EXPECT_THAT(counts, ::testing::ElementsAre(5));
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, EmptyCountOnlyEmitsOneZeroCount)
{
  gqe::storage::chunked_in_memory_read_task task{{}, {}, {}, device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());

  auto output = task.next(iterator, ctx_ref());
  ASSERT_TRUE(output);
  EXPECT_EQ(gqe::qep::get_row_count(*output), 0);
  EXPECT_FALSE(task.next(iterator, ctx_ref()));
}

TEST_F(ChunkedInMemoryReaderTest, ConcurrentNextDrainsEachRowGroupOnce)
{
  row_group_storage row_groups;
  for (int32_t value = 0; value < 8; ++value) {
    append_row_group(row_groups, {{value}});
  }

  gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                 {0},
                                                 {cudf::data_type{cudf::type_id::INT32}},
                                                 device_source_metadata()};
  auto iterator = task.initialize({}, ctx_ref());
  cudf::get_default_stream().sync();

  std::vector<gqe::query_stream> streams;
  streams.reserve(row_groups.size());
  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    streams.push_back(get_query_ctx()->create_stream("chunked-reader-worker"));
  }

  std::barrier start{8};
  std::vector<std::future<std::optional<gqe::qep::state_container>>> workers;
  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    workers.push_back(std::async(std::launch::async, [&, i] {
      start.arrive_and_wait();
      auto output = task.next(iterator, ctx_ref(), streams[i]);
      streams[i].ref().sync();
      return output;
    }));
  }

  std::vector<gqe::qep::state_container> outputs;
  for (auto& worker : workers) {
    auto output = worker.get();
    ASSERT_TRUE(output);
    outputs.push_back(std::move(*output));
  }
  EXPECT_FALSE(task.next(iterator, ctx_ref()));

  std::vector<cudf::table_view> views;
  for (auto const& output : outputs) {
    auto table = gqe::qep::to_table_view(output);
    ASSERT_TRUE(table);
    views.push_back(*table);
  }
  auto combined = cudf::concatenate(views);
  auto sorted   = cudf::sort(combined->view());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values({0, 1, 2, 3, 4, 5, 6, 7});
  auto expected = make_table_from_columns(expected_values.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), sorted->view());
}

TEST_F(ChunkedInMemoryReaderTest, SourceSplitsPreserveProjectionAndIndependentCursors)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1}, {10}});
  append_row_group(row_groups, {{2, 3}, {20, 30}});
  append_row_group(row_groups, {{4}, {40}});
  gqe::storage::chunked_in_memory_read_task task{
    row_group_refs(row_groups),
    {1, 0},
    {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT32}},
    device_source_metadata()};
  EXPECT_EQ(task.work_unit_count(), 3);
  std::vector<gqe::qep::source_work_range> const ranges{{0, 1}, {1, 3}};
  for (auto const range : ranges) {
    auto first  = task.split(range);
    auto second = task.split(range);
    for (auto const* source : {first.get(), second.get()}) {
      auto const* residency = dynamic_cast<gqe::qep::input_residency_trait const*>(source);
      ASSERT_NE(residency, nullptr);
      EXPECT_EQ(residency->input_residency().residency, gqe::qep::data_residency::device);
    }
    auto first_iterator    = first->initialize({}, ctx_ref());
    auto second_iterator   = second->initialize({}, ctx_ref());
    auto expect_next_group = [&](gqe::qep::iterate_task const& source,
                                 gqe::qep::state_container const& iterator,
                                 std::size_t group_index) {
      auto chunk = source.next(iterator, ctx_ref());
      ASSERT_TRUE(chunk);
      auto actual = gqe::qep::to_table_view(*chunk);
      ASSERT_TRUE(actual);
      auto const& group = row_groups[group_index];
      auto const& projected_first =
        static_cast<gqe::storage::contiguous_column const&>(group.get_column(1));
      auto const& projected_second =
        static_cast<gqe::storage::contiguous_column const&>(group.get_column(0));
      CUDF_TEST_EXPECT_TABLES_EQUAL(
        (cudf::table_view{{projected_first.view(), projected_second.view()}}), *actual);
    };

    expect_next_group(*first, first_iterator, range.begin);
    expect_next_group(*second, second_iterator, range.begin);
    // Exhaust the first cursor while the second still has its remaining row groups.
    for (auto group_index = range.begin + 1; group_index < range.end; ++group_index) {
      expect_next_group(*first, first_iterator, group_index);
    }
    EXPECT_FALSE(first->next(first_iterator, ctx_ref()));
    first->finalize(std::move(first_iterator), ctx_ref());
    for (auto group_index = range.begin + 1; group_index < range.end; ++group_index) {
      expect_next_group(*second, second_iterator, group_index);
    }
    EXPECT_FALSE(second->next(second_iterator, ctx_ref()));
    second->finalize(std::move(second_iterator), ctx_ref());
  }
}

TEST_F(ChunkedInMemoryReaderTest, SourceSplitPreservesPruningAfterClone)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;
  row_group_storage row_groups;
  append_row_group(row_groups, {{100, 101}}, 2);
  append_row_group(row_groups, {{0, 1}}, 2);
  append_row_group(row_groups, {{20, 21}}, 2);
  auto split_reader = [&] {
    gqe::storage::chunked_in_memory_read_task task{row_group_refs(row_groups),
                                                   {0},
                                                   {cudf::data_type{cudf::type_id::INT32}},
                                                   device_source_metadata(),
                                                   greater_than(15)};
    return task.split({1, 3});
  }();
  auto clone    = gqe::qep::clone_qep_task_as<gqe::qep::iterate_task>(*split_reader);
  auto iterator = clone->initialize({}, ctx_ref());
  auto chunk    = clone->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto actual = gqe::qep::to_table_view(*chunk);
  ASSERT_TRUE(actual);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({20, 21});
  CUDF_TEST_EXPECT_TABLES_EQUAL((cudf::table_view{{expected}}), *actual);
  EXPECT_FALSE(clone->next(iterator, ctx_ref()));
  clone->finalize(std::move(iterator), ctx_ref());
}

TEST_F(ChunkedInMemoryReaderTest, SourceSplitCountOnlyAndInvalidRanges)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1}});
  append_row_group(row_groups, {{2, 3}});
  gqe::storage::chunked_in_memory_read_task task{
    row_group_refs(row_groups), {}, {}, device_source_metadata()};
  auto split_reader = task.split({1, 2});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  auto chunk        = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 2);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  auto empty_split    = task.split({2, 2});
  auto empty_iterator = empty_split->initialize({}, ctx_ref());
  auto empty_chunk    = empty_split->next(empty_iterator, ctx_ref());
  ASSERT_TRUE(empty_chunk);
  EXPECT_EQ(gqe::qep::get_row_count(*empty_chunk), 0);
  EXPECT_FALSE(empty_split->next(empty_iterator, ctx_ref()));
  empty_split->finalize(std::move(empty_iterator), ctx_ref());
  EXPECT_THROW(std::ignore = task.split({2, 1}), std::invalid_argument);
  EXPECT_THROW(std::ignore = task.split({0, 3}), std::invalid_argument);
  gqe::storage::chunked_in_memory_read_task empty{{}, {}, {}, device_source_metadata()};
  EXPECT_EQ(empty.work_unit_count(), 0);
  EXPECT_THROW(std::ignore = empty.split({0, 1}), std::invalid_argument);
}

TEST_F(ChunkedInMemoryReaderTest, EmptySourceSplitPreservesOutputShape)
{
  auto const type = cudf::data_type{cudf::type_id::INT32};
  gqe::storage::chunked_in_memory_read_task source{{}, {0}, {type}, device_source_metadata()};
  EXPECT_EQ(source.work_unit_count(), 0);
  auto split_reader = source.split({0, 0});
  ASSERT_NE(dynamic_cast<gqe::storage::chunked_in_memory_read_task*>(split_reader.get()), nullptr);
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
