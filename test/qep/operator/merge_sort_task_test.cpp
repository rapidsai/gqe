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

#include <gqe/qep/operator/merge_sort.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

class merge_sort_private_state final : public gqe::qep::task_private_state {};

gqe::qep::state_container make_owned_table(std::vector<std::unique_ptr<cudf::column>> columns)
{
  return gqe::qep::state_container_builder()
    .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
    .build();
}

gqe::qep::state_container make_two_column_table(std::initializer_list<int64_t> first,
                                                std::initializer_list<int64_t> second)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper(first).release());
  columns.push_back(int64_column_wrapper(second).release());
  return make_owned_table(std::move(columns));
}

gqe::qep::position_list_handle positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

gqe::qep::shared_state make_indirect_state()
{
  gqe::qep::state_container sources;
  sources.push_back(gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{1}.release()},
    gqe::query_stream::default_stream()));
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(sources));
  auto source   = gqe::qep::indirect_gather_source_state::make(
    std::move(relation), positions({0}), cudf::out_of_bounds_policy::DONT_CHECK);
  return gqe::qep::indirect_column_state::make(std::move(source), 0);
}

class QepMergeSortTaskTest : public gqe::test::BaseFixture {
 protected:
  QepMergeSortTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::qep::state_container run(
    gqe::qep::merge_sort_task const& task,
    std::vector<gqe::qep::state_container_view> inputs,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
  {
    auto accumulator = task.initialize(ctx_ref, stream_handle, mr);
    for (auto input : inputs) {
      task.next(input, accumulator, ctx_ref, stream_handle, mr);
    }
    return task.finalize(std::move(accumulator), ctx_ref, stream_handle, mr);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepMergeSortTaskTest, RejectsInvalidConstruction)
{
  using type     = cudf::data_type;
  auto const i64 = type{cudf::type_id::INT64};

  EXPECT_THROW(
    gqe::qep::merge_sort_task({0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, {}),
    std::invalid_argument);
  EXPECT_THROW(gqe::qep::merge_sort_task({}, {}, {}, {i64}), std::invalid_argument);
  EXPECT_THROW(gqe::qep::merge_sort_task({0}, {}, {cudf::null_order::AFTER}, {i64}),
               std::invalid_argument);
  EXPECT_THROW(gqe::qep::merge_sort_task({0}, {cudf::order::ASCENDING}, {}, {i64}),
               std::invalid_argument);
  EXPECT_THROW(
    gqe::qep::merge_sort_task({-1}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, {i64}),
    std::invalid_argument);
  EXPECT_THROW(
    gqe::qep::merge_sort_task({1}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, {i64}),
    std::invalid_argument);
}

TEST_F(QepMergeSortTaskTest, RejectsUnsupportedAndMalformedInputShapes)
{
  auto const i64 = cudf::data_type{cudf::type_id::INT64};
  gqe::qep::merge_sort_task task{{0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, {i64}};
  auto accumulator = task.initialize(ctx_ref);

  gqe::qep::state_container empty;
  EXPECT_THROW(task.next(empty, accumulator, ctx_ref), std::logic_error);

  auto count_only = gqe::qep::make_row_count_container(1);
  EXPECT_THROW(task.next(count_only, accumulator, ctx_ref), std::logic_error);

  auto masked = gqe::qep::state_container_builder()
                  .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
                             gqe::query_stream::default_stream())
                  .add_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{1}.release()},
                             gqe::query_stream::default_stream())
                  .build();
  EXPECT_THROW(task.next(masked, accumulator, ctx_ref), std::logic_error);

  gqe::qep::state_container indirect{make_indirect_state()};
  EXPECT_THROW(task.next(indirect, accumulator, ctx_ref), std::logic_error);

  gqe::qep::state_container private_input{gqe::qep::make_shared_state(
    gqe::qep::state_kind::task_private{std::make_unique<merge_sort_private_state>()})};
  EXPECT_THROW(task.next(private_input, accumulator, ctx_ref), std::logic_error);

  gqe::qep::state_container null_slot{nullptr};
  EXPECT_THROW(task.next(null_slot, accumulator, ctx_ref), std::logic_error);

  gqe::qep::state_container null_column{gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{nullptr}, gqe::query_stream::default_stream())};
  EXPECT_THROW(task.next(null_column, accumulator, ctx_ref), std::logic_error);

  auto wrong_count = make_two_column_table({1}, {2});
  EXPECT_THROW(task.next(wrong_count, accumulator, ctx_ref), std::logic_error);

  std::vector<std::unique_ptr<cudf::column>> wrong_type_columns;
  wrong_type_columns.push_back(int32_column_wrapper{1}.release());
  auto wrong_type = make_owned_table(std::move(wrong_type_columns));
  EXPECT_THROW(task.next(wrong_type, accumulator, ctx_ref), std::logic_error);

  gqe::qep::merge_sort_task two_column_task{
    {0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, {i64, i64}};
  auto two_column_accumulator = two_column_task.initialize(ctx_ref);
  gqe::qep::state_container mismatched_rows;
  mismatched_rows.push_back(gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
    gqe::query_stream::default_stream()));
  mismatched_rows.push_back(gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{3}.release()},
    gqe::query_stream::default_stream()));
  EXPECT_THROW(two_column_task.next(mismatched_rows, two_column_accumulator, ctx_ref),
               std::logic_error);
}

TEST_F(QepMergeSortTaskTest, SortsNonLeadingKeyAndPreservesLogicalColumnOrder)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{30, 10, 20}.release());
  columns.push_back(int64_column_wrapper{3, 1, 2}.release());
  columns.push_back(int64_column_wrapper{300, 100, 200}.release());
  auto input = make_owned_table(std::move(columns));
  gqe::qep::merge_sort_task task{{1},
                                 {cudf::order::ASCENDING},
                                 {cudf::null_order::AFTER},
                                 {cudf::data_type{cudf::type_id::INT64},
                                  cudf::data_type{cudf::type_id::INT64},
                                  cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {input});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_first{10, 20, 30};
  int64_column_wrapper expected_key{1, 2, 3};
  int64_column_wrapper expected_third{100, 200, 300};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_key);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_third);
}

TEST_F(QepMergeSortTaskTest, MultipleChunksAreMergedIntoOneGlobalOrdering)
{
  auto first  = make_two_column_table({1, 4, 7}, {10, 40, 70});
  auto second = make_two_column_table({2, 3, 8}, {20, 30, 80});
  gqe::qep::merge_sort_task task{
    {0},
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {first, second});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_keys{1, 2, 3, 4, 7, 8};
  int64_column_wrapper expected_payload{10, 20, 30, 40, 70, 80};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_payload);
}

TEST_F(QepMergeSortTaskTest, MixedKeyOrdersAndNullPrecedencesSortNullableColumns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    int32_column_wrapper{{2, 0, 1, 1, 0}, {true, false, true, true, false}}.release());
  columns.push_back(
    int32_column_wrapper{{0, 5, 3, 0, 0}, {false, true, true, false, false}}.release());
  columns.push_back(
    int64_column_wrapper{{20, 10, 30, 40, 50}, {true, true, false, true, true}}.release());
  auto input = make_owned_table(std::move(columns));
  gqe::qep::merge_sort_task task{{0, 1},
                                 {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                 {cudf::null_order::AFTER, cudf::null_order::BEFORE},
                                 {cudf::data_type{cudf::type_id::INT32},
                                  cudf::data_type{cudf::type_id::INT32},
                                  cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {input});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int32_column_wrapper expected_first{{1, 1, 2, 0, 0}, {true, true, true, false, false}};
  int32_column_wrapper expected_second{{3, 0, 0, 5, 0}, {true, false, false, true, false}};
  int64_column_wrapper expected_payload{{30, 40, 20, 10, 50}, {false, true, true, true, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_second);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_payload);
}

TEST_F(QepMergeSortTaskTest, RepeatedKeysProduceOneGlobalOrdering)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{30, 10, 20}.release());
  columns.push_back(int64_column_wrapper{3, 1, 2}.release());
  columns.push_back(int64_column_wrapper{300, 100, 200}.release());
  auto input = make_owned_table(std::move(columns));
  gqe::qep::merge_sort_task task{{1, 1},
                                 {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                 {cudf::null_order::AFTER, cudf::null_order::AFTER},
                                 {cudf::data_type{cudf::type_id::INT64},
                                  cudf::data_type{cudf::type_id::INT64},
                                  cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {input});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_first{10, 20, 30};
  int64_column_wrapper expected_key{1, 2, 3};
  int64_column_wrapper expected_third{100, 200, 300};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_key);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_third);
}

TEST_F(QepMergeSortTaskTest, SupportsSlicedBorrowedAndOwnedInputStates)
{
  int64_column_wrapper key_owner{9, 3, 1, 2, 8};
  auto sliced_key = cudf::slice(key_owner, {1, 4}).front();
  auto borrowed   = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced_key});
  auto owned      = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{30, 10, 20}.release()},
    gqe::query_stream::default_stream());
  gqe::qep::state_container input{owned, borrowed};
  gqe::qep::merge_sort_task task{
    {1},
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {input});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_payload{10, 20, 30};
  int64_column_wrapper expected_keys{1, 2, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_payload);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_keys);
}

TEST_F(QepMergeSortTaskTest, EmptyLifecycleResultsAreSchemaCorrect)
{
  auto const types = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                                  cudf::data_type{cudf::type_id::INT32}};
  gqe::qep::merge_sort_task task{{0}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, types};

  auto no_chunks = run(task, {});
  auto empty     = gqe::qep::to_table_view(no_chunks);
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(empty->num_columns(), 2);
  EXPECT_EQ(empty->num_rows(), 0);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_empty_column(types[0]));
  columns.push_back(cudf::make_empty_column(types[1]));
  auto zero_chunk  = make_owned_table(std::move(columns));
  auto zero_result = run(task, {zero_chunk});
  auto zero_table  = gqe::qep::to_table_view(zero_result);
  ASSERT_TRUE(zero_table.has_value());
  EXPECT_EQ(zero_table->num_columns(), 2);
  EXPECT_EQ(zero_table->num_rows(), 0);
}

TEST_F(QepMergeSortTaskTest, ProducesRegularMaterializedTable)
{
  auto input = make_two_column_table({2, 1, 1}, {1, 2, 3});
  gqe::qep::merge_sort_task task{
    {0, 1},
    {cudf::order::ASCENDING, cudf::order::DESCENDING},
    {cudf::null_order::AFTER, cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};

  auto result = run(task, {input});
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_first{1, 1, 2};
  int64_column_wrapper expected_second{3, 2, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_second);
}

TEST_F(QepMergeSortTaskTest, ResultRetainsOwnedPayloadAfterInputContainerDies)
{
  gqe::qep::merge_sort_task task{
    {0},
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  gqe::qep::state_container result;
  {
    auto input = make_two_column_table({3, 1, 2}, {30, 10, 20});
    result     = run(task, {input});
  }

  int64_column_wrapper expected_payload{10, 20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(result)->column(1), expected_payload);
}

TEST_F(QepMergeSortTaskTest, CloneIsIndependentAndEquivalent)
{
  auto original = std::make_unique<gqe::qep::merge_sort_task>(
    std::vector<cudf::size_type>{0},
    std::vector<cudf::order>{cudf::order::ASCENDING},
    std::vector<cudf::null_order>{cudf::null_order::AFTER},
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64}});
  auto clone_task = original->clone();
  original.reset();
  auto* clone = dynamic_cast<gqe::qep::merge_sort_task*>(clone_task.get());
  ASSERT_NE(clone, nullptr);
  auto input = make_two_column_table({2, 1}, {20, 10});

  auto result = run(*clone, {input});
  int64_column_wrapper expected{10, 20};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(result)->column(1), expected);
}

TEST_F(QepMergeSortTaskTest, ConcurrentNextCallsProduceOneGlobalOrdering)
{
  gqe::qep::merge_sort_task task{
    {0},
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  std::vector<gqe::qep::state_container> chunks;
  chunks.push_back(make_two_column_table({7, 3}, {70, 30}));
  chunks.push_back(make_two_column_table({8, 1}, {80, 10}));
  chunks.push_back(make_two_column_table({6, 2}, {60, 20}));
  chunks.push_back(make_two_column_table({5, 4}, {50, 40}));
  // The column wrappers produce the input chunks on the default stream. The executor normally
  // orders producer work before `next`; reproduce that contract before reading the chunks from
  // independent streams below.
  gqe::query_stream::default_stream().ref().sync();
  auto accumulator = task.initialize(ctx_ref);
  std::vector<gqe::query_stream> streams;
  streams.reserve(chunks.size());
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    streams.push_back(get_query_ctx()->create_stream("merge-sort-worker"));
  }
  std::vector<std::thread> threads;
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    threads.emplace_back(
      [&, index]() { task.next(chunks[index], accumulator, ctx_ref, streams[index]); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& stream : streams) {
    stream.ref().sync();
  }

  auto result = task.finalize(std::move(accumulator), ctx_ref);
  // `fold_finalize_adapter_task` retains the accumulator until its finalization stream is
  // synchronized during result publication. Mirror that lifecycle here, then release the
  // per-stream sorted chunks before destroying their explicit streams.
  gqe::query_stream::default_stream().ref().sync();
  accumulator.clear();
  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_keys{1, 2, 3, 4, 5, 6, 7, 8};
  int64_column_wrapper expected_payload{10, 20, 30, 40, 50, 60, 70, 80};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_payload);
}

TEST_F(QepMergeSortTaskTest, UsesExplicitStreamAndMemoryResourceForOutputAllocations)
{
  auto first  = make_two_column_table({3, 1}, {30, 10});
  auto second = make_two_column_table({4, 2}, {40, 20});
  gqe::qep::merge_sort_task task{
    {0},
    {cudf::order::ASCENDING},
    {cudf::null_order::AFTER},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor mr{cudf::get_current_device_resource_ref()};

  auto result = run(task, {first, second}, stream_handle, mr);
  stream_handle.ref().sync();
  auto table = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_TRUE(
    mr.get_outstanding_allocations().contains(const_cast<void*>(table->column(0).head())));
  EXPECT_TRUE(
    mr.get_outstanding_allocations().contains(const_cast<void*>(table->column(1).head())));
}
