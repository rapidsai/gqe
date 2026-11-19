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

#include "../lifetime_probe.hpp"

#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <arrow/api.h>

#include <array>
#include <barrier>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

cudf::table one_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

qep::state_container owned_table(
  std::unique_ptr<cudf::column> column,
  gqe::query_stream const& stream_handle = gqe::query_stream::default_stream())
{
  return qep::state_container_builder()
    .add_state(one_column_table(std::move(column)), stream_handle)
    .build();
}

class IntermediateStateWriteTaskTest : public gqe::test::BaseFixture {
 protected:
  IntermediateStateWriteTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  qep::state_container run(std::vector<qep::state_container> const& chunks,
                           std::vector<cudf::data_type> output_data_types)
  {
    qep::intermediate_state_write_task task{std::move(output_data_types)};
    auto accumulator = task.initialize(ctx_ref);
    for (auto const& chunk : chunks) {
      task.next(chunk, accumulator, ctx_ref);
    }
    return task.finalize(std::move(accumulator), ctx_ref);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(IntermediateStateWriteTaskTest, SingleRegularChunkTransfersOwnedAndBorrowedSlots)
{
  int64_column_wrapper borrowed_owner{10, 20, 30};
  auto input = qep::state_container_builder()
                 .add_state(state_kind::cudf_column{int32_column_wrapper{1, 2, 3}.release()},
                            gqe::query_stream::default_stream())
                 .add_state(state_kind::cudf_column_view{borrowed_owner})
                 .build();
  auto const owned_slot    = input[0];
  auto const borrowed_slot = input[1];

  auto result =
    run({input}, {cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::INT64}});

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], owned_slot);
  EXPECT_EQ(result[1], borrowed_slot);
  EXPECT_TRUE(std::holds_alternative<state_kind::cudf_column>(*result[0]));
  EXPECT_TRUE(std::holds_alternative<state_kind::cudf_column_view>(*result[1]));
}

TEST_F(IntermediateStateWriteTaskTest, ConcatenatesNullableSlicedStringsAndNestedColumns)
{
  string_column_wrapper string_owner({"skip", "héllo", "世界", "skip"}, {true, true, false, true});
  cudf::test::lists_column_wrapper<int64_t> list_owner{{9}, {1, 2}, {3}, {8}};
  auto const strings = cudf::slice(string_owner, {1, 3}).front();
  auto const lists   = cudf::slice(list_owner, {1, 3}).front();
  int64_column_wrapper second_value_owner{30};
  ASSERT_EQ(strings.size(), 2);
  ASSERT_EQ(lists.size(), 2);

  auto first = qep::state_container_builder()
                 .add_state(state_kind::cudf_column_view{strings})
                 .add_state(state_kind::cudf_column{int64_column_wrapper{10, 20}.release()},
                            gqe::query_stream::default_stream())
                 .add_state(state_kind::cudf_column_view{lists})
                 .build();
  auto second =
    qep::state_container_builder()
      .add_state(state_kind::cudf_column{string_column_wrapper({"GPU"}, {true}).release()},
                 gqe::query_stream::default_stream())
      .add_state(state_kind::cudf_column_view{second_value_owner})
      .add_state(
        state_kind::cudf_column{cudf::test::lists_column_wrapper<int64_t>{{4, 5}}.release()},
        gqe::query_stream::default_stream())
      .build();

  auto result = run({first, second},
                    {cudf::data_type{cudf::type_id::STRING},
                     cudf::data_type{cudf::type_id::INT64},
                     cudf::data_type{cudf::type_id::LIST}});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());

  string_column_wrapper expected_strings({"héllo", "", "GPU"}, {true, false, true});
  int64_column_wrapper expected_values{10, 20, 30};
  cudf::test::lists_column_wrapper<int64_t> expected_lists{{1, 2}, {3}, {4, 5}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_strings);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_values);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_lists);
}

TEST_F(IntermediateStateWriteTaskTest, PreservesZeroRowRegularSchema)
{
  auto first  = owned_table(int64_column_wrapper{}.release());
  auto second = owned_table(int64_column_wrapper{}.release());

  auto result = run({first, second}, {cudf::data_type{cudf::type_id::INT64}});
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->column(0).type().id(), cudf::type_id::INT64);
}

TEST_F(IntermediateStateWriteTaskTest, SumsCountOnlyChunksAndChecksOverflow)
{
  auto result = run({qep::make_row_count_container(0),
                     qep::make_row_count_container(3),
                     qep::make_row_count_container(4)},
                    {});
  ASSERT_TRUE(qep::try_row_count(result).has_value());
  EXPECT_EQ(*qep::try_row_count(result), 7);

  qep::intermediate_state_write_task task{std::vector<cudf::data_type>{}};
  auto accumulator = task.initialize(ctx_ref);
  auto maximum     = qep::make_row_count_container(std::numeric_limits<cudf::size_type>::max());
  auto one         = qep::make_row_count_container(1);
  task.next(maximum, accumulator, ctx_ref);
  EXPECT_THROW(task.next(one, accumulator, ctx_ref), std::overflow_error);
}

TEST_F(IntermediateStateWriteTaskTest, RejectsMismatchedAndMixedChunks)
{
  qep::intermediate_state_write_task task{{cudf::data_type{cudf::type_id::INT64}}};

  auto accumulator = task.initialize(ctx_ref);
  auto int64_input = owned_table(int64_column_wrapper{1}.release());
  auto int32_input = owned_table(int32_column_wrapper{2}.release());
  task.next(int64_input, accumulator, ctx_ref);
  EXPECT_THROW(task.next(int32_input, accumulator, ctx_ref), std::logic_error);

  accumulator = task.initialize(ctx_ref);
  task.next(int64_input, accumulator, ctx_ref);
  auto two_columns = qep::state_container_builder()
                       .add_state(state_kind::cudf_column{int64_column_wrapper{1}.release()},
                                  gqe::query_stream::default_stream())
                       .add_state(state_kind::cudf_column{int64_column_wrapper{2}.release()},
                                  gqe::query_stream::default_stream())
                       .build();
  EXPECT_THROW(task.next(two_columns, accumulator, ctx_ref), std::logic_error);

  accumulator = task.initialize(ctx_ref);
  task.next(int64_input, accumulator, ctx_ref);
  auto count = qep::make_row_count_container(1);
  EXPECT_THROW(task.next(count, accumulator, ctx_ref), std::logic_error);

  qep::intermediate_state_write_task count_task{std::vector<cudf::data_type>{}};
  accumulator = count_task.initialize(ctx_ref);
  EXPECT_THROW(count_task.next(int64_input, accumulator, ctx_ref), std::logic_error);
}

TEST_F(IntermediateStateWriteTaskTest, RejectsUnsupportedShapesAndFinalizesAbsentInput)
{
  qep::intermediate_state_write_task task{{cudf::data_type{cudf::type_id::INT64}}};
  auto accumulator = task.initialize(ctx_ref);

  qep::state_container empty;
  task.next(empty, accumulator, ctx_ref);

  qep::state_container null_slot{nullptr};
  EXPECT_THROW(task.next(null_slot, accumulator, ctx_ref), std::logic_error);

  auto owned_masked =
    qep::state_container_builder()
      .add_state(state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
                 gqe::query_stream::default_stream())
      .add_state(state_kind::cudf_column{int64_column_wrapper{1, 2}.release()},
                 gqe::query_stream::default_stream())
      .build();
  EXPECT_THROW(task.next(owned_masked, accumulator, ctx_ref), std::logic_error);

  bool_column_wrapper mask_owner{true, false};
  int64_column_wrapper data_owner{1, 2};
  auto borrowed_masked = qep::state_container_builder()
                           .add_state(state_kind::valid_mask_view{mask_owner})
                           .add_state(state_kind::cudf_column_view{data_owner})
                           .build();
  EXPECT_THROW(task.next(borrowed_masked, accumulator, ctx_ref), std::logic_error);

  auto source =
    qep::make_shared_state(state_kind::cudf_column{int64_column_wrapper{10, 20}.release()},
                           gqe::query_stream::default_stream());
  auto relation      = qep::indirect_table_source_state::make({source});
  auto gather_source = qep::indirect_gather_source_state::make(
    relation,
    qep::make_position_list(int32_column_wrapper{1, 0}.release(),
                            gqe::query_stream::default_stream()));
  qep::state_container indirect{qep::indirect_column_state::make(gather_source, 0)};
  EXPECT_THROW(task.next(indirect, accumulator, ctx_ref), std::logic_error);

  auto result = task.finalize(std::move(accumulator), ctx_ref);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->column(0).type().id(), cudf::type_id::INT64);

  qep::intermediate_state_write_task count_task{std::vector<cudf::data_type>{}};
  auto count_result = count_task.finalize(count_task.initialize(ctx_ref), ctx_ref);
  ASSERT_TRUE(qep::try_row_count(count_result).has_value());
  EXPECT_EQ(*qep::try_row_count(count_result), 0);
}

TEST_F(IntermediateStateWriteTaskTest, ConcurrentNextRetainsEveryChunk)
{
  constexpr std::size_t num_chunks = 16;
  qep::intermediate_state_write_task task{{cudf::data_type{cudf::type_id::INT64}}};
  auto accumulator = task.initialize(ctx_ref);
  std::vector<qep::state_container> chunks;
  chunks.reserve(num_chunks);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    chunks.push_back(owned_table(int64_column_wrapper{static_cast<int64_t>(i)}.release()));
  }

  std::barrier start{num_chunks};
  std::vector<std::thread> threads;
  threads.reserve(num_chunks);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    threads.emplace_back([&, i] {
      start.arrive_and_wait();
      task.next(chunks[i], accumulator, ctx_ref);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  auto result = task.finalize(std::move(accumulator), ctx_ref);
  auto sorted = cudf::sort(*qep::to_table_view(result));
  std::vector<int64_t> expected_values(num_chunks);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    expected_values[i] = static_cast<int64_t>(i);
  }
  int64_column_wrapper expected(expected_values.begin(), expected_values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(IntermediateStateWriteTaskTest, MultiChunkFinalizeKeepsInputsInAccumulator)
{
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  qep::intermediate_state_write_task task{{cudf::data_type{cudf::type_id::INT64}}};
  auto accumulator         = task.initialize(ctx_ref, stream_handle);
  auto first               = owned_table(int64_column_wrapper{1, 2}.release());
  auto second              = owned_table(int64_column_wrapper{3, 4}.release());
  bool is_first_destroyed  = false;
  bool is_second_destroyed = false;
  first.front()  = gqe::test::track_lifetime(std::move(first.front()), is_first_destroyed);
  second.front() = gqe::test::track_lifetime(std::move(second.front()), is_second_destroyed);

  task.next(first, accumulator, ctx_ref, stream_handle);
  task.next(second, accumulator, ctx_ref, stream_handle);
  first.clear();
  second.clear();

  auto result = task.finalize(std::move(accumulator), ctx_ref, stream_handle);
  EXPECT_FALSE(is_first_destroyed);
  EXPECT_FALSE(is_second_destroyed);

  stream_handle.ref().sync();
  accumulator.clear();
  EXPECT_TRUE(is_first_destroyed);
  EXPECT_TRUE(is_second_destroyed);

  int64_column_wrapper expected{1, 2, 3, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(result)->column(0), expected);
}

/** @brief Query-owned pipeline output remains valid after its context is replaced by Arrow data. */
TEST_F(IntermediateStateWriteTaskTest, QueryOwnedStreamResultSurvivesArrowHandoff)
{
  std::shared_ptr<arrow::RecordBatch> arrow_result;
  {
    gqe::query_context query_context{gqe::make_optimization_parameters(true)};
    gqe::context_reference local_ctx_ref{get_task_manager_ctx(), &query_context};
    auto const producer_stream = query_context.create_stream("pipeline=1/intermediate-write");
    auto const mr              = cudf::get_current_device_resource_ref();

    int64_column_wrapper first_source{1, 2};
    int64_column_wrapper second_source{3, 4};
    cudf::get_default_stream().sync();
    auto first = owned_table(
      std::make_unique<cudf::column>(first_source, producer_stream.ref(), mr), producer_stream);
    auto second = owned_table(
      std::make_unique<cudf::column>(second_source, producer_stream.ref(), mr), producer_stream);

    qep::intermediate_state_write_task writer{{cudf::data_type{cudf::type_id::INT64}}};
    auto accumulator = writer.initialize(local_ctx_ref, producer_stream, mr);
    writer.next(first, accumulator, local_ctx_ref, producer_stream, mr);
    writer.next(second, accumulator, local_ctx_ref, producer_stream, mr);
    auto result = writer.finalize(std::move(accumulator), local_ctx_ref, producer_stream, mr);
    auto const result_view = qep::to_table_view(result);
    ASSERT_TRUE(result_view.has_value());

    auto const serialization_stream = query_context.create_stream("result-serialization");
    query_context.close_and_synchronize_streams();
    std::array<cudf::table_view, 1> result_views{*result_view};
    std::array<std::string, 1> column_names{"value"};
    arrow_result =
      gqe::utility::cudf_table_to_arrow(result_views, column_names, serialization_stream.ref(), mr);

    result.clear();
    accumulator.clear();
  }

  ASSERT_NE(arrow_result, nullptr);
  auto values = std::dynamic_pointer_cast<arrow::Int64Array>(arrow_result->column(0));
  ASSERT_NE(values, nullptr);
  ASSERT_EQ(values->length(), 4);
  EXPECT_EQ(values->Value(0), 1);
  EXPECT_EQ(values->Value(1), 2);
  EXPECT_EQ(values->Value(2), 3);
  EXPECT_EQ(values->Value(3), 4);
}

TEST_F(IntermediateStateWriteTaskTest, CloneUsesIndependentAccumulatorAndExplicitResultResource)
{
  qep::intermediate_state_write_task task{{cudf::data_type{cudf::type_id::INT64}}};
  auto clone_base = task.clone();
  auto* clone     = dynamic_cast<qep::intermediate_state_write_task*>(clone_base.get());
  ASSERT_NE(clone, nullptr);

  auto first_accumulator  = task.initialize(ctx_ref);
  auto second_accumulator = clone->initialize(ctx_ref);
  auto first              = owned_table(int64_column_wrapper{1, 2}.release());
  auto second             = owned_table(int64_column_wrapper{3}.release());
  task.next(first, first_accumulator, ctx_ref);
  clone->next(second, second_accumulator, ctx_ref);
  EXPECT_EQ(qep::get_row_count(task.finalize(std::move(first_accumulator), ctx_ref)), 2);
  EXPECT_EQ(qep::get_row_count(clone->finalize(std::move(second_accumulator), ctx_ref)), 1);

  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto accumulator = task.initialize(ctx_ref, stream_handle, result_mr);
  auto third       = owned_table(int64_column_wrapper{4}.release());
  auto fourth      = owned_table(int64_column_wrapper{5}.release());
  task.next(third, accumulator, ctx_ref, stream_handle, result_mr);
  task.next(fourth, accumulator, ctx_ref, stream_handle, result_mr);
  auto result = task.finalize(std::move(accumulator), ctx_ref, stream_handle, result_mr);
  stream_handle.ref().sync();
  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table->column(0).head())));
}

TEST_F(IntermediateStateWriteTaskTest, RecombinesBoundedIntermediateReaderOutputs)
{
  auto input = owned_table(int64_column_wrapper{1, 2, 3, 4, 5, 6, 7}.release());
  qep::intermediate_state_read_task reader{3};
  auto iterator = reader.initialize(input, ctx_ref);
  qep::intermediate_state_write_task writer{{cudf::data_type{cudf::type_id::INT64}}};
  auto accumulator = writer.initialize(ctx_ref);

  while (auto chunk = reader.next(iterator, ctx_ref)) {
    writer.next(*chunk, accumulator, ctx_ref);
  }
  reader.finalize(std::move(iterator), ctx_ref);
  auto result = writer.finalize(std::move(accumulator), ctx_ref);

  int64_column_wrapper expected{1, 2, 3, 4, 5, 6, 7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(result)->column(0), expected);
}
