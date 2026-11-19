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

#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/top_k.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace qep              = gqe::qep;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

std::vector<cudf::data_type> schema()
{
  return {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}};
}

qep::state_container table(std::initializer_list<int64_t> keys,
                           std::initializer_list<int32_t> payload)
{
  return qep::state_container_builder()
    .add_state(qep::state_kind::cudf_column{int64_column_wrapper(keys).release()},
               gqe::query_stream::default_stream())
    .add_state(qep::state_kind::cudf_column{int32_column_wrapper(payload).release()},
               gqe::query_stream::default_stream())
    .build();
}

qep::state_container nullable_table(std::initializer_list<int64_t> keys,
                                    std::initializer_list<bool> validity,
                                    std::initializer_list<int32_t> payload)
{
  return qep::state_container_builder()
    .add_state(
      qep::state_kind::cudf_column{
        int64_column_wrapper(keys.begin(), keys.end(), validity.begin()).release()},
      gqe::query_stream::default_stream())
    .add_state(qep::state_kind::cudf_column{int32_column_wrapper(payload).release()},
               gqe::query_stream::default_stream())
    .build();
}

qep::state_container indirect_table(std::initializer_list<int64_t> keys,
                                    std::initializer_list<int32_t> payload,
                                    std::initializer_list<int32_t> positions,
                                    std::initializer_list<bool> valid)
{
  auto relation = qep::indirect_table_source_state::make(table(keys, payload));
  auto source   = qep::indirect_gather_source_state::make(
    std::move(relation),
    qep::make_position_list(int32_column_wrapper(positions).release(),
                            gqe::query_stream::default_stream()));
  return qep::state_container_builder()
    .add_state(qep::state_kind::valid_mask{bool_column_wrapper(valid).release()},
               gqe::query_stream::default_stream())
    .add_state(qep::indirect_column_state::make(source, 0))
    .add_state(qep::indirect_column_state::make(std::move(source), 1))
    .build();
}

qep::state_container materialize(qep::top_k_task const& task,
                                 std::span<qep::state_container const> chunks)
{
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  for (auto const& chunk : chunks) {
    task.next(chunk, accumulator, ctx_ref);
  }
  auto indirect = task.finalize(std::move(accumulator), ctx_ref);
  if (!qep::indirect_table_view::try_from(indirect).has_value()) {
    throw std::logic_error{"top_k_task::finalize did not return an indirect table"};
  }

  auto gathered = qep::gather_task{}.next(indirect, ctx_ref);
  if (!gathered.has_value()) { throw std::logic_error{"gather_task emitted no output"}; }
  return std::move(*gathered);
}

cudf::table_view as_table(qep::state_container_view result)
{
  auto view = qep::to_table_view(result);
  if (!view.has_value()) { throw std::logic_error{"top-K result is not a plain table"}; }
  return *view;
}

}  // namespace

TEST(QepTopKTaskTest, SelectsGlobalDescendingRowsAcrossChunks)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 3, schema()};
  auto result =
    materialize(task, std::array{table({5, 1, 9}, {50, 10, 90}), table({7, 3, 8}, {70, 30, 80})});

  auto view = as_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0), int64_column_wrapper{9, 8, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{90, 80, 70});
}

TEST(QepTopKTaskTest, CompactsMaskedIndirectInputBeforeCandidateSelection)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 2, schema()};
  auto input =
    indirect_table({5, 9, 7, 3}, {50, 90, 70, 30}, {3, 0, 2, 1}, {true, false, true, true});
  auto result = materialize(task, std::array{std::move(input)});

  auto view = as_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0), int64_column_wrapper{9, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{90, 70});
}

TEST(QepTopKTaskTest, SelectsRowsWithEmptyChunksBeforeAndAfterNonEmptyChunk)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 3, schema()};
  auto result = materialize(
    task, std::array{table({}, {}), table({2, 7, 3, 5}, {20, 70, 30, 50}), table({}, {})});

  auto view = as_table(result);
  ASSERT_EQ(view.num_columns(), 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0), int64_column_wrapper{7, 5, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{70, 50, 30});
}

TEST(QepTopKTaskTest, SelectsGlobalDescendingNullableRowsAcrossChunks)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 3, schema()};
  auto result = materialize(task,
                            std::array{nullable_table({5, 0, 9}, {true, false, true}, {50, 0, 90}),
                                       table({7, 3, 8}, {70, 30, 80})});

  auto view = as_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0),
                                 int64_column_wrapper{{0, 9, 8}, {false, true, true}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{0, 90, 80});
}

TEST(QepTopKTaskTest, AppliesOffsetAfterFinalAscendingOrder)
{
  qep::top_k_task task{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 2, 3, schema()};
  auto result = materialize(
    task, std::array{table({8, 1, 6}, {80, 10, 60}), table({2, 7, 3, 5}, {20, 70, 30, 50})});

  auto view = as_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0), int64_column_wrapper{3, 5, 6});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{30, 50, 60});
}

TEST(QepTopKTaskTest, AppliesOffsetAfterFinalAscendingNullableOrder)
{
  qep::top_k_task task{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 2, 3, schema()};
  auto result = materialize(task,
                            std::array{nullable_table({8, 0, 6}, {true, false, true}, {80, 0, 60}),
                                       table({2, 7, 3, 5}, {20, 70, 30, 50})});

  auto view = as_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(0),
                                 int64_column_wrapper{{5, 6, 7}, {true, true, true}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.column(1), int32_column_wrapper{50, 60, 70});
}

TEST(QepTopKTaskTest, SingleAndMultipleChunksMatchWithOffsetNullsAndDuplicateKeys)
{
  qep::top_k_task task{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 1, 6, schema()};

  auto single   = materialize(task,
                            std::array{nullable_table({4, 0, 2, 2, 1, 0, 3},
                                                        {true, false, true, true, true, false, true},
                                                        {40, 0, 20, 20, 10, 0, 30})});
  auto multiple = materialize(
    task,
    std::array{nullable_table({4, 0, 2}, {true, false, true}, {40, 0, 20}),
               nullable_table({2, 1, 0, 3}, {true, true, false, true}, {20, 10, 0, 30})});
  auto expected = nullable_table(
    {2, 2, 3, 4, 0, 0}, {true, true, true, true, false, false}, {20, 20, 30, 40, 0, 0});

  CUDF_TEST_EXPECT_TABLES_EQUAL(as_table(single), as_table(expected));
  CUDF_TEST_EXPECT_TABLES_EQUAL(as_table(multiple), as_table(expected));
}

TEST(QepTopKTaskTest, HonorsBeforeAndAfterNullPrecedence)
{
  auto input = nullable_table({0, 2, 1}, {false, true, true}, {0, 20, 10});

  qep::top_k_task before_task{0, cudf::order::ASCENDING, cudf::null_order::BEFORE, 0, 3, schema()};
  auto before = materialize(before_task, std::array{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(before).column(0),
                                 int64_column_wrapper{{0, 1, 2}, {false, true, true}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(before).column(1), int32_column_wrapper{0, 10, 20});

  qep::top_k_task after_task{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 3, schema()};
  auto after = materialize(after_task, std::array{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(after).column(0),
                                 int64_column_wrapper{{1, 2, 0}, {true, true, false}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(after).column(1), int32_column_wrapper{10, 20, 0});
}

TEST(QepTopKTaskTest, EmptyInputProducesSchemaCorrectEmptyResult)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 10, schema()};
  auto result = materialize(task, std::span<qep::state_container const>{});
  auto view   = as_table(result);
  EXPECT_EQ(view.num_columns(), 2);
  EXPECT_EQ(view.num_rows(), 0);
  EXPECT_EQ(view.column(0).type(), cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(view.column(1).type(), cudf::data_type{cudf::type_id::INT32});
}

TEST(QepTopKTaskTest, CountZeroAndOffsetPastEndProduceEmptyResults)
{
  qep::top_k_task zero{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 0, schema()};
  EXPECT_EQ(as_table(materialize(zero, std::array{table({2, 1}, {20, 10})})).num_rows(), 0);

  qep::top_k_task past_end{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 5, 2, schema()};
  EXPECT_EQ(as_table(materialize(past_end, std::array{table({2, 1}, {20, 10})})).num_rows(), 0);
}

TEST(QepTopKTaskTest, RejectsInvalidParametersAndInput)
{
  EXPECT_THROW(
    (qep::top_k_task{-1, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 1, schema()}),
    std::invalid_argument);
  EXPECT_THROW((qep::top_k_task{0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 1, {}}),
               std::invalid_argument);
  EXPECT_THROW(
    (qep::top_k_task{2, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 1, schema()}),
    std::out_of_range);
  EXPECT_NO_THROW(
    (qep::top_k_task{1, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 1, schema()}));
}

TEST(QepTopKTaskTest, CloneIsIndependentAndEquivalent)
{
  auto original = std::make_unique<qep::top_k_task>(
    0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 1, schema());
  auto clone = original->clone();
  original.reset();
  auto* cloned = dynamic_cast<qep::top_k_task*>(clone.get());
  ASSERT_NE(cloned, nullptr);

  auto result = materialize(*cloned, std::array{table({1, 4, 2}, {10, 40, 20})});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(result).column(0), int64_column_wrapper{4});
}

TEST(QepTopKTaskTest, SingleChunkRetainsNextStorageAndAllocatesPositionsDuringFinalize)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 2, schema()};
  gqe::context_reference ctx_ref{};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor next_mr{cudf::get_current_device_resource_ref()};
  rmm::mr::tracking_resource_adaptor finalize_mr{cudf::get_current_device_resource_ref()};

  auto accumulator = task.initialize(ctx_ref, stream_handle, next_mr);
  auto input       = table({1, 3, 2}, {10, 30, 20});
  task.next(input, accumulator, ctx_ref, stream_handle, next_mr);
  auto indirect = task.finalize(std::move(accumulator), ctx_ref, stream_handle, finalize_mr);
  stream_handle.ref().sync();

  // A single chunk bypasses concatenation: finalize retains next()'s candidate columns and
  // allocates only the returned position list from its result resource.
  auto indirect_view = qep::indirect_table_view::try_from(indirect);
  ASSERT_TRUE(indirect_view.has_value());
  ASSERT_FALSE(indirect_view->indirect_columns.empty());
  auto const& gather_source = indirect_view->indirect_columns.front().gather_source;
  auto const source_table   = gather_source->relation_source()->contiguous_table_view();
  for (auto const& column : source_table) {
    auto* const head = const_cast<void*>(column.head());
    EXPECT_TRUE(next_mr.get_outstanding_allocations().contains(head));
    EXPECT_FALSE(finalize_mr.get_outstanding_allocations().contains(head));
  }
  auto* const positions_head =
    const_cast<void*>(gather_source->position_list()->column().view().head());
  EXPECT_TRUE(finalize_mr.get_outstanding_allocations().contains(positions_head));
  EXPECT_FALSE(next_mr.get_outstanding_allocations().contains(positions_head));

  qep::gather_task gather;
  auto gathered = gather.next(indirect, ctx_ref, stream_handle, finalize_mr);
  ASSERT_TRUE(gathered.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(*gathered).column(0), int64_column_wrapper{3, 2});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(*gathered).column(1), int32_column_wrapper{30, 20});
}

TEST(QepTopKTaskTest, MultipleChunksAllocateConcatenatedStorageAndPositionsDuringFinalize)
{
  qep::top_k_task task{0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 2, schema()};
  gqe::context_reference ctx_ref{};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor next_mr{cudf::get_current_device_resource_ref()};
  rmm::mr::tracking_resource_adaptor finalize_mr{cudf::get_current_device_resource_ref()};

  auto accumulator = task.initialize(ctx_ref, stream_handle, next_mr);
  auto first       = table({1, 3}, {10, 30});
  auto second      = table({2, 4}, {20, 40});
  task.next(first, accumulator, ctx_ref, stream_handle, next_mr);
  task.next(second, accumulator, ctx_ref, stream_handle, next_mr);
  auto indirect = task.finalize(std::move(accumulator), ctx_ref, stream_handle, finalize_mr);
  stream_handle.ref().sync();

  // Multiple chunks are concatenated during finalize, so both the new candidate storage and the
  // returned position list use finalize's result resource.
  auto indirect_view = qep::indirect_table_view::try_from(indirect);
  ASSERT_TRUE(indirect_view.has_value());
  ASSERT_FALSE(indirect_view->indirect_columns.empty());
  auto const& gather_source = indirect_view->indirect_columns.front().gather_source;
  auto const source_table   = gather_source->relation_source()->contiguous_table_view();
  for (auto const& column : source_table) {
    auto* const head = const_cast<void*>(column.head());
    EXPECT_TRUE(finalize_mr.get_outstanding_allocations().contains(head));
    EXPECT_FALSE(next_mr.get_outstanding_allocations().contains(head));
  }
  auto* const positions_head =
    const_cast<void*>(gather_source->position_list()->column().view().head());
  EXPECT_TRUE(finalize_mr.get_outstanding_allocations().contains(positions_head));
  EXPECT_FALSE(next_mr.get_outstanding_allocations().contains(positions_head));

  auto gathered = qep::gather_task{}.next(indirect, ctx_ref, stream_handle, finalize_mr);
  ASSERT_TRUE(gathered.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(*gathered).column(0), int64_column_wrapper{4, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(as_table(*gathered).column(1), int32_column_wrapper{40, 30});
}
