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

#include <gqe/qep/operator/reduce_aggregate.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }

qep::state_container make_table(std::vector<std::unique_ptr<cudf::column>> columns)
{
  return qep::state_container_builder()
    .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
    .build();
}

qep::state_container make_single_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return make_table(std::move(columns));
}

qep::state_container make_indirect_reduce_input(bool is_masked)
{
  auto source = qep::state_container_builder()
                  .add_state(state_kind::cudf_column{int64_column_wrapper{10, 20, 30}.release()},
                             gqe::query_stream::default_stream())
                  .add_state(state_kind::cudf_column{int64_column_wrapper{100, 200, 300}.release()},
                             gqe::query_stream::default_stream())
                  .build();
  auto relation      = qep::indirect_table_source_state::make(std::move(source));
  auto gather_source = qep::indirect_gather_source_state::make(
    std::move(relation),
    qep::make_position_list(int32_column_wrapper{2, 0, 1}.release(),
                            gqe::query_stream::default_stream()));
  auto value  = qep::indirect_column_state::make(gather_source, 0);
  auto unused = qep::indirect_column_state::make(std::move(gather_source), 1);
  if (!is_masked) { return {std::move(value), std::move(unused)}; }
  return qep::state_container_builder()
    .add_state(state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
               gqe::query_stream::default_stream())
    .add_state(std::move(value))
    .add_state(std::move(unused))
    .build();
}

qep::state_container run_finalize(qep::reduce_aggregate_task const& task,
                                  std::vector<qep::state_container> const& chunks)
{
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  for (auto const& chunk : chunks) {
    task.next(chunk, accumulator, ctx_ref);
  }
  return task.finalize(std::move(accumulator), ctx_ref);
}

cudf::table_view result_table(qep::state_container_view result)
{
  auto table = qep::to_table_view(result);
  EXPECT_TRUE(table.has_value());
  return *table;
}

}  // namespace

TEST(QepReduceAggregateTaskTest, CountAllUsesCountOnlyChunks)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};

  auto result =
    run_finalize(task, {qep::make_row_count_container(3), qep::make_row_count_container(5)});

  int64_column_wrapper expected{8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), expected);
}

TEST(QepReduceAggregateTaskTest, SelectivelyReducesIndirectValues)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM, 0}}, {int64_type()}};
  auto result = run_finalize(task, {make_indirect_reduce_input(false)});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{60});
}

TEST(QepReduceAggregateTaskTest, PureCountAllDoesNotGatherIndirectColumns)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  auto input       = make_indirect_reduce_input(false);
  rmm::mr::statistics_resource_adaptor next_mr{cudf::get_current_device_resource_ref()};

  task.next(input, accumulator, ctx_ref, gqe::query_stream::default_stream(), next_mr);
  EXPECT_EQ(next_mr.get_allocations_counter().total, 0);
  auto result = task.finalize(std::move(accumulator), ctx_ref);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{3});
}

TEST(QepReduceAggregateTaskTest, MaskedIndirectCountAllCountsOnlySurvivingRows)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};
  auto result = run_finalize(task, {make_indirect_reduce_input(true)});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{2});
}

TEST(QepReduceAggregateTaskTest, ReducesMultiplePlainChunks)
{
  qep::reduce_aggregate_task task{
    {{cudf::aggregation::SUM, 0},
     {cudf::aggregation::MIN, 0},
     {cudf::aggregation::MAX, 0},
     {cudf::aggregation::COUNT_VALID, 0},
     {cudf::aggregation::COUNT_ALL, std::nullopt}},
    {int64_type(), int64_type(), int64_type(), int64_type(), int64_type()}};

  std::vector<std::unique_ptr<cudf::column>> first_columns;
  first_columns.push_back(
    int64_column_wrapper{{1, 20, 3, 0}, {true, false, true, false}}.release());
  std::vector<std::unique_ptr<cudf::column>> second_columns;
  second_columns.push_back(int64_column_wrapper{4, 5}.release());

  auto result = run_finalize(
    task, {make_table(std::move(first_columns)), make_table(std::move(second_columns))});
  auto table = result_table(result);

  // SUM: 1 + 3 + 4 + 5 = 13
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{13});
  // MIN: 1 (0 is invalid)
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), int64_column_wrapper{1});
  // MAX: 5 (20 is invalid)
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), int64_column_wrapper{5});
  // COUNT_VALID
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(3), int64_column_wrapper{4});
  // COUNT_ALL
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(4), int64_column_wrapper{6});
}

TEST(QepReduceAggregateTaskTest, SumOfSquaresCombinesChunkPartialsWithSum)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM_OF_SQUARES, 0}}, {int64_type()}};

  auto result = run_finalize(task,
                             {make_single_column_table(int64_column_wrapper{1, 2}.release()),
                              make_single_column_table(int64_column_wrapper{3}.release())});

  // Chunk partials are 5 and 9; their final sum is 14.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{14});
}

TEST(QepReduceAggregateTaskTest, MaskedInputSkipsRejectedRows)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM, 0},
                                   {cudf::aggregation::COUNT_VALID, 0},
                                   {cudf::aggregation::COUNT_ALL, std::nullopt}},
                                  {int64_type(), int64_type(), int64_type()}};

  auto data = make_single_column_table(
    int64_column_wrapper{{10, 20, 30, 40}, {true, true, false, true}}.release());
  auto input =
    qep::state_container_builder()
      .add_state(state_kind::valid_mask{bool_column_wrapper{true, false, true, false}.release()},
                 gqe::query_stream::default_stream())
      .add_state(data[0])
      .build();

  auto result = run_finalize(task, {input});
  auto table  = result_table(result);

  // SUM: only 10 passes all the masks
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{10});
  // COUNT_VALID: only 1 because of the qep valid_mask
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), int64_column_wrapper{1});
  // COUNT_ALL: only 2 because of the qep valid_mask
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), int64_column_wrapper{2});
}

TEST(QepReduceAggregateTaskTest, CountAllTreatsNullValidMaskRowsAsRejected)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};

  auto data = make_single_column_table(int64_column_wrapper{{10, 20}, {false, false}}.release());
  auto input =
    qep::state_container_builder()
      .add_state(
        state_kind::valid_mask{bool_column_wrapper{{true, false}, {false, false}}.release()},
        gqe::query_stream::default_stream())
      .add_state(data[0])
      .build();

  auto result = run_finalize(task, {input});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{0});
}

TEST(QepReduceAggregateTaskTest, EmptyFinalizeEmitsSqlAggregateDefaults)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM, 0},
                                   {cudf::aggregation::COUNT_VALID, 0},
                                   {cudf::aggregation::COUNT_ALL, std::nullopt}},
                                  {int64_type(), int64_type(), int64_type()}};

  auto result = run_finalize(task, {});
  auto table  = result_table(result);

  // SUM: Should be NULL because no data
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{{0}, {false}});
  // COUNT_VALID
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), int64_column_wrapper{0});
  // COUNT_ALL
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), int64_column_wrapper{0});
}

TEST(QepReduceAggregateTaskTest, UsesExplicitStreamAndResultResource)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};
  gqe::context_reference ctx_ref{};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto accumulator = task.initialize(ctx_ref, stream_handle, result_mr);
  auto input       = qep::make_row_count_container(7);
  task.next(input, accumulator, ctx_ref, stream_handle, result_mr);
  auto result = task.finalize(std::move(accumulator), ctx_ref, stream_handle, result_mr);
  stream_handle.ref().sync();

  auto table = result_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{7});
  // make sure that the memory is on the correct resource
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}

TEST(QepReduceAggregateTaskTest, EmptySumUsesExplicitStreamAndResultResource)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM, 0}}, {int64_type()}};
  gqe::context_reference ctx_ref{};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto accumulator = task.initialize(ctx_ref, stream_handle, result_mr);
  auto result      = task.finalize(std::move(accumulator), ctx_ref, stream_handle, result_mr);
  stream_handle.ref().sync();

  auto table = result_table(result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{{0}, {false}});
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}

TEST(QepReduceAggregateTaskTest, RejectsValueIndexOutsideInputColumns)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::SUM, 1}}, {int64_type()}};
  gqe::context_reference ctx_ref{};

  auto plain             = make_single_column_table(int64_column_wrapper{42}.release());
  auto plain_accumulator = task.initialize(ctx_ref);
  EXPECT_THROW(task.next(plain, plain_accumulator, ctx_ref), std::out_of_range);

  auto data   = make_single_column_table(int64_column_wrapper{42}.release());
  auto masked = qep::state_container_builder()
                  .add_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                             gqe::query_stream::default_stream())
                  .add_state(data[0])
                  .build();
  auto masked_accumulator = task.initialize(ctx_ref);
  EXPECT_THROW(task.next(masked, masked_accumulator, ctx_ref), std::out_of_range);

  qep::reduce_aggregate_task negative_task{{{cudf::aggregation::SUM, -1}}, {int64_type()}};
  auto negative_plain             = make_single_column_table(int64_column_wrapper{42}.release());
  auto negative_plain_accumulator = negative_task.initialize(ctx_ref);
  EXPECT_THROW(negative_task.next(negative_plain, negative_plain_accumulator, ctx_ref),
               std::out_of_range);

  auto negative_data   = make_single_column_table(int64_column_wrapper{42}.release());
  auto negative_masked = qep::state_container_builder()
                           .add_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                                      gqe::query_stream::default_stream())
                           .add_state(negative_data[0])
                           .build();
  auto negative_masked_accumulator = negative_task.initialize(ctx_ref);
  EXPECT_THROW(negative_task.next(negative_masked, negative_masked_accumulator, ctx_ref),
               std::out_of_range);
}

TEST(QepReduceAggregateTaskTest, RejectsInvalidConstruction)
{
  // missing type
  EXPECT_THROW((qep::reduce_aggregate_task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {}}),
               std::invalid_argument);
  // invalid type
  EXPECT_THROW(
    std::ignore = (qep::reduce_aggregate_task{{{cudf::aggregation::COUNT_ALL, std::nullopt}},
                                              {cudf::data_type{cudf::type_id::FLOAT64}}})
                    .initialize(gqe::context_reference{}),
    std::invalid_argument);
}

TEST(QepReduceAggregateTaskTest, CombinesCountOnlyAndMaskedChunks)
{
  qep::reduce_aggregate_task task{{{cudf::aggregation::COUNT_ALL, std::nullopt}}, {int64_type()}};
  auto plain  = qep::make_row_count_container(1);
  auto data   = make_single_column_table(int64_column_wrapper{42}.release());
  auto masked = qep::state_container_builder()
                  .add_state(state_kind::valid_mask{bool_column_wrapper{true}.release()},
                             gqe::query_stream::default_stream())
                  .add_state(data[0])
                  .build();

  for (bool const masked_first : {false, true}) {
    auto result =
      run_finalize(task, masked_first ? std::vector{masked, plain} : std::vector{plain, masked});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{2});
  }
}

TEST(QepReduceAggregateTaskTest, CombinesMaskedAndUnmaskedValueChunks)
{
  qep::reduce_aggregate_task task{
    {{cudf::aggregation::COUNT_ALL, std::nullopt},
     {cudf::aggregation::SUM, 1},
     {cudf::aggregation::MIN, 0},
     {cudf::aggregation::MAX, 1},
     {cudf::aggregation::COUNT_VALID, 1}},
    {int64_type(), int64_type(), int64_type(), int64_type(), int64_type()}};

  auto plain  = make_indirect_reduce_input(false);
  auto masked = make_indirect_reduce_input(true);
  for (bool const masked_first : {false, true}) {
    auto result =
      run_finalize(task, masked_first ? std::vector{masked, plain} : std::vector{plain, masked});
    auto table = result_table(result);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), int64_column_wrapper{5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), int64_column_wrapper{1100});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), int64_column_wrapper{10});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(3), int64_column_wrapper{300});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(4), int64_column_wrapper{5});
  }
}

TEST(QepReduceAggregateTaskTest, CloneIsIndependentAndEquivalent)
{
  auto original = std::make_unique<qep::reduce_aggregate_task>(
    std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>>{
      {cudf::aggregation::COUNT_ALL, std::nullopt}},
    std::vector<cudf::data_type>{int64_type()});
  auto clone = original->clone();
  original.reset();
  ASSERT_EQ(original, nullptr);

  auto* cloned_task = dynamic_cast<qep::reduce_aggregate_task*>(clone.get());
  ASSERT_NE(cloned_task, nullptr);

  auto result = run_finalize(*cloned_task, {qep::make_row_count_container(9)});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_table(result).column(0), int64_column_wrapper{9});
}
