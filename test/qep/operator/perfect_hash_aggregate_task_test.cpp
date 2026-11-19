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

#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

[[nodiscard]] cudf::data_type int32_type() { return cudf::data_type{cudf::type_id::INT32}; }

[[nodiscard]] cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }

template <typename... Columns>
[[nodiscard]] qep::state_container make_table(Columns&&... input_columns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(sizeof...(input_columns));
  (columns.push_back(std::forward<Columns>(input_columns)), ...);
  return qep::state_container_builder()
    .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
    .build();
}

template <typename... Columns>
[[nodiscard]] cudf::table make_cudf_table(Columns&&... input_columns)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(sizeof...(input_columns));
  (columns.push_back(std::forward<Columns>(input_columns)), ...);
  return cudf::table{std::move(columns)};
}

[[nodiscard]] qep::state_container make_masked_table(std::unique_ptr<cudf::column> mask,
                                                     qep::state_container const& data)
{
  qep::state_container_builder builder;
  builder.add_state(state_kind::valid_mask{std::move(mask)}, gqe::query_stream::default_stream());
  for (auto const& state : data) {
    builder.add_state(state);
  }
  return builder.build();
}

[[nodiscard]] qep::state_container make_indirect_aggregate_input(bool is_masked)
{
  auto source        = make_table(int64_column_wrapper{100, 200, 300, 400}.release(),
                           int64_column_wrapper{5, 7, 11, 3}.release(),
                           int32_column_wrapper{1, 1, 2, 1}.release());
  auto relation      = qep::indirect_table_source_state::make(std::move(source));
  auto gather_source = qep::indirect_gather_source_state::make(
    std::move(relation),
    qep::make_position_list(int32_column_wrapper{3, 0, 2, 1}.release(),
                            gqe::query_stream::default_stream()));
  qep::state_container columns;
  columns.push_back(qep::indirect_column_state::make(gather_source, 0));
  columns.push_back(qep::indirect_column_state::make(gather_source, 1));
  columns.push_back(qep::indirect_column_state::make(std::move(gather_source), 2));
  if (!is_masked) { return columns; }
  return make_masked_table(bool_column_wrapper{true, false, true, true}.release(), columns);
}

[[nodiscard]] cudf::table_view result_table(qep::state_container_view result)
{
  auto table = qep::to_table_view(result);
  if (!table.has_value()) {
    ADD_FAILURE() << "Result is not a regular-table state";
    return {};
  }
  return *table;
}

void expect_tables_equal_unordered(cudf::table_view actual, cudf::table_view expected)
{
  auto sorted_actual   = cudf::sort(actual);
  auto sorted_expected = cudf::sort(expected);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_actual->view(), sorted_expected->view());
}

[[nodiscard]] qep::state_container run_fold(qep::perfect_hash_aggregate_task const& task,
                                            std::vector<qep::state_container> const& chunks)
{
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  for (auto const& chunk : chunks) {
    task.next(chunk, accumulator, ctx_ref);
  }
  return task.finalize(std::move(accumulator), ctx_ref);
}

}  // namespace

TEST(QepPerfectHashAggregateTaskTest, PartialProducesQ1LikeGroupedResult)
{
  qep::perfect_hash_partial_aggregate_task task{
    {0, 1},
    {{cudf::aggregation::SUM, 2},
     {cudf::aggregation::COUNT_VALID, 2},
     {cudf::aggregation::COUNT_ALL, 2}},
    {int32_type(), int32_type(), int64_type(), int64_type(), int64_type()}};
  auto input = make_table(int32_column_wrapper{1, 1, 1, 2, 1}.release(),
                          int32_column_wrapper{10, 10, 20, 10, 10}.release(),
                          int64_column_wrapper{5, 7, 3, 11, 2}.release());

  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected = make_cudf_table(int32_column_wrapper{1, 1, 2}.release(),
                                  int32_column_wrapper{10, 20, 10}.release(),
                                  int64_column_wrapper{14, 3, 11}.release(),
                                  int64_column_wrapper{3, 1, 1}.release(),
                                  int64_column_wrapper{3, 1, 1}.release());

  expect_tables_equal_unordered(result_table(*result), expected.view());
  EXPECT_FALSE(qep::masked_table_view::try_from(*result).has_value());
}

TEST(QepPerfectHashAggregateTaskTest, PartialSelectivelyPacksAndRemapsIndirectInputs)
{
  qep::perfect_hash_partial_aggregate_task task{
    {2}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};

  auto result = task.next(make_indirect_aggregate_input(false), gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{15, 11}.release());
  expect_tables_equal_unordered(result_table(*result), expected.view());

  auto masked_result = task.next(make_indirect_aggregate_input(true), gqe::context_reference{});
  ASSERT_TRUE(masked_result.has_value());
  auto masked_expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{10, 11}.release());
  expect_tables_equal_unordered(result_table(*masked_result), masked_expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, CombineRemainsDirectOnly)
{
  qep::perfect_hash_aggregate_task task{
    {2}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);

  EXPECT_THROW(task.next(make_indirect_aggregate_input(false), accumulator, ctx_ref),
               std::invalid_argument);
}

TEST(QepPerfectHashAggregateTaskTest, MaskNullsAndSlicedColumnsAreHandled)
{
  int32_column_wrapper key_owner{99, 1, 2, 1, 3, 99};
  int64_column_wrapper value_owner{99, 10, 20, 30, 40, 99};
  auto const key_slice   = cudf::slice(key_owner, {1, 5}).front();
  auto const value_slice = cudf::slice(value_owner, {1, 5}).front();
  auto data              = qep::state_container_builder()
                .add_state(state_kind::cudf_column_view{key_slice})
                .add_state(state_kind::cudf_column_view{value_slice})
                .build();
  auto input = make_masked_table(
    bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(), data);
  qep::perfect_hash_partial_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};

  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 3}.release(), int64_column_wrapper{10, 40}.release());

  expect_tables_equal_unordered(result_table(*result), expected.view());

  qep::perfect_hash_aggregate_task combine{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto combined = run_fold(combine, {input, input});
  auto combined_expected =
    make_cudf_table(int32_column_wrapper{1, 3}.release(), int64_column_wrapper{20, 80}.release());
  expect_tables_equal_unordered(result_table(combined), combined_expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, RejectsNullsInRowsExcludedByValidMask)
{
  qep::perfect_hash_partial_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};

  auto null_key_data = make_table(int32_column_wrapper{{1, 2}, {true, false}}.release(),
                                  int64_column_wrapper{3, 4}.release());
  auto masked_null_key =
    make_masked_table(bool_column_wrapper{true, false}.release(), null_key_data);
  EXPECT_THROW(std::ignore = task.next(masked_null_key, gqe::context_reference{}),
               std::invalid_argument);

  auto null_value_data = make_table(int32_column_wrapper{1, 2}.release(),
                                    int64_column_wrapper{{3, 4}, {true, false}}.release());
  auto masked_null_value =
    make_masked_table(bool_column_wrapper{true, false}.release(), null_value_data);
  EXPECT_THROW(std::ignore = task.next(masked_null_value, gqe::context_reference{}),
               std::invalid_argument);
}

TEST(QepPerfectHashAggregateTaskTest, PartialPreservesSchemaForZeroGroups)
{
  qep::perfect_hash_partial_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto empty = make_table(int32_column_wrapper{}.release(), int64_column_wrapper{}.release());
  auto empty_result = task.next(empty, gqe::context_reference{});
  ASSERT_TRUE(empty_result.has_value());
  auto empty_table = result_table(*empty_result);
  ASSERT_EQ(empty_table.num_columns(), 2);
  EXPECT_EQ(empty_table.num_rows(), 0);
  EXPECT_EQ(empty_table.column(0).type(), int32_type());
  EXPECT_EQ(empty_table.column(1).type(), int64_type());

  auto data =
    make_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{3, 4}.release());
  auto all_rejected    = make_masked_table(bool_column_wrapper{false, false}.release(), data);
  auto rejected_result = task.next(all_rejected, gqe::context_reference{});
  ASSERT_TRUE(rejected_result.has_value());
  auto rejected_table = result_table(*rejected_result);
  ASSERT_EQ(rejected_table.num_columns(), 2);
  EXPECT_EQ(rejected_table.num_rows(), 0);
  EXPECT_EQ(rejected_table.column(0).type(), int32_type());
  EXPECT_EQ(rejected_table.column(1).type(), int64_type());
}

TEST(QepPerfectHashAggregateTaskTest, ApplyConcatApplyCombinesMultiplePartials)
{
  qep::perfect_hash_partial_aggregate_task partial{
    {0},
    {{cudf::aggregation::SUM, 1}, {cudf::aggregation::COUNT_ALL, 1}},
    {int32_type(), int64_type(), int64_type()}};
  auto chunk1 =
    make_table(int32_column_wrapper{1, 2, 1}.release(), int64_column_wrapper{5, 8, 7}.release());
  auto chunk2 =
    make_table(int32_column_wrapper{2, 3, 2}.release(), int64_column_wrapper{4, 9, 6}.release());
  auto partial1 = partial.next(chunk1, gqe::context_reference{});
  auto partial2 = partial.next(chunk2, gqe::context_reference{});
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());

  qep::perfect_hash_aggregate_task combine{
    {0},
    {{cudf::aggregation::SUM, 1}, {cudf::aggregation::SUM, 2}},
    {int32_type(), int64_type(), int64_type()}};
  std::vector<qep::state_container> partials;
  partials.push_back(std::move(*partial1));
  partials.push_back(std::move(*partial2));
  auto result   = run_fold(combine, partials);
  auto expected = make_cudf_table(int32_column_wrapper{1, 2, 3}.release(),
                                  int64_column_wrapper{12, 18, 9}.release(),
                                  int64_column_wrapper{2, 3, 1}.release());

  expect_tables_equal_unordered(result_table(result), expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, EmptyFoldUsesDeclaredSchema)
{
  qep::perfect_hash_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto result = run_fold(task, {});
  auto table  = result_table(result);

  ASSERT_EQ(table.num_columns(), 2);
  EXPECT_EQ(table.num_rows(), 0);
  EXPECT_EQ(table.column(0).type(), int32_type());
  EXPECT_EQ(table.column(1).type(), int64_type());
}

TEST(QepPerfectHashAggregateTaskTest, ConcurrentNextRetainsChunkOwnership)
{
  qep::perfect_hash_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{3, 5}.release()));
  chunks.push_back(
    make_table(int32_column_wrapper{2, 3}.release(), int64_column_wrapper{7, 11}.release()));
  chunks.push_back(
    make_table(int32_column_wrapper{1, 3}.release(), int64_column_wrapper{13, 17}.release()));

  std::vector<std::future<void>> calls;
  calls.reserve(chunks.size());
  for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    calls.push_back(
      std::async(std::launch::async, [&task, &chunks, &accumulator, &ctx_ref, chunk_idx] {
        task.next(chunks[chunk_idx], accumulator, ctx_ref);
      }));
  }
  for (auto& call : calls) {
    call.get();
  }
  chunks.clear();

  auto result   = task.finalize(std::move(accumulator), ctx_ref);
  auto expected = make_cudf_table(int32_column_wrapper{1, 2, 3}.release(),
                                  int64_column_wrapper{16, 12, 28}.release());
  expect_tables_equal_unordered(result_table(result), expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, EnforcesKeyTypeAndSupportsAggregateWidening)
{
  auto input =
    make_table(int32_column_wrapper{1, 1, 2}.release(), int32_column_wrapper{3, 4, 5}.release());

  qep::perfect_hash_partial_aggregate_task mismatched_key_type{
    {0}, {{cudf::aggregation::SUM, 1}}, {int64_type(), int64_type()}};
  EXPECT_THROW(std::ignore = mismatched_key_type.next(input, gqe::context_reference{}),
               std::invalid_argument);

  qep::perfect_hash_partial_aggregate_task widening_sum{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto result = widening_sum.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{7, 5}.release());
  expect_tables_equal_unordered(result_table(*result), expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, SupportsDuplicateConfigurationEntries)
{
  qep::perfect_hash_partial_aggregate_task task{
    {0, 0},
    {{cudf::aggregation::SUM, 1}, {cudf::aggregation::SUM, 1}},
    {int32_type(), int32_type(), int64_type(), int64_type()}};
  auto input =
    make_table(int32_column_wrapper{1, 1, 2}.release(), int64_column_wrapper{3, 4, 5}.release());

  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected = make_cudf_table(int32_column_wrapper{1, 2}.release(),
                                  int32_column_wrapper{1, 2}.release(),
                                  int64_column_wrapper{7, 5}.release(),
                                  int64_column_wrapper{7, 5}.release());
  expect_tables_equal_unordered(result_table(*result), expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, RejectsInvalidConfigurationAndInputs)
{
  EXPECT_THROW(
    (qep::perfect_hash_partial_aggregate_task{{}, {{cudf::aggregation::SUM, 0}}, {int64_type()}}),
    std::invalid_argument);
  EXPECT_THROW(
    (qep::perfect_hash_partial_aggregate_task{{0}, {{cudf::aggregation::SUM, 1}}, {int32_type()}}),
    std::invalid_argument);
  EXPECT_THROW((qep::perfect_hash_partial_aggregate_task{
                 {0}, {{cudf::aggregation::MAX, 1}}, {int32_type(), int64_type()}}),
               std::invalid_argument);
  EXPECT_THROW(
    (qep::perfect_hash_partial_aggregate_task{
      {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), cudf::data_type{cudf::type_id::BOOL8}}}),
    std::invalid_argument);

  qep::perfect_hash_partial_aggregate_task out_of_range{
    {1}, {{cudf::aggregation::SUM, 0}}, {int32_type(), int64_type()}};
  auto one_column = make_table(int64_column_wrapper{1}.release());
  EXPECT_THROW(std::ignore = out_of_range.next(one_column, gqe::context_reference{}),
               std::out_of_range);

  qep::perfect_hash_partial_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto null_key = make_table(int32_column_wrapper{{1, 2}, {true, false}}.release(),
                             int64_column_wrapper{3, 4}.release());
  EXPECT_THROW(std::ignore = task.next(null_key, gqe::context_reference{}), std::invalid_argument);
  auto null_value = make_table(int32_column_wrapper{1, 2}.release(),
                               int64_column_wrapper{{3, 4}, {true, false}}.release());
  EXPECT_THROW(std::ignore = task.next(null_value, gqe::context_reference{}),
               std::invalid_argument);

  qep::perfect_hash_partial_aggregate_task count_all_out_of_range{
    {0}, {{cudf::aggregation::COUNT_ALL, 2}}, {int32_type(), int64_type()}};
  EXPECT_THROW(std::ignore = count_all_out_of_range.next(null_value, gqe::context_reference{}),
               std::out_of_range);

  qep::perfect_hash_partial_aggregate_task count_all{
    {0}, {{cudf::aggregation::COUNT_ALL, 1}}, {int32_type(), int64_type()}};
  auto count_result = count_all.next(null_value, gqe::context_reference{});
  ASSERT_TRUE(count_result.has_value());
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{1, 1}.release());
  expect_tables_equal_unordered(result_table(*count_result), expected.view());
}

TEST(QepPerfectHashAggregateTaskTest, ClonesUseExplicitStreamAndSupportedResultResource)
{
  auto partial_original = std::make_unique<qep::perfect_hash_partial_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1}},
    std::vector<cudf::data_type>{int32_type(), int64_type()});
  auto combine_original = std::make_unique<qep::perfect_hash_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1}},
    std::vector<cudf::data_type>{int32_type(), int64_type()});
  auto partial_clone = partial_original->clone();
  auto combine_clone = combine_original->clone();
  partial_original.reset();
  combine_original.reset();
  auto* partial = dynamic_cast<qep::perfect_hash_partial_aggregate_task*>(partial_clone.get());
  auto* combine = dynamic_cast<qep::perfect_hash_aggregate_task*>(combine_clone.get());
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);

  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  gqe::context_reference ctx_ref{};
  auto input =
    make_table(int32_column_wrapper{1, 1, 2}.release(), int64_column_wrapper{3, 4, 5}.release());
  auto partial_result = partial->next(input, ctx_ref, stream_handle, result_mr);
  ASSERT_TRUE(partial_result.has_value());
  auto accumulator = combine->initialize(ctx_ref, stream_handle, result_mr);
  combine->next(*partial_result, accumulator, ctx_ref, stream_handle, result_mr);
  auto result = combine->finalize(std::move(accumulator), ctx_ref, stream_handle, result_mr);
  stream_handle.ref().sync();

  auto table = result_table(result);
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{7, 5}.release());
  expect_tables_equal_unordered(table, expected.view());
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}
