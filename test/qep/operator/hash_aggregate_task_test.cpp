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

#include <gqe/qep/operator/hash_aggregate.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper       = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper      = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper      = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper     = cudf::test::strings_column_wrapper;
using decimal128_column_wrapper = cudf::test::fixed_point_column_wrapper<__int128_t>;

auto const decimal_scale = numeric::scale_type{-2};

[[nodiscard]] cudf::data_type int32_type() { return cudf::data_type{cudf::type_id::INT32}; }
[[nodiscard]] cudf::data_type int64_type() { return cudf::data_type{cudf::type_id::INT64}; }
[[nodiscard]] cudf::data_type string_type() { return cudf::data_type{cudf::type_id::STRING}; }

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

[[nodiscard]] qep::state_container run_fold(qep::hash_aggregate_task const& task,
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

TEST(QepHashAggregateTaskTest, PartialProducesMultiKeyAggregateColumnsInOrder)
{
  qep::hash_partial_aggregate_task task{{0, 1},
                                        {{cudf::aggregation::SUM, 2},
                                         {cudf::aggregation::MIN, 2},
                                         {cudf::aggregation::MAX, 2},
                                         {cudf::aggregation::COUNT_VALID, 2},
                                         {cudf::aggregation::COUNT_ALL, 2}}};
  auto input = make_table(int32_column_wrapper{1, 1, 1, 2, 1}.release(),
                          int32_column_wrapper{10, 10, 20, 10, 10}.release(),
                          int32_column_wrapper{5, 7, 3, 11, 2}.release());

  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected = make_cudf_table(int32_column_wrapper{1, 1, 2}.release(),
                                  int32_column_wrapper{10, 20, 10}.release(),
                                  int64_column_wrapper{14, 3, 11}.release(),
                                  int32_column_wrapper{2, 3, 11}.release(),
                                  int32_column_wrapper{7, 3, 11}.release(),
                                  int32_column_wrapper{3, 1, 1}.release(),
                                  int32_column_wrapper{3, 1, 1}.release());

  expect_tables_equal_unordered(result_table(*result), expected.view());
  EXPECT_FALSE(qep::masked_table_view::try_from(*result).has_value());
}

TEST(QepHashAggregateTaskTest, PartialSelectivelyPacksAndRemapsIndirectInputs)
{
  qep::hash_partial_aggregate_task task{{2}, {{cudf::aggregation::SUM, 1}}};

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

TEST(QepHashAggregateTaskTest, SlicedMaskedInputExcludesFalseAndNullMaskRows)
{
  int32_column_wrapper key_owner{{99, 1, 0, 2, 0, 99}, {true, true, false, true, false, true}};
  int32_column_wrapper value_owner{{99, 10, 20, 30, 40, 99}, {true, true, true, false, true, true}};
  auto const key_slice   = cudf::slice(key_owner, {1, 5}).front();
  auto const value_slice = cudf::slice(value_owner, {1, 5}).front();
  auto data              = qep::state_container_builder()
                .add_state(state_kind::cudf_column_view{key_slice})
                .add_state(state_kind::cudf_column_view{value_slice})
                .build();
  auto input = make_masked_table(
    bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(), data);
  qep::hash_partial_aggregate_task task{{0}, {{cudf::aggregation::SUM, 1}}};

  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected = make_cudf_table(int32_column_wrapper{{1, 0}, {true, false}}.release(),
                                  int64_column_wrapper{{10, 40}, {true, true}}.release());
  expect_tables_equal_unordered(result_table(*result), expected.view());
}

TEST(QepHashAggregateTaskTest, EmptyAndAllRejectedPartialsPreserveFirstPassSchema)
{
  qep::hash_partial_aggregate_task task{
    {0}, {{cudf::aggregation::SUM, 1}, {cudf::aggregation::COUNT_ALL, 1}}};
  auto empty = make_table(int32_column_wrapper{}.release(), int32_column_wrapper{}.release());
  auto empty_result = task.next(empty, gqe::context_reference{});
  ASSERT_TRUE(empty_result.has_value());
  auto empty_table = result_table(*empty_result);
  ASSERT_EQ(empty_table.num_columns(), 3);
  EXPECT_EQ(empty_table.num_rows(), 0);
  EXPECT_EQ(empty_table.column(0).type(), int32_type());
  EXPECT_EQ(empty_table.column(1).type(), int64_type());
  EXPECT_EQ(empty_table.column(2).type(), int32_type());

  auto data =
    make_table(int32_column_wrapper{1, 2}.release(), int32_column_wrapper{3, 4}.release());
  auto all_rejected    = make_masked_table(bool_column_wrapper{false, false}.release(), data);
  auto rejected_result = task.next(all_rejected, gqe::context_reference{});
  ASSERT_TRUE(rejected_result.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_table(*rejected_result), empty_table);
}

TEST(QepHashAggregateTaskTest, KeyOnlyAggregationCombinesMaskedAndPlainChunks)
{
  qep::hash_partial_aggregate_task partial{{0}, {}};
  auto masked_data = make_table(int32_column_wrapper{1, 2, 1}.release());
  auto masked_input =
    make_masked_table(bool_column_wrapper{true, false, true}.release(), masked_data);
  auto plain_input = make_table(int32_column_wrapper{2, 3, 2}.release());

  auto masked_partial = partial.next(masked_input, gqe::context_reference{});
  auto plain_partial  = partial.next(plain_input, gqe::context_reference{});
  ASSERT_TRUE(masked_partial.has_value());
  ASSERT_TRUE(plain_partial.has_value());

  qep::hash_aggregate_task combine{{0}, {}, {int32_type()}};
  auto result   = run_fold(combine, {*masked_partial, *plain_partial});
  auto expected = make_cudf_table(int32_column_wrapper{1, 2, 3}.release());
  expect_tables_equal_unordered(result_table(result), expected.view());
}

TEST(QepHashAggregateTaskTest, FoldCombinesOverlappingPartialsAndWidensCounts)
{
  qep::hash_partial_aggregate_task partial{
    {0}, {{cudf::aggregation::SUM, 1}, {cudf::aggregation::COUNT_ALL, 1}}};
  auto chunk1 =
    make_table(int32_column_wrapper{1, 2, 1}.release(), int32_column_wrapper{5, 8, 7}.release());
  auto chunk2 =
    make_table(int32_column_wrapper{2, 3, 2}.release(), int32_column_wrapper{4, 9, 6}.release());
  auto partial1 = partial.next(chunk1, gqe::context_reference{});
  auto partial2 = partial.next(chunk2, gqe::context_reference{});
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());
  EXPECT_EQ(result_table(*partial1).column(2).type(), int32_type());

  qep::hash_aggregate_task combine{{0},
                                   {{cudf::aggregation::SUM, 1}, {cudf::aggregation::SUM, 2}},
                                   {int32_type(), int64_type(), int64_type()}};
  auto result   = run_fold(combine, {*partial1, *partial2});
  auto expected = make_cudf_table(int32_column_wrapper{1, 2, 3}.release(),
                                  int64_column_wrapper{12, 18, 9}.release(),
                                  int64_column_wrapper{2, 3, 1}.release());
  expect_tables_equal_unordered(result_table(result), expected.view());
}

TEST(QepHashAggregateTaskTest, StringKeysRoundTripThroughPartialAndFold)
{
  qep::hash_partial_aggregate_task partial{{0}, {{cudf::aggregation::SUM, 1}}};
  auto chunk1   = make_table(string_column_wrapper{"año", "GPU", "año"}.release(),
                           int32_column_wrapper{1, 2, 3}.release());
  auto chunk2   = make_table(string_column_wrapper{"\u4e16\u754c", "GPU"}.release(),
                           int32_column_wrapper{5, 7}.release());
  auto partial1 = partial.next(chunk1, gqe::context_reference{});
  auto partial2 = partial.next(chunk2, gqe::context_reference{});
  ASSERT_TRUE(partial1.has_value());
  ASSERT_TRUE(partial2.has_value());

  qep::hash_aggregate_task combine{
    {0}, {{cudf::aggregation::SUM, 1}}, {string_type(), int64_type()}};
  auto result   = run_fold(combine, {*partial1, *partial2});
  auto expected = make_cudf_table(string_column_wrapper{"año", "GPU", "\u4e16\u754c"}.release(),
                                  int64_column_wrapper{4, 9, 5}.release());
  expect_tables_equal_unordered(result_table(result), expected.view());
}

TEST(QepHashAggregateTaskTest, EmptyFoldUsesDeclaredFinalSchema)
{
  qep::hash_aggregate_task task{{0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto result = run_fold(task, {});
  auto table  = result_table(result);
  ASSERT_EQ(table.num_columns(), 2);
  EXPECT_EQ(table.num_rows(), 0);
  EXPECT_EQ(table.column(0).type(), int32_type());
  EXPECT_EQ(table.column(1).type(), int64_type());
}

TEST(QepHashAggregateTaskTest, ConcurrentNextRetainsChunkOwnership)
{
  qep::hash_aggregate_task task{{0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int32_column_wrapper{1, 2}.release(), int32_column_wrapper{3, 5}.release()));
  chunks.push_back(
    make_table(int32_column_wrapper{2, 3}.release(), int32_column_wrapper{7, 11}.release()));
  chunks.push_back(
    make_table(int32_column_wrapper{1, 3}.release(), int32_column_wrapper{13, 17}.release()));

  std::vector<std::future<void>> calls;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    calls.push_back(std::async(std::launch::async, [&task, &chunks, &accumulator, &ctx_ref, i] {
      task.next(chunks[i], accumulator, ctx_ref);
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

TEST(QepHashAggregateTaskTest, DuplicateSpecificationsPreserveOutputOrder)
{
  qep::hash_partial_aggregate_task task{{0, 0},
                                        {{cudf::aggregation::MAX, 1}, {cudf::aggregation::SUM, 1}}};
  auto input =
    make_table(int32_column_wrapper{1, 1, 2}.release(), int32_column_wrapper{3, 4, 5}.release());
  auto result = task.next(input, gqe::context_reference{});
  ASSERT_TRUE(result.has_value());
  auto expected = make_cudf_table(int32_column_wrapper{1, 2}.release(),
                                  int32_column_wrapper{1, 2}.release(),
                                  int32_column_wrapper{4, 5}.release(),
                                  int64_column_wrapper{7, 5}.release());
  expect_tables_equal_unordered(result_table(*result), expected.view());
}

TEST(QepHashAggregateTaskTest, RejectsInvalidConfigurationAndInputShapes)
{
  EXPECT_THROW((qep::hash_partial_aggregate_task{{}, {{cudf::aggregation::SUM, 0}}}),
               std::invalid_argument);
  EXPECT_THROW((qep::hash_partial_aggregate_task{{0}, {{cudf::aggregation::PRODUCT, 1}}}),
               std::invalid_argument);
  EXPECT_THROW((qep::hash_partial_aggregate_task{{0}, {{cudf::aggregation::MEAN, 1}}}),
               std::invalid_argument);
  EXPECT_THROW((qep::hash_aggregate_task{
                 {0}, {{cudf::aggregation::PRODUCT, 1}}, {int32_type(), int64_type()}}),
               std::invalid_argument);
  EXPECT_THROW((qep::hash_aggregate_task{{0}, {{cudf::aggregation::SUM, 1}}, {int32_type()}}),
               std::invalid_argument);
  EXPECT_THROW((qep::hash_aggregate_task{
                 {0}, {{cudf::aggregation::COUNT_ALL, 1}}, {int32_type(), int32_type()}}),
               std::invalid_argument);

  qep::hash_partial_aggregate_task out_of_range{{1}, {{cudf::aggregation::SUM, 0}}};
  auto one_column = make_table(int32_column_wrapper{1}.release());
  EXPECT_THROW(std::ignore = out_of_range.next(one_column, gqe::context_reference{}),
               std::out_of_range);

  qep::hash_aggregate_task fold{{0}, {{cudf::aggregation::SUM, 1}}, {int32_type(), int64_type()}};
  auto malformed   = make_table(int32_column_wrapper{1}.release());
  auto accumulator = fold.initialize(gqe::context_reference{});
  EXPECT_THROW(fold.next(malformed, accumulator, gqe::context_reference{}), std::invalid_argument);
  auto wrong_types =
    make_table(int64_column_wrapper{1}.release(), int32_column_wrapper{2}.release());
  EXPECT_THROW(fold.next(wrong_types, accumulator, gqe::context_reference{}),
               std::invalid_argument);
  auto valid = make_table(int32_column_wrapper{1}.release(), int32_column_wrapper{2}.release());
  EXPECT_THROW(fold.next(valid, {}, gqe::context_reference{}), std::logic_error);
}

TEST(QepHashAggregateTaskTest, ClonesUseExplicitStreamAndResultResource)
{
  auto partial_original = std::make_unique<qep::hash_partial_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1}});
  auto combine_original = std::make_unique<qep::hash_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1}},
    std::vector<cudf::data_type>{int32_type(), int64_type()});
  auto partial_clone = partial_original->clone();
  auto combine_clone = combine_original->clone();
  partial_original.reset();
  combine_original.reset();
  auto* partial = dynamic_cast<qep::hash_partial_aggregate_task*>(partial_clone.get());
  auto* combine = dynamic_cast<qep::hash_aggregate_task*>(combine_clone.get());
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(combine, nullptr);

  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  gqe::context_reference ctx_ref{};
  auto data  = make_table(int32_column_wrapper{1, 1, 2, 3}.release(),
                         int32_column_wrapper{3, 4, 5, 7}.release());
  auto input = make_masked_table(
    bool_column_wrapper{{true, true, true, true}, {true, false, true, false}}.release(), data);
  auto partial_result = partial->next(input, ctx_ref, stream_handle, result_mr);
  ASSERT_TRUE(partial_result.has_value());
  auto accumulator = combine->initialize(ctx_ref, stream_handle, result_mr);
  combine->next(*partial_result, accumulator, ctx_ref, stream_handle, result_mr);
  auto result = combine->finalize(std::move(accumulator), ctx_ref, stream_handle, result_mr);
  stream_handle.ref().sync();

  auto table = result_table(result);
  auto expected =
    make_cudf_table(int32_column_wrapper{1, 2}.release(), int64_column_wrapper{3, 5}.release());
  expect_tables_equal_unordered(table, expected.view());
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}

TEST(QepHashAggregateTaskTest, Decimal128SumAligns16Bytes)
{
  // Use 9 column keys to force alignment errors in 8-byte alignment.
  qep::hash_partial_aggregate_task partial{{0},
                                           {{cudf::aggregation::SUM, 1},
                                            {cudf::aggregation::SUM, 2},
                                            {cudf::aggregation::SUM, 3},
                                            {cudf::aggregation::SUM, 4},
                                            {cudf::aggregation::SUM, 5}}};
  auto chunk =
    make_table(int32_column_wrapper{1, 2, 1, 3, 4, 5, 6, 7, 8, 9}.release(),
               int32_column_wrapper{{1, 2, 3, 4, 0, 0, 0, 0, 0, 0}}.release(),
               decimal128_column_wrapper{{5, 6, 7, 8, 0, 0, 0, 0, 0, 0}, decimal_scale}.release(),
               int64_column_wrapper{{1, 2, 3, 4, 0, 0, 0, 0, 0, 0}}.release(),
               int64_column_wrapper{{1, 2, 3, 4, 0, 0, 0, 0, 0, 0}}.release(),
               decimal128_column_wrapper{{5, 6, 7, 8, 0, 0, 0, 0, 0, 0}, decimal_scale}.release());

  auto partial_aggr = partial.next(chunk, gqe::context_reference{});

  auto result = result_table(*partial_aggr);

  auto expected = make_cudf_table(
    int32_column_wrapper{{1, 2, 3, 4, 5, 6, 7, 8, 9}}.release(),
    int64_column_wrapper{{4, 2, 4, 0, 0, 0, 0, 0, 0}}.release(),
    decimal128_column_wrapper{{12, 6, 8, 0, 0, 0, 0, 0, 0}, decimal_scale}.release(),
    int64_column_wrapper{{4, 2, 4, 0, 0, 0, 0, 0, 0}}.release(),
    int64_column_wrapper{{4, 2, 4, 0, 0, 0, 0, 0, 0}}.release(),
    decimal128_column_wrapper{{12, 6, 8, 0, 0, 0, 0, 0, 0}, decimal_scale}.release());
  expect_tables_equal_unordered(result, expected.view());
}
