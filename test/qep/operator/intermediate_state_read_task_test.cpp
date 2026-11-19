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

#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

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

gqe::qep::shared_state owned_int64(std::initializer_list<int64_t> values)
{
  return gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values).release()},
    gqe::query_stream::default_stream());
}

gqe::qep::position_list_handle positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

gqe::qep::indirect_gather_source gather_source(
  gqe::qep::state_container sources,
  gqe::qep::position_list_handle map,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(sources));
  return gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(map), policy);
}

gqe::qep::shared_state indirect(gqe::qep::indirect_gather_source source,
                                cudf::size_type source_column_index = 0)
{
  return gqe::qep::indirect_column_state::make(std::move(source), source_column_index);
}

/** @brief Foreign private state used to verify that the reader rejects another task's iterator. */
class unrelated_private_state final : public gqe::qep::task_private_state {};

/** @brief Fixture providing a QEP context and a helper that drains a reader iterator. */
class IntermediateStateReadTaskTest : public gqe::test::BaseFixture {
 protected:
  IntermediateStateReadTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  std::vector<gqe::qep::state_container> drain(
    gqe::qep::intermediate_state_read_task const& task,
    gqe::qep::state_container& iterator,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream())
  {
    std::vector<gqe::qep::state_container> output;
    while (auto chunk = task.next(iterator, ctx_ref, stream_handle)) {
      output.push_back(std::move(*chunk));
    }
    return output;
  }

  gqe::context_reference ctx_ref;
};

TEST_F(IntermediateStateReadTaskTest, PlainTableEmitsBoundedBorrowedChunks)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{0, 1, 2, 3, 4, 5, 6}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::intermediate_state_read_task task{3};
  auto iterator = task.initialize(input, ctx_ref);
  auto chunks   = drain(task, iterator);

  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(gqe::qep::get_row_count(chunks[0]), 3);
  EXPECT_EQ(gqe::qep::get_row_count(chunks[1]), 3);
  EXPECT_EQ(gqe::qep::get_row_count(chunks[2]), 1);
  for (auto const& chunk : chunks) {
    ASSERT_EQ(chunk.size(), 1);
    EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*chunk.front()));
  }

  int64_column_wrapper expected0{0, 1, 2};
  int64_column_wrapper expected1{3, 4, 5};
  int64_column_wrapper expected2{6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(chunks[0])->column(0), expected0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(chunks[1])->column(0), expected1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(chunks[2])->column(0), expected2);
}

TEST_F(IntermediateStateReadTaskTest, ExactLimitProducesTwoChunks)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int32_column_wrapper{1, 2, 3, 4, 5, 6}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::intermediate_state_read_task task{3};
  auto iterator = task.initialize(input, ctx_ref);

  EXPECT_EQ(drain(task, iterator).size(), 2);
}

TEST_F(IntermediateStateReadTaskTest, CompleteRangePlainOutputPreservesOriginalViewMetadata)
{
  int64_column_wrapper owner({10, 20, 30, 40, 50}, {true, false, true, false, true});
  auto const source = cudf::slice(owner, {1, 4}).front();
  ASSERT_EQ(source.offset(), 1);
  ASSERT_EQ(source.null_count(), 2);

  auto assert_complete_view = [&](gqe::qep::intermediate_state_read_task const& task) {
    auto input = gqe::qep::state_container_builder()
                   .add_state(gqe::qep::state_kind::cudf_column_view{source})
                   .build();
    auto iterator     = task.initialize(input, ctx_ref);
    auto maybe_output = task.next(iterator, ctx_ref);
    ASSERT_TRUE(maybe_output.has_value());
    ASSERT_FALSE(task.next(iterator, ctx_ref).has_value());
    auto const table = gqe::qep::to_table_view(*maybe_output);
    ASSERT_TRUE(table.has_value());
    auto const result = table->column(0);
    EXPECT_EQ(result.head<int64_t>(), source.head<int64_t>());
    EXPECT_EQ(result.null_mask(), source.null_mask());
    EXPECT_EQ(result.offset(), source.offset());
    EXPECT_EQ(result.size(), source.size());
    EXPECT_EQ(result.null_count(), source.null_count());
  };

  gqe::qep::intermediate_state_read_task single{
    gqe::qep::intermediate_state_read_task::single_output};
  assert_complete_view(single);

  gqe::qep::intermediate_state_read_task oversized{10};
  assert_complete_view(oversized);
}

TEST_F(IntermediateStateReadTaskTest, CompleteRangeMaskedOutputPreservesOriginalViewMetadata)
{
  bool_column_wrapper mask_owner({true, false, true, true, false}, {true, true, false, true, true});
  int64_column_wrapper data_owner({10, 20, 30, 40, 50}, {true, false, true, true, true});
  auto const source_mask = cudf::slice(mask_owner, {1, 4}).front();
  auto const source_data = cudf::slice(data_owner, {1, 4}).front();
  auto input             = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask_view{source_mask})
                 .add_state(gqe::qep::state_kind::cudf_column_view{source_data})
                 .build();
  gqe::qep::intermediate_state_read_task task{
    gqe::qep::intermediate_state_read_task::single_output};
  auto iterator = task.initialize(input, ctx_ref);
  auto output   = task.next(iterator, ctx_ref);
  ASSERT_TRUE(output.has_value());
  auto result = gqe::qep::masked_table_view::try_from(*output);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result->mask.head<bool>(), source_mask.head<bool>());
  EXPECT_EQ(result->mask.null_mask(), source_mask.null_mask());
  EXPECT_EQ(result->mask.offset(), source_mask.offset());
  EXPECT_EQ(result->mask.null_count(), source_mask.null_count());
  EXPECT_EQ(result->columns.column(0).head<int64_t>(), source_data.head<int64_t>());
  EXPECT_EQ(result->columns.column(0).null_mask(), source_data.null_mask());
  EXPECT_EQ(result->columns.column(0).offset(), source_data.offset());
  EXPECT_EQ(result->columns.column(0).null_count(), source_data.null_count());
}

TEST_F(IntermediateStateReadTaskTest, PreservesNullsStringsAndSlicedBorrowedColumns)
{
  int64_column_wrapper borrowed_owner{0, 10, 20, 30, 40};
  auto const borrowed = cudf::slice(borrowed_owner, {1, 4}).front();
  cudf::test::lists_column_wrapper<int64_t> lists_owner{{0}, {1, 2}, {3}, {4, 5}, {6}};
  auto const lists = cudf::slice(lists_owner, {1, 4}).front();
  auto input       = gqe::qep::state_container_builder()
                 .add_state(
                   gqe::qep::state_kind::cudf_column{
                     string_column_wrapper({"one", "deux", "三"}, {true, false, true}).release()},
                   gqe::query_stream::default_stream())
                 .add_state(gqe::qep::state_kind::cudf_column_view{borrowed})
                 .add_state(gqe::qep::state_kind::cudf_column_view{lists})
                 .build();
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  gqe::qep::intermediate_state_read_task task{2};
  auto iterator = task.initialize(input, ctx_ref, stream_handle);
  auto chunks   = drain(task, iterator, stream_handle);
  stream_handle.ref().sync();

  ASSERT_EQ(chunks.size(), 2);
  auto first  = *gqe::qep::to_table_view(chunks[0]);
  auto second = *gqe::qep::to_table_view(chunks[1]);
  string_column_wrapper expected_strings0({"one", ""}, {true, false});
  string_column_wrapper expected_strings1({"三"}, {true});
  int64_column_wrapper expected_values0{10, 20};
  int64_column_wrapper expected_values1{30};
  cudf::test::lists_column_wrapper<int64_t> expected_lists0{{1, 2}, {3}};
  cudf::test::lists_column_wrapper<int64_t> expected_lists1{{4, 5}};
  EXPECT_EQ(first.column(0).null_count(), 1);
  EXPECT_EQ(second.column(0).null_count(), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first.column(0), expected_strings0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second.column(0), expected_strings1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first.column(1), expected_values0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second.column(1), expected_values1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first.column(2), expected_lists0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second.column(2), expected_lists1);
}

TEST_F(IntermediateStateReadTaskTest, BorrowedMaskedTableSlicesMaskAndDataInLockstep)
{
  bool_column_wrapper mask_owner({true, false, true, true}, {true, true, false, true});
  int64_column_wrapper data_owner{0, 10, 20, 30, 40, 50};
  auto const data = cudf::slice(data_owner, {1, 5}).front();
  auto input      = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask_view{mask_owner})
                 .add_state(gqe::qep::state_kind::cudf_column_view{data})
                 .build();
  gqe::qep::intermediate_state_read_task task{3};
  auto iterator = task.initialize(input, ctx_ref);
  auto chunks   = drain(task, iterator);

  ASSERT_EQ(chunks.size(), 2);
  for (auto const& chunk : chunks) {
    EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::valid_mask_view>(*chunk.front()));
    EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*chunk.back()));
    ASSERT_TRUE(gqe::qep::masked_table_view::try_from(chunk).has_value());
  }
  auto first  = *gqe::qep::masked_table_view::try_from(chunks[0]);
  auto second = *gqe::qep::masked_table_view::try_from(chunks[1]);
  bool_column_wrapper expected_mask0{{true, false, false}, {true, true, false}};
  bool_column_wrapper expected_mask1({true}, {true});
  int64_column_wrapper expected_data0{10, 20, 30};
  int64_column_wrapper expected_data1{40};
  EXPECT_EQ(first.mask.null_count(), 1);
  EXPECT_EQ(second.mask.null_count(), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first.mask, expected_mask0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second.mask, expected_mask1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first.columns.column(0), expected_data0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second.columns.column(0), expected_data1);
}

TEST_F(IntermediateStateReadTaskTest, CountOnlyAlwaysEmitsSingleOutput)
{
  auto bounded_input = gqe::qep::make_row_count_container(7);
  gqe::qep::intermediate_state_read_task bounded{1};
  auto bounded_iterator = bounded.initialize(bounded_input, ctx_ref);
  auto chunks           = drain(bounded, bounded_iterator);
  ASSERT_EQ(chunks.size(), 1);
  EXPECT_EQ(*gqe::qep::try_row_count(chunks.front()), 7);

  auto single_input = gqe::qep::make_row_count_container(7);
  gqe::qep::intermediate_state_read_task single{
    gqe::qep::intermediate_state_read_task::single_output};
  auto single_iterator = single.initialize(single_input, ctx_ref);
  auto single_chunks   = drain(single, single_iterator);
  ASSERT_EQ(single_chunks.size(), 1);
  EXPECT_EQ(*gqe::qep::try_row_count(single_chunks.front()), 7);
}

TEST_F(IntermediateStateReadTaskTest, ZeroRowInputsEmitOneShapeBearingOutput)
{
  auto plain =
    gqe::qep::state_container_builder()
      .add_state(one_column_table(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING})),
                 gqe::query_stream::default_stream())
      .build();
  auto masked =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{cudf::make_empty_column(
                   cudf::data_type{cudf::type_id::BOOL8})},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64})),
                 gqe::query_stream::default_stream())
      .build();
  auto count = gqe::qep::make_row_count_container(0);
  gqe::qep::intermediate_state_read_task task{3};

  auto plain_iterator = task.initialize(plain, ctx_ref);
  auto plain_chunks   = drain(task, plain_iterator);
  ASSERT_EQ(plain_chunks.size(), 1);
  EXPECT_EQ(gqe::qep::to_table_view(plain_chunks.front())->num_rows(), 0);
  EXPECT_TRUE(
    std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*plain_chunks.front().front()));

  auto masked_iterator = task.initialize(masked, ctx_ref);
  auto masked_chunks   = drain(task, masked_iterator);
  ASSERT_EQ(masked_chunks.size(), 1);
  EXPECT_TRUE(gqe::qep::masked_table_view::try_from(masked_chunks.front()).has_value());

  auto count_iterator = task.initialize(count, ctx_ref);
  auto count_chunks   = drain(task, count_iterator);
  ASSERT_EQ(count_chunks.size(), 1);
  EXPECT_EQ(*gqe::qep::try_row_count(count_chunks.front()), 0);
}

TEST_F(IntermediateStateReadTaskTest, SingleIndirectOutputPreservesCompleteShapeWithoutCopies)
{
  auto source =
    gather_source({owned_int64({10, 20, 30}), owned_int64({100, 200, 300})}, positions({2, 0}));
  gqe::qep::state_container input{indirect(source, 1), indirect(source, 0)};
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  gqe::qep::intermediate_state_read_task task{
    gqe::qep::intermediate_state_read_task::single_output};

  auto iterator = task.initialize(input, ctx_ref, gqe::query_stream::default_stream(), result_mr);
  auto output   = task.next(iterator, ctx_ref);
  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(task.next(iterator, ctx_ref).has_value());
  ASSERT_EQ(output->size(), input.size());
  EXPECT_EQ((*output)[0], input[0]);
  EXPECT_EQ((*output)[1], input[1]);
  EXPECT_TRUE(result_mr.get_outstanding_allocations().empty());

  auto view = gqe::qep::indirect_table_view::try_from(*output);
  ASSERT_TRUE(view.has_value());
  ASSERT_EQ(view->indirect_columns.size(), 2);
  EXPECT_EQ(view->indirect_columns[0].gather_source, source);
  EXPECT_EQ(view->indirect_columns[1].gather_source, source);

  auto gathered = gqe::qep::gather_task{}.next(*output, ctx_ref);
  ASSERT_TRUE(gathered.has_value());
  auto table = gqe::qep::to_table_view(*gathered);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper{300, 100});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), int64_column_wrapper{30, 10});
}

TEST_F(IntermediateStateReadTaskTest,
       BoundedIndirectOutputSlicesSharedSourcesDirectColumnsAndMaskInLockstep)
{
  auto first_source =
    gather_source({owned_int64({10, 20, 30, 40, 50}), owned_int64({100, 200, 300, 400, 500})},
                  positions({4, 1, 3, 0, 2}),
                  cudf::out_of_bounds_policy::NULLIFY);
  auto second_source = gather_source({owned_int64({7, 8, 9, 10})}, positions({3, 0, 2, 1, 3}));
  int64_column_wrapper direct_owner({999, 1000, 1001, 1002, 1003, 1004, 1005},
                                    {true, true, false, true, true, false, true});
  auto const direct = cudf::slice(direct_owner, {1, 6}).front();
  auto input        = gqe::qep::state_container_builder()
                 .add_state(
                   gqe::qep::state_kind::valid_mask{
                     bool_column_wrapper{true, false, true, true, false}.release()},
                   gqe::query_stream::default_stream())
                 .add_state(indirect(first_source, 1))
                 .add_state(gqe::qep::state_kind::cudf_column_view{direct})
                 .add_state(indirect(first_source, 0))
                 .add_state(indirect(second_source, 0))
                 .build();
  auto const input_mask = *gqe::qep::to_valid_mask_view(input.front());
  gqe::qep::intermediate_state_read_task task{2};
  auto iterator = task.initialize(input, ctx_ref);
  auto chunks   = drain(task, iterator);

  ASSERT_EQ(chunks.size(), 3);
  std::array<std::vector<int32_t>, 3> const expected_first_positions{
    std::vector<int32_t>{4, 1}, std::vector<int32_t>{3, 0}, std::vector<int32_t>{2}};
  std::array<std::vector<int32_t>, 3> const expected_second_positions{
    std::vector<int32_t>{3, 0}, std::vector<int32_t>{2, 1}, std::vector<int32_t>{3}};
  std::array<std::vector<bool>, 3> const expected_masks{
    std::vector<bool>{true, false}, std::vector<bool>{true, true}, std::vector<bool>{false}};
  std::array<std::vector<int64_t>, 3> const expected_direct{
    std::vector<int64_t>{1000, 1001}, std::vector<int64_t>{1002, 1003}, std::vector<int64_t>{1004}};
  std::array<std::vector<bool>, 3> const expected_direct_validity{
    std::vector<bool>{true, false}, std::vector<bool>{true, true}, std::vector<bool>{false}};

  for (std::size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    auto view = gqe::qep::indirect_table_view::try_from(chunks[chunk_index]);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->row_count, chunk_index == 2 ? 1 : 2);
    ASSERT_TRUE(view->mask.has_value());
    ASSERT_EQ(view->columns.size(), 4);
    ASSERT_EQ(view->indirect_columns.size(), 3);
    EXPECT_EQ(view->indirect_columns[0].output_index, 0);
    EXPECT_EQ(view->indirect_columns[1].output_index, 2);
    EXPECT_EQ(view->indirect_columns[2].output_index, 3);

    auto const shared_chunk_source = view->indirect_columns[0].gather_source;
    EXPECT_EQ(view->indirect_columns[1].gather_source, shared_chunk_source);
    EXPECT_NE(view->indirect_columns[2].gather_source, shared_chunk_source);
    EXPECT_EQ(shared_chunk_source->relation_source(), first_source->relation_source());
    EXPECT_EQ(shared_chunk_source->bounds_policy(), cudf::out_of_bounds_policy::NULLIFY);
    EXPECT_EQ(view->indirect_columns[2].gather_source->relation_source(),
              second_source->relation_source());

    auto const begin                = static_cast<cudf::size_type>(chunk_index * 2);
    auto const end                  = std::min<cudf::size_type>(begin + 2, 5);
    auto const expected_mask_view   = cudf::slice(input_mask, {begin, end}).front();
    auto const expected_direct_view = cudf::slice(direct, {begin, end}).front();
    auto const* mask_state =
      std::get_if<gqe::qep::state_kind::valid_mask_view>(chunks[chunk_index].front().get());
    auto const* direct_state =
      std::get_if<gqe::qep::state_kind::cudf_column_view>(view->columns[1].get());
    ASSERT_NE(mask_state, nullptr);
    ASSERT_NE(direct_state, nullptr);
    EXPECT_EQ(mask_state->column().head(), expected_mask_view.head());
    EXPECT_EQ(mask_state->column().offset(), expected_mask_view.offset());
    EXPECT_EQ(mask_state->column().null_mask(), expected_mask_view.null_mask());
    EXPECT_EQ(mask_state->column().null_count(), expected_mask_view.null_count());
    EXPECT_EQ(direct_state->column.head(), expected_direct_view.head());
    EXPECT_EQ(direct_state->column.offset(), expected_direct_view.offset());
    EXPECT_EQ(direct_state->column.null_mask(), expected_direct_view.null_mask());
    EXPECT_EQ(direct_state->column.null_count(), expected_direct_view.null_count());

    int32_column_wrapper expected_first(expected_first_positions[chunk_index].begin(),
                                        expected_first_positions[chunk_index].end());
    int32_column_wrapper expected_second(expected_second_positions[chunk_index].begin(),
                                         expected_second_positions[chunk_index].end());
    bool_column_wrapper expected_mask(expected_masks[chunk_index].begin(),
                                      expected_masks[chunk_index].end());
    int64_column_wrapper expected_direct_column(expected_direct[chunk_index].begin(),
                                                expected_direct[chunk_index].end(),
                                                expected_direct_validity[chunk_index].begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(shared_chunk_source->position_list()->column().view(),
                                   expected_first);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      view->indirect_columns[2].gather_source->position_list()->column().view(), expected_second);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*view->mask, expected_mask);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_column_view(view->columns[1]).value(),
                                   expected_direct_column);
  }

  auto first = gqe::qep::gather_task{}.next(chunks[0], ctx_ref);
  ASSERT_TRUE(first.has_value());
  auto first_table = gqe::qep::to_table_view(*first);
  ASSERT_TRUE(first_table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_table->column(0), int64_column_wrapper({500}, {true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_table->column(1), int64_column_wrapper({1000}, {true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_table->column(2), int64_column_wrapper({50}, {true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_table->column(3), int64_column_wrapper{10});

  auto second = gqe::qep::gather_task{}.next(chunks[1], ctx_ref);
  ASSERT_TRUE(second.has_value());
  auto second_table = gqe::qep::to_table_view(*second);
  ASSERT_TRUE(second_table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second_table->column(0),
                                 int64_column_wrapper({400, 100}, {true, true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second_table->column(1),
                                 int64_column_wrapper({1002, 1003}, {true, true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second_table->column(2),
                                 int64_column_wrapper({40, 10}, {true, true}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(second_table->column(3), int64_column_wrapper{9, 8});

  auto third = gqe::qep::gather_task{}.next(chunks[2], ctx_ref);
  ASSERT_TRUE(third.has_value());
  auto third_table = gqe::qep::to_table_view(*third);
  ASSERT_TRUE(third_table.has_value());
  EXPECT_EQ(third_table->num_columns(), 4);
  EXPECT_EQ(third_table->num_rows(), 0);
}

TEST_F(IntermediateStateReadTaskTest, ZeroRowIndirectInputEmitsOneShapeBearingOutput)
{
  auto source = gather_source({owned_int64({10, 20})}, positions({}));
  auto input  = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask{cudf::make_empty_column(
                              cudf::data_type{cudf::type_id::BOOL8})},
                            gqe::query_stream::default_stream())
                 .add_state(indirect(source))
                 .add_state(gqe::qep::state_kind::cudf_column{cudf::make_empty_column(
                              cudf::data_type{cudf::type_id::INT64})},
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::intermediate_state_read_task task{2};
  auto iterator = task.initialize(input, ctx_ref);
  auto chunks   = drain(task, iterator);

  ASSERT_EQ(chunks.size(), 1);
  auto view = gqe::qep::indirect_table_view::try_from(chunks.front());
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 0);
  ASSERT_TRUE(view->mask.has_value());
  EXPECT_EQ(view->columns.size(), 2);
  EXPECT_EQ(view->indirect_columns.size(), 1);
  EXPECT_EQ(view->indirect_columns.front().gather_source, source);
}

TEST_F(IntermediateStateReadTaskTest, BoundedIndirectOutputOwnsLifetimeAfterFinalization)
{
  auto source_state        = owned_int64({10, 20, 30});
  auto map                 = positions({2, 0, 1});
  auto source              = gather_source({source_state}, map);
  bool is_direct_destroyed = false;
  bool is_mask_destroyed   = false;
  auto direct_state = gqe::test::track_lifetime(owned_int64({100, 200, 300}), is_direct_destroyed);
  auto mask_state   = gqe::test::track_lifetime(
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
      gqe::query_stream::default_stream()),
    is_mask_destroyed);
  std::weak_ptr<gqe::qep::indirect_gather_source_state const> source_weak{source};
  std::weak_ptr<gqe::qep::position_list_state const> map_weak{map};
  gqe::qep::state_container input{mask_state, indirect(source), direct_state};
  source.reset();
  map.reset();
  source_state.reset();
  direct_state.reset();
  mask_state.reset();

  gqe::qep::intermediate_state_read_task task{2};
  auto iterator = task.initialize(input, ctx_ref);
  input.clear();
  auto output = *task.next(iterator, ctx_ref);
  task.finalize(std::move(iterator), ctx_ref);

  EXPECT_FALSE(source_weak.expired());
  EXPECT_FALSE(map_weak.expired());
  EXPECT_FALSE(is_direct_destroyed);
  EXPECT_FALSE(is_mask_destroyed);
  auto const output_mask = gqe::qep::to_valid_mask_view(output.front());
  ASSERT_TRUE(output_mask.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output_mask, bool_column_wrapper{true, false});
  auto gathered = gqe::qep::gather_task{}.next(output, ctx_ref);
  ASSERT_TRUE(gathered.has_value());
  auto table = gqe::qep::to_table_view(*gathered);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper{30});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), int64_column_wrapper{100});

  gathered.reset();
  output.clear();
  EXPECT_TRUE(source_weak.expired());
  EXPECT_TRUE(map_weak.expired());
  EXPECT_TRUE(is_direct_destroyed);
  EXPECT_TRUE(is_mask_destroyed);
}

TEST_F(IntermediateStateReadTaskTest, ConcurrentCallsClaimEveryIndirectChunkOnce)
{
  constexpr int num_threads          = 8;
  constexpr cudf::size_type num_rows = 64;
  std::vector<int64_t> values(num_rows);
  std::vector<int32_t> map(num_rows);
  std::iota(values.begin(), values.end(), int64_t{0});
  std::iota(map.rbegin(), map.rend(), int32_t{0});
  auto source_state = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values.begin(), values.end()).release()},
    gqe::query_stream::default_stream());
  auto source = gather_source(
    {source_state},
    gqe::qep::make_position_list(int32_column_wrapper(map.begin(), map.end()).release(),
                                 gqe::query_stream::default_stream()));
  gqe::qep::state_container input{indirect(source)};
  gqe::qep::intermediate_state_read_task task{1};
  auto iterator = task.initialize(input, ctx_ref);
  std::barrier start{num_threads};
  std::mutex mutex;
  std::vector<gqe::qep::state_container> chunks;

  auto worker = [&] {
    start.arrive_and_wait();
    while (auto chunk = task.next(iterator, ctx_ref)) {
      std::lock_guard guard{mutex};
      chunks.push_back(std::move(*chunk));
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_EQ(chunks.size(), static_cast<std::size_t>(num_rows));
  std::vector<gqe::qep::state_container> materialized;
  std::vector<cudf::table_view> views;
  materialized.reserve(chunks.size());
  views.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto gathered = gqe::qep::gather_task{}.next(chunk, ctx_ref);
    ASSERT_TRUE(gathered.has_value());
    materialized.push_back(std::move(*gathered));
    views.push_back(*gqe::qep::to_table_view(materialized.back()));
  }
  auto combined = cudf::concatenate(views);
  auto sorted   = cudf::sort(combined->view());
  int64_column_wrapper expected(values.begin(), values.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TEST_F(IntermediateStateReadTaskTest, IndirectPositionSlicesUseSuppliedStreamAndResultResource)
{
  auto source =
    gather_source({owned_int64({10, 20, 30, 40, 50}), owned_int64({100, 200, 300, 400, 500})},
                  positions({4, 3, 2, 1, 0}));
  gqe::qep::state_container input{indirect(source, 0), indirect(source, 1)};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  gqe::qep::intermediate_state_read_task task{2};

  auto iterator = task.initialize(input, ctx_ref, stream_handle, result_mr);
  auto chunks   = drain(task, iterator, stream_handle);
  stream_handle.ref().sync();

  ASSERT_EQ(chunks.size(), 3);
  std::set<void*> position_allocations;
  for (auto const& chunk : chunks) {
    auto view = gqe::qep::indirect_table_view::try_from(chunk);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(view->indirect_columns.size(), 2);
    auto const first_source = view->indirect_columns[0].gather_source;
    EXPECT_EQ(view->indirect_columns[1].gather_source, first_source);
    auto* head = const_cast<void*>(first_source->position_list()->column().view().head());
    EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(head));
    position_allocations.insert(head);
  }
  EXPECT_EQ(position_allocations.size(), chunks.size());
}

TEST_F(IntermediateStateReadTaskTest, ConcurrentCallsClaimEveryRangeOnce)
{
  constexpr int num_threads          = 8;
  constexpr cudf::size_type num_rows = 64;
  std::vector<int64_t> values(num_rows);
  std::iota(values.begin(), values.end(), int64_t{0});
  auto input =
    gqe::qep::state_container_builder()
      .add_state(one_column_table(int64_column_wrapper(values.begin(), values.end()).release()),
                 gqe::query_stream::default_stream())
      .build();
  gqe::qep::intermediate_state_read_task task{1};
  auto iterator = task.initialize(input, ctx_ref);
  std::barrier start{num_threads};
  std::mutex mutex;
  std::vector<std::pair<cudf::size_type, cudf::size_type>> ranges;

  auto worker = [&] {
    start.arrive_and_wait();
    while (auto chunk = task.next(iterator, ctx_ref)) {
      auto const column = gqe::qep::to_table_view(*chunk)->column(0);
      std::lock_guard lock{mutex};
      ranges.emplace_back(column.offset(), column.size());
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::ranges::sort(ranges);
  ASSERT_EQ(ranges.size(), static_cast<std::size_t>(num_rows));
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    EXPECT_EQ(ranges[static_cast<std::size_t>(i)],
              (std::pair<cudf::size_type, cudf::size_type>{i, 1}));
  }
}

TEST_F(IntermediateStateReadTaskTest, ConcurrentZeroRowCallsEmitOnce)
{
  constexpr int num_threads = 16;
  auto input =
    gqe::qep::state_container_builder()
      .add_state(one_column_table(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64})),
                 gqe::query_stream::default_stream())
      .build();
  gqe::qep::intermediate_state_read_task task{3};
  auto iterator = task.initialize(input, ctx_ref);
  std::barrier start{num_threads};
  std::atomic<int> output_count{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      if (task.next(iterator, ctx_ref)) { output_count.fetch_add(1); }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(output_count.load(), 1);
}

TEST_F(IntermediateStateReadTaskTest, ConcurrentCallsAtEndOfStreamReturnNoOutput)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::intermediate_state_read_task task{
    gqe::qep::intermediate_state_read_task::single_output};
  auto iterator = task.initialize(input, ctx_ref);
  ASSERT_TRUE(task.next(iterator, ctx_ref).has_value());

  constexpr int num_threads = 16;
  std::barrier start{num_threads};
  std::atomic<int> unexpected_outputs{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      if (task.next(iterator, ctx_ref)) { unexpected_outputs.fetch_add(1); }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(unexpected_outputs.load(), 0);
}

TEST_F(IntermediateStateReadTaskTest, TaskAndCloneInitializeIndependentIterators)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3, 4}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::intermediate_state_read_task task{2};
  auto clone_base = task.clone();
  auto* clone     = dynamic_cast<gqe::qep::intermediate_state_read_task*>(clone_base.get());
  ASSERT_NE(clone, nullptr);
  auto first  = task.initialize(input, ctx_ref);
  auto second = clone->initialize(input, ctx_ref);
  EXPECT_EQ(gqe::qep::get_row_count(*task.next(first, ctx_ref)), 2);
  EXPECT_EQ(gqe::qep::get_row_count(*clone->next(second, ctx_ref)), 2);
  EXPECT_EQ(gqe::qep::get_row_count(*task.next(first, ctx_ref)), 2);
  EXPECT_EQ(gqe::qep::get_row_count(*clone->next(second, ctx_ref)), 2);
}

TEST_F(IntermediateStateReadTaskTest, RejectsInvalidConfigurationAndInputShapes)
{
  EXPECT_THROW(gqe::qep::intermediate_state_read_task{0}, std::invalid_argument);
  EXPECT_THROW(gqe::qep::intermediate_state_read_task{-1}, std::invalid_argument);
  gqe::qep::intermediate_state_read_task task{3};

  gqe::qep::state_container empty;
  EXPECT_THROW(std::ignore = task.initialize(empty, ctx_ref), std::logic_error);

  auto mixed = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::row_count{1})
                 .add_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{1}.release()},
                            gqe::query_stream::default_stream())
                 .build();
  EXPECT_THROW(std::ignore = task.initialize(mixed, ctx_ref), std::logic_error);

  auto private_input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::task_private{std::make_unique<unrelated_private_state>()})
      .build();
  EXPECT_THROW(std::ignore = task.initialize(private_input, ctx_ref), std::logic_error);
}

TEST_F(IntermediateStateReadTaskTest, OutputAnchorsOwnedInputAfterFinalization)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  bool is_source_destroyed = false;
  input.front() = gqe::test::track_lifetime(std::move(input.front()), is_source_destroyed);
  gqe::qep::intermediate_state_read_task task{2};
  auto iterator = task.initialize(input, ctx_ref);
  input.clear();
  auto output = *task.next(iterator, ctx_ref);
  ASSERT_TRUE(task.next(iterator, ctx_ref).has_value());
  task.finalize(std::move(iterator), ctx_ref);
  EXPECT_FALSE(is_source_destroyed);
  int64_column_wrapper expected{1, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(output)->column(0), expected);
  output.clear();
  EXPECT_TRUE(is_source_destroyed);
}

TEST_F(IntermediateStateReadTaskTest, MaskAndDataOutputsAnchorOwnedInputs)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{10, 20, 30}.release()),
                 gqe::query_stream::default_stream())
      .build();
  bool is_mask_destroyed = false;
  bool is_data_destroyed = false;
  input[0]               = gqe::test::track_lifetime(std::move(input[0]), is_mask_destroyed);
  input[1]               = gqe::test::track_lifetime(std::move(input[1]), is_data_destroyed);
  gqe::qep::intermediate_state_read_task task{
    gqe::qep::intermediate_state_read_task::single_output};
  auto iterator = task.initialize(input, ctx_ref);
  input.clear();
  auto output = *task.next(iterator, ctx_ref);
  task.finalize(std::move(iterator), ctx_ref);
  EXPECT_FALSE(is_mask_destroyed);
  EXPECT_FALSE(is_data_destroyed);
  EXPECT_TRUE(gqe::qep::masked_table_view::try_from(output).has_value());
  output.clear();
  EXPECT_TRUE(is_mask_destroyed);
  EXPECT_TRUE(is_data_destroyed);
}
