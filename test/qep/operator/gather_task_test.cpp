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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

cudf::table one_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

gqe::qep::indirect_gather_source gather_source(
  gqe::qep::state_container sources,
  gqe::qep::position_list_handle positions,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(sources));
  return gqe::qep::indirect_gather_source_state::make(
    std::move(relation), std::move(positions), policy);
}

gqe::qep::shared_state indirect(gqe::qep::indirect_gather_source source,
                                cudf::size_type source_column_index)
{
  return gqe::qep::indirect_column_state::make(std::move(source), source_column_index);
}

gqe::qep::shared_state indirect(
  gqe::qep::shared_state source,
  gqe::qep::position_list_handle positions,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  return indirect(gather_source({std::move(source)}, std::move(positions), policy),
                  cudf::size_type{0});
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

class private_state final : public gqe::qep::task_private_state {};

class QepGatherTaskTest : public gqe::test::BaseFixture {
 protected:
  QepGatherTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::qep::state_container run(
    gqe::qep::state_container_view input,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
  {
    auto result = task.next(input, ctx_ref, stream_handle, mr);
    EXPECT_TRUE(result.has_value());
    return std::move(*result);
  }

  gqe::qep::gather_task task;
  gqe::context_reference ctx_ref;
};

/**
 * @brief An empty input is an executor-contract violation, not an end-of-stream signal.
 *
 * Per the `qep::task` early-exit protocol, the executor never invokes `next` with early-exited
 * (empty) inputs, and tasks do not handle or propagate the exit themselves.
 */
TEST_F(QepGatherTaskTest, RejectsEmptyInput)
{
  EXPECT_THROW(std::ignore = task.next({}, ctx_ref), std::logic_error);
}

TEST_F(QepGatherTaskTest, GathersValidAndTrueRowsAndRemovesMask)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(
        gqe::qep::state_kind::valid_mask{
          bool_column_wrapper{{true, true, false, true}, {true, false, true, true}}.release()},
        gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{1, 2, 3, 4}.release()),
                 gqe::query_stream::default_stream())
      .build();

  auto result = run(input);
  auto table  = gqe::qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result[0]));
  int64_column_wrapper expected{1, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
}

TEST_F(QepGatherTaskTest, PreservesMultipleColumnOrderTypesAndDataNulls)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int32_column_wrapper{{1, 2, 3}, {true, true, false}}.release());
  columns.push_back(string_column_wrapper{{"one", "deux", "三"}, {true, false, true}}.release());
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, true, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
      .build();

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  ASSERT_EQ(table.num_columns(), 2);
  EXPECT_EQ(table.column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(table.column(1).type().id(), cudf::type_id::STRING);
  int32_column_wrapper expected_numbers{{1, 2, 3}, {true, true, false}};
  string_column_wrapper expected_strings{{"one", "deux", "三"}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_numbers);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_strings);
}

TEST_F(QepGatherTaskTest, AllFalseAndZeroRowInputsEmitSchemaCorrectColumns)
{
  auto all_false =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{false, false}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{1, 2}.release()),
                 gqe::query_stream::default_stream())
      .build();
  auto false_result = run(all_false);
  auto false_table  = *gqe::qep::to_table_view(false_result);
  EXPECT_EQ(false_table.num_rows(), 0);
  EXPECT_EQ(false_table.column(0).type().id(), cudf::type_id::INT64);

  auto zero_rows =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{cudf::make_empty_column(
                   cudf::data_type{cudf::type_id::BOOL8})},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING})),
                 gqe::query_stream::default_stream())
      .build();
  auto zero_result = run(zero_rows);
  auto zero_table  = *gqe::qep::to_table_view(zero_result);
  ASSERT_EQ(zero_result.size(), 1);
  EXPECT_EQ(zero_table.num_rows(), 0);
  EXPECT_EQ(zero_table.column(0).type().id(), cudf::type_id::STRING);
}

TEST_F(QepGatherTaskTest, SupportsMixedOwnedBorrowedAndSlicedColumns)
{
  int64_column_wrapper owner{0, 10, 20, 30, 40};
  auto sliced            = cudf::slice(owner, {1, 4}).front();
  auto owned             = int32_column_wrapper{1, 2, 3}.release();
  auto const* owned_head = owned->view().head();
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(gqe::qep::state_kind::cudf_column{std::move(owned)},
                 gqe::query_stream::default_stream())
      .add_state(gqe::qep::state_kind::cudf_column_view{sliced})
      .build();

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int32_column_wrapper expected_owned{1, 3};
  int64_column_wrapper expected_sliced{10, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_owned);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_sliced);
  EXPECT_NE(table.column(0).head(), owned_head);
  EXPECT_NE(table.column(1).head(), sliced.head());
}

TEST_F(QepGatherTaskTest, CloneExecutesAfterOriginalIsDestroyed)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{5, 6}.release()),
                 gqe::query_stream::default_stream())
      .build();
  auto original = std::make_unique<gqe::qep::gather_task>();
  auto clone    = original->clone();
  original.reset();
  auto* gather = dynamic_cast<gqe::qep::gather_task*>(clone.get());
  ASSERT_NE(gather, nullptr);
  auto result = gather->next(input, ctx_ref);
  ASSERT_TRUE(result.has_value());
  int64_column_wrapper expected{6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(*result)->column(0), expected);
}

TEST_F(QepGatherTaskTest, UsesExplicitStreamAndResultResource)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()),
                 gqe::query_stream::default_stream())
      .build();
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto result = run(input, stream_handle, result_mr);
  stream_handle.ref().sync();
  auto table = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected{1, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected);
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}

TEST_F(QepGatherTaskTest, GathersOrderedRepeatedAndEmptyIndirectPositions)
{
  auto source = owned_int64({10, 20, 30});
  auto input  = gqe::qep::state_container{indirect(source, positions({0, 2, 2, 1}))};
  auto result = run(input);
  int64_column_wrapper expected{10, 30, 30, 20};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(result)->column(0), expected);

  auto empty        = gqe::qep::state_container{indirect(source, positions({}))};
  auto empty_result = run(empty);
  auto empty_table  = *gqe::qep::to_table_view(empty_result);
  EXPECT_EQ(empty_table.num_rows(), 0);
  EXPECT_EQ(empty_table.column(0).type().id(), cudf::type_id::INT64);
}

TEST_F(QepGatherTaskTest, PreservesInterleavedLogicalOrderAcrossSharedMaps)
{
  auto left_map     = positions({2, 0});
  auto right_map    = positions({1, 0});
  auto left_source  = owned_int64({10, 11, 12});
  auto right_source = owned_int64({20, 21});
  auto left_gather  = gather_source({left_source}, left_map);
  auto right_gather = gather_source({right_source}, right_map);
  auto input        = gqe::qep::state_container{
    indirect(left_gather, 0), indirect(right_gather, 0), indirect(left_gather, 0)};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected_left{12, 10};
  int64_column_wrapper expected_right{21, 20};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_left);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_right);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), expected_left);
}

TEST_F(QepGatherTaskTest, SharedGatherSourceSupportsOrderedAndRepeatedSourceColumns)
{
  auto shared_gather =
    gather_source({owned_int64({10, 11, 12}), owned_int64({20, 21, 22})}, positions({2, 0}));
  auto input = gqe::qep::state_container{
    indirect(shared_gather, 1), indirect(shared_gather, 0), indirect(shared_gather, 1)};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected_first{22, 20};
  int64_column_wrapper expected_second{12, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_second);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), expected_first);
}

TEST_F(QepGatherTaskTest, DistinctRelationsSharingOneMaskedPositionListStaySeparate)
{
  auto shared_map    = positions({2, 0, 1});
  auto first_gather  = gather_source({owned_int64({10, 11, 12})}, shared_map);
  auto second_gather = gather_source({owned_int64({100, 101, 102, 103})}, shared_map);
  auto input         = gqe::qep::state_container{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
      gqe::query_stream::default_stream()),
    indirect(first_gather, 0),
    indirect(second_gather, 0)};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected_first{12, 11};
  int64_column_wrapper expected_second{102, 101};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_second);
}

TEST_F(QepGatherTaskTest, ShallowForwardsDirectColumnsInMixedUnmaskedInput)
{
  auto direct_state = owned_int64({100, 200});
  auto const* direct_head =
    std::get<gqe::qep::state_kind::cudf_column>(*direct_state).column->view().head();
  auto input =
    gqe::qep::state_container{direct_state, indirect(owned_int64({10, 20}), positions({1, 0}))};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  EXPECT_EQ(result.front(), direct_state);
  EXPECT_EQ(table.column(0).head(), direct_head);
  int64_column_wrapper expected_indirect{20, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_indirect);
}

TEST_F(QepGatherTaskTest, AppliesNullableMaskBeforeMixedIndirectMaterialization)
{
  auto input = gqe::qep::state_container{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{
        bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release()},
      gqe::query_stream::default_stream()),
    indirect(owned_int64({10, 20, 30}), positions({2, 1, 0, 2})),
    owned_int64({100, 200, 300, 400})};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected_indirect{30, 30};
  int64_column_wrapper expected_direct{100, 400};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_indirect);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_direct);
}

TEST_F(QepGatherTaskTest, SupportsSlicedNullableSourcesAndNullifyingBounds)
{
  int64_column_wrapper owner{{0, 10, 20, 30}, {true, true, false, true}};
  auto sliced = cudf::slice(owner, {1, 4}).front();
  auto source = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  auto input  = gqe::qep::state_container{
    indirect(source, positions({2, 1, 8}), cudf::out_of_bounds_policy::NULLIFY)};

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected{{30, 0, 0}, {true, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(0), expected);
}

TEST_F(QepGatherTaskTest, IndirectStateKeepsOwnersAliveUntilMaterialization)
{
  gqe::qep::state_container input;
  {
    auto source = owned_int64({10, 20, 30});
    auto map    = positions({2, 0});
    input.push_back(indirect(source, map));
  }

  auto result = run(input);
  auto table  = *gqe::qep::to_table_view(result);
  int64_column_wrapper expected{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected);
}

TEST_F(QepGatherTaskTest, IndirectFinalColumnsUseExplicitStreamAndResultResource)
{
  auto input = gqe::qep::state_container{indirect(owned_int64({10, 20, 30}), positions({2, 0}))};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto result = run(input, stream_handle, result_mr);
  stream_handle.ref().sync();
  auto table = *gqe::qep::to_table_view(result);
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(table.column(0).head())));
}

TEST_F(QepGatherTaskTest, PlainTableAndCountOnlyInputsAreIdempotent)
{
  auto plain = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  auto plain_result = run(plain);
  EXPECT_EQ(plain_result.front(), plain.front());

  auto count_only        = gqe::qep::make_row_count_container(7);
  auto count_only_result = run(count_only);
  ASSERT_EQ(count_only_result.size(), 1);
  EXPECT_EQ(count_only_result.front(), count_only.front());
  EXPECT_EQ(gqe::qep::try_row_count(count_only_result), 7);
}

TEST_F(QepGatherTaskTest, RejectsMalformedStateShapes)
{
  gqe::qep::state_container private_only{gqe::qep::make_shared_state(
    gqe::qep::state_kind::task_private{std::make_unique<private_state>()})};
  EXPECT_THROW((void)task.next(private_only, ctx_ref), std::logic_error);
  gqe::qep::state_container null_slot{nullptr};
  EXPECT_THROW((void)task.next(null_slot, ctx_ref), std::logic_error);

  gqe::qep::state_container lone_mask{gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
    gqe::query_stream::default_stream())};
  EXPECT_THROW((void)task.next(lone_mask, ctx_ref), std::logic_error);

  auto trailing =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{1}.release()),
                 gqe::query_stream::default_stream())
      .add_state(gqe::qep::state_kind::row_count{1})
      .build();
  EXPECT_THROW((void)task.next(trailing, ctx_ref), std::logic_error);

  gqe::qep::state_container raw_positions{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{positions({0})})};
  EXPECT_THROW((void)task.next(raw_positions, ctx_ref), std::logic_error);
}
