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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

std::unique_ptr<gqe::expression> greater_than(int64_t threshold)
{
  return std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::literal_expression<int64_t>>(threshold));
}

cudf::table one_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

/** @brief Build two indirect logical columns sharing one source and position list. */
gqe::qep::state_container indirect_filter_input(bool is_masked)
{
  auto source_columns =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()},
                 gqe::query_stream::default_stream())
      .add_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{10, 20, 30}.release()},
                 gqe::query_stream::default_stream())
      .build();
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(source_columns));
  auto source   = gqe::qep::indirect_gather_source_state::make(
    std::move(relation),
    gqe::qep::make_position_list(int32_column_wrapper{2, 0, 1}.release(),
                                 gqe::query_stream::default_stream()));
  auto predicate = gqe::qep::indirect_column_state::make(source, 0);
  auto payload   = gqe::qep::indirect_column_state::make(std::move(source), 1);
  if (!is_masked) { return {std::move(predicate), std::move(payload)}; }
  return gqe::qep::state_container_builder()
    .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, true, false}.release()},
               gqe::query_stream::default_stream())
    .add_state(std::move(predicate))
    .add_state(std::move(payload))
    .build();
}

class QepFilterTaskTest : public gqe::test::BaseFixture {
 protected:
  QepFilterTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::qep::masked_table_view run(
    gqe::qep::filter_task const& task,
    gqe::qep::state_container_view input,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
  {
    result = task.next(input, ctx_ref, stream_handle, mr);
    EXPECT_TRUE(result.has_value());
    auto masked = gqe::qep::masked_table_view::try_from(*result);
    EXPECT_TRUE(masked.has_value());
    return *masked;
  }

  gqe::context_reference ctx_ref;
  std::optional<gqe::qep::state_container> result;
};

TEST_F(QepFilterTaskTest, ComputedPredicateProducesMaskAndPassesColumnsThrough)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{1, 2, 3, 4}.release());
  columns.push_back(int64_column_wrapper{10, 20, 30, 40}.release());
  auto input = gqe::qep::state_container_builder()
                 .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
                 .build();
  auto const* first_slot  = input[0].get();
  auto const* second_slot = input[1].get();

  gqe::qep::filter_task task{greater_than(2)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_EQ((*result)[1].get(), first_slot);
  EXPECT_EQ((*result)[2].get(), second_slot);
  EXPECT_EQ(masked.columns.num_columns(), 2);
}

TEST_F(QepFilterTaskTest, BareBooleanReferenceProducesOwnedIndependentMask)
{
  auto bools            = bool_column_wrapper{true, false, true}.release();
  auto const input_head = bools->view().head();
  auto input            = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::cudf_column{std::move(bools)},
                            gqe::query_stream::default_stream())
                 .build();

  gqe::qep::filter_task task{std::make_unique<gqe::column_reference_expression>(0)};
  auto masked = run(task, input);

  bool_column_wrapper expected{true, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_NE(masked.mask.head(), input_head);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::valid_mask>(*(*result)[0]));
}

TEST_F(QepFilterTaskTest, PreservesMixedOwnedAndBorrowedStateKinds)
{
  auto owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()},
    gqe::query_stream::default_stream());
  int64_column_wrapper borrowed_owner{4, 5, 6};
  auto borrowed =
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{borrowed_owner});
  gqe::qep::state_container input{owned, borrowed};

  gqe::qep::filter_task task{greater_than(0)};
  run(task, input);

  EXPECT_EQ((*result)[1], owned);
  EXPECT_EQ((*result)[2], borrowed);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*(*result)[1]));
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column_view>(*(*result)[2]));
}

TEST_F(QepFilterTaskTest, SelectivelyEvaluatesIndirectPredicateAndPreservesPayloadIdentity)
{
  for (bool const is_masked : {false, true}) {
    auto input               = indirect_filter_input(is_masked);
    auto const column_offset = is_masked ? std::size_t{1} : std::size_t{0};
    auto const predicate     = input[column_offset];
    auto const payload       = input[column_offset + 1];
    gqe::qep::filter_task task{greater_than(1)};

    auto output = task.next(input, ctx_ref);
    ASSERT_TRUE(output.has_value());
    auto logical = gqe::qep::logical_table_view::try_from(*output);
    ASSERT_TRUE(logical.has_value());
    ASSERT_TRUE(logical->valid_mask().has_value());
    EXPECT_EQ(logical->column_state(0), predicate);
    EXPECT_EQ(logical->column_state(1), payload);
    EXPECT_NE(gqe::qep::try_get_indirect_column(logical->column_state(1)), nullptr);
    if (is_masked) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*logical->valid_mask(),
                                          bool_column_wrapper{true, false, false});
    } else {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*logical->valid_mask(),
                                          bool_column_wrapper{true, false, true});
    }
  }
}

TEST_F(QepFilterTaskTest, EmptyRowsRemainSchemaBearing)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}));
  auto input = gqe::qep::state_container_builder()
                 .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
                 .build();

  gqe::qep::filter_task task{greater_than(0)};
  auto masked = run(task, input);

  EXPECT_EQ(masked.row_count, 0);
  EXPECT_EQ(masked.columns.num_columns(), 1);
  EXPECT_EQ(result->size(), 2);
}

TEST_F(QepFilterTaskTest, AllFalsePredicatePreservesRowGeometry)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::filter_task task{greater_than(100)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, false, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
  EXPECT_EQ(masked.row_count, 3);
}

TEST_F(QepFilterTaskTest, NullPredicateRowsDoNotSurviveMaterialization)
{
  auto values = int64_column_wrapper{{1, 2, 3, 4}, {true, false, true, false}}.release();
  auto input =
    gqe::qep::state_container_builder()
      .add_state(one_column_table(std::move(values)), gqe::query_stream::default_stream())
      .build();
  gqe::qep::filter_task task{greater_than(1)};
  auto masked = run(task, input);

  auto materialized = cudf::apply_retention_mask(masked.columns, masked.mask);
  int64_column_wrapper expected{3};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(materialized->view().column(0), expected);
}

TEST_F(QepFilterTaskTest, SlicedBorrowedColumnKeepsAlignmentAndIdentity)
{
  int64_column_wrapper owner{0, 1, 2, 3, 4};
  auto const sliced = cudf::slice(owner, {1, 4}).front();
  auto slot         = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  gqe::qep::state_container input{slot};

  gqe::qep::filter_task task{greater_than(1)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
  EXPECT_EQ((*result)[1], slot);
  EXPECT_EQ(masked.columns.column(0).offset(), sliced.offset());
}

TEST_F(QepFilterTaskTest, ChainedMaskPreservesSelection)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(
                   gqe::qep::state_kind::valid_mask{
                     bool_column_wrapper{{false, true, true}, {true, true, false}}.release()},
                   gqe::query_stream::default_stream())
                 .add_state(one_column_table(int64_column_wrapper{-1, 2, 3}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::filter_task task{greater_than(0)};
  auto masked = run(task, input);

  auto selected = cudf::apply_retention_mask(masked.columns, masked.mask);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(selected->view().column(0), int64_column_wrapper{2});
  EXPECT_EQ((*result)[1], input[1]);
}

TEST_F(QepFilterTaskTest, MaskedOutNonNullRowCannotBeResurrectedByIsNull)
{
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{false, true, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(one_column_table(int64_column_wrapper{{-1, 2, 3}, {true, false, true}}.release()),
                 gqe::query_stream::default_stream())
      .build();
  auto condition = std::make_unique<gqe::is_null_expression>(
    std::make_shared<gqe::column_reference_expression>(0));
  gqe::qep::filter_task task{std::move(condition)};
  auto masked = run(task, input);

  bool_column_wrapper expected{false, true, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
}

TEST_F(QepFilterTaskTest, MaskedSliceRunsOnExplicitStreamAndResultResource)
{
  int64_column_wrapper owner{0, 1, 2, 3, 4};
  auto sliced = cudf::slice(owner, {1, 4}).front();
  auto input =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
                 gqe::query_stream::default_stream())
      .add_state(gqe::qep::state_kind::cudf_column_view{sliced})
      .build();
  gqe::qep::filter_task task{greater_than(1)};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto masked = run(task, input, stream_handle, result_mr);
  stream_handle.ref().sync();

  bool_column_wrapper expected{false, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(masked.mask, expected);
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(masked.mask.head())));
  result.reset();
}

TEST_F(QepFilterTaskTest, RejectsInvalidConstructionAndInputShapes)
{
  EXPECT_THROW(gqe::qep::filter_task{nullptr}, std::invalid_argument);
  auto constant_condition =
    std::make_unique<gqe::equal_expression>(std::make_shared<gqe::literal_expression<int64_t>>(1),
                                            std::make_shared<gqe::literal_expression<int64_t>>(1));
  EXPECT_THROW(gqe::qep::filter_task{std::move(constant_condition)}, std::invalid_argument);
  gqe::qep::filter_task task{greater_than(0)};
  gqe::qep::state_container empty;
  EXPECT_THROW((void)task.next(empty, ctx_ref), std::logic_error);
  auto count_only = gqe::qep::make_row_count_container(3);
  EXPECT_THROW((void)task.next(count_only, ctx_ref), std::logic_error);

  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  gqe::qep::filter_task non_boolean{std::make_unique<gqe::column_reference_expression>(0)};
  EXPECT_THROW((void)non_boolean.next(input, ctx_ref), std::logic_error);
}

TEST_F(QepFilterTaskTest, CloneIsIndependentAndEquivalent)
{
  auto input = gqe::qep::state_container_builder()
                 .add_state(one_column_table(int64_column_wrapper{1, 2, 3}.release()),
                            gqe::query_stream::default_stream())
                 .build();
  auto original = std::make_unique<gqe::qep::filter_task>(greater_than(1));
  auto clone    = original->clone();
  original.reset();
  auto* cloned_filter = dynamic_cast<gqe::qep::filter_task*>(clone.get());
  ASSERT_NE(cloned_filter, nullptr);

  auto masked = run(*cloned_filter, input);
  bool_column_wrapper expected{false, true, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, expected);
}

TEST_F(QepFilterTaskTest, ChainedNullableBooleanMasksSelectOnlyTrueTrue)
{
  for (auto const size : {0, 9, 32, 128, 1024, 1031}) {
    std::vector<bool> previous(size), predicate(size), previous_valid(size), predicate_valid(size);
    for (int i = 0; i < size; ++i) {
      previous[i]        = (i % 9) / 3 == 0;
      previous_valid[i]  = (i % 9) / 3 != 2;
      predicate[i]       = i % 3 == 0;
      predicate_valid[i] = i % 3 != 2;
    }
    bool_column_wrapper old_mask(previous.begin(), previous.end(), previous_valid.begin());
    bool_column_wrapper values(predicate.begin(), predicate.end(), predicate_valid.begin());
    auto input = gqe::qep::state_container_builder()
                   .add_state(gqe::qep::state_kind::valid_mask{old_mask.release()},
                              gqe::query_stream::default_stream())
                   .add_state(gqe::qep::state_kind::cudf_column{values.release()},
                              gqe::query_stream::default_stream())
                   .build();
    gqe::qep::filter_task task{std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(0))};
    auto masked   = run(task, input);
    auto selected = cudf::apply_retention_mask(masked.columns, masked.mask);
    EXPECT_EQ(selected->num_rows(), (size + 8) / 9);
    std::vector<bool> expected((size + 8) / 9, true);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(selected->view().column(0),
                                        bool_column_wrapper(expected.begin(), expected.end()));
    auto selected_again  = task.next(*result, ctx_ref);
    auto again           = gqe::qep::masked_table_view::try_from(*selected_again);
    auto compacted_again = cudf::apply_retention_mask(again->columns, again->mask);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(selected->view(), compacted_again->view());
  }
}

TEST_F(QepFilterTaskTest, SlicedNullableSelectionAndBorrowedPredicateRemainUnchanged)
{
  bool_column_wrapper old_owner{{true, true, true, false, true, true},
                                {true, true, false, true, true, true}};
  bool_column_wrapper predicate_owner{{false, true, true, true, true, false, true},
                                      {true, true, true, true, false, true, true}};
  auto old_mask  = cudf::slice(old_owner, {1, 5}).front();
  auto predicate = cudf::slice(predicate_owner, {2, 6}).front();
  auto input     = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::valid_mask_view{old_mask})
                 .add_state(gqe::qep::state_kind::cudf_column_view{predicate})
                 .build();
  gqe::qep::filter_task task{std::make_unique<gqe::column_reference_expression>(0)};
  auto masked = run(task, input);
  EXPECT_FALSE(masked.mask.nullable());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(masked.mask, bool_column_wrapper{true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(old_owner,
                                 (bool_column_wrapper{{true, true, true, false, true, true},
                                                      {true, true, false, true, true, true}}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    predicate_owner,
    (bool_column_wrapper{{false, true, true, true, true, false, true},
                         {true, true, true, true, false, true, true}}));
  EXPECT_EQ((*result)[1], input[1]);
}
