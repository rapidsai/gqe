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

#include <gqe/qep/shapes/indirect_table.hpp>

#include "../lifetime_probe.hpp"

#include <gqe/qep/state.hpp>
#include <gqe/utility/error.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

namespace {

gqe::qep::shared_state owned_int64(std::initializer_list<int64_t> values)
{
  return gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values).release()},
    gqe::query_stream::default_stream());
}

gqe::qep::shared_state owned_int32(std::span<int32_t const> values,
                                   gqe::query_stream const& stream_handle)
{
  auto const stream = stream_handle.ref();
  auto column       = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream);
  GQE_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                               values.data(),
                               values.size_bytes(),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  stream.sync();
  return gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{std::move(column)},
                                     stream_handle);
}

gqe::qep::position_list_handle positions(std::span<int32_t const> values,
                                         gqe::query_stream const& stream_handle)
{
  auto const stream  = stream_handle.ref();
  auto state         = owned_int32(values, stream_handle);
  auto const* column = std::get_if<gqe::qep::state_kind::cudf_column>(state.get());
  auto copy          = std::make_unique<cudf::column>(column->column->view(), stream);
  stream.sync();
  return gqe::qep::make_position_list(std::move(copy), stream_handle);
}

[[nodiscard]] bool contains_stream(std::span<gqe::query_stream const> streams,
                                   gqe::query_stream const& expected)
{
  return std::ranges::any_of(
    streams, [expected](auto const& stream) { return stream.ref().get() == expected.ref().get(); });
}

gqe::qep::position_list_handle positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

gqe::qep::indirect_gather_source one_column_gather(
  gqe::qep::shared_state source,
  gqe::qep::position_list_handle map,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make({std::move(source)});
  return gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(map), policy);
}

gqe::qep::shared_state indirect(gqe::qep::indirect_gather_source gather,
                                cudf::size_type source_column_index = 0)
{
  return gqe::qep::indirect_column_state::make(std::move(gather), source_column_index);
}

}  // namespace

TEST(QepPositionListState, ValidatesStorageTypeAndNullability)
{
  auto valid = positions({2, 0, 2});
  EXPECT_EQ(valid->column().size(), 3);

  auto wrapped        = gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{valid});
  auto const& generic = std::get<gqe::qep::state_kind::position_list>(*wrapped);
  EXPECT_EQ(generic.state(), valid);

  auto wrong_type = int64_column_wrapper{0, 1}.release();
  EXPECT_THROW(std::ignore = gqe::qep::make_position_list(std::move(wrong_type),
                                                          gqe::query_stream::default_stream()),
               std::invalid_argument);
  auto nullable = int32_column_wrapper{{0, 1}, {true, false}}.release();
  EXPECT_THROW(std::ignore = gqe::qep::make_position_list(std::move(nullable),
                                                          gqe::query_stream::default_stream()),
               std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::qep::make_position_list(nullptr, gqe::query_stream::default_stream()),
    std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{nullptr}),
    std::invalid_argument);
}

TEST(QepStateColumnView, ConvertsOwnedBorrowedAndSlicedColumns)
{
  auto owned      = owned_int64({1, 2, 3});
  auto owned_view = gqe::qep::to_column_view(owned);
  ASSERT_TRUE(owned_view.has_value());
  EXPECT_EQ(owned_view->head(),
            std::get<gqe::qep::state_kind::cudf_column>(*owned).column->view().head());

  int64_column_wrapper owner{0, 10, 20, 30};
  auto sliced        = cudf::slice(owner, {1, 4}).front();
  auto borrowed      = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  auto borrowed_view = gqe::qep::to_column_view(borrowed);
  ASSERT_TRUE(borrowed_view.has_value());
  EXPECT_EQ(borrowed_view->head(), sliced.head());
  EXPECT_EQ(borrowed_view->offset(), sliced.offset());
}

TEST(QepStateColumnView, RejectsWrongKindAndThrowsForMalformedStorage)
{
  auto wrong_kind = gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1});
  auto null_owned = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{nullptr},
                                                gqe::query_stream::default_stream());

  EXPECT_FALSE(gqe::qep::to_column_view(wrong_kind).has_value());
  EXPECT_THROW(std::ignore = gqe::qep::to_column_view(nullptr), std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::to_column_view(null_owned), std::logic_error);
}

TEST(QepStateValidMaskView, ConvertsOwnedAndBorrowedMasksAndRejectsOtherKinds)
{
  auto owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
    gqe::query_stream::default_stream());
  auto const owned_view = gqe::qep::to_valid_mask_view(owned);
  ASSERT_TRUE(owned_view.has_value());
  EXPECT_EQ(owned_view->head(),
            std::get<gqe::qep::state_kind::valid_mask>(*owned).column().view().head());

  bool_column_wrapper owner({true, false, true, true}, {true, true, false, true});
  auto const sliced = cudf::slice(owner, {1, 4}).front();
  auto borrowed     = gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask_view{sliced});
  auto const borrowed_view = gqe::qep::to_valid_mask_view(borrowed);
  ASSERT_TRUE(borrowed_view.has_value());
  EXPECT_EQ(borrowed_view->head(), sliced.head());
  EXPECT_EQ(borrowed_view->null_mask(), sliced.null_mask());
  EXPECT_EQ(borrowed_view->offset(), sliced.offset());
  EXPECT_EQ(borrowed_view->null_count(), sliced.null_count());

  EXPECT_FALSE(gqe::qep::to_valid_mask_view(owned_int64({1, 2})).has_value());
  EXPECT_THROW(std::ignore = gqe::qep::to_valid_mask_view(nullptr), std::logic_error);
}

TEST(QepStateTableView, ThrowsForMalformedStorageAndRejectsWrongKinds)
{
  gqe::qep::state_container null_slot{nullptr};
  gqe::qep::state_container null_owned{gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{nullptr}, gqe::query_stream::default_stream())};
  gqe::qep::state_container wrong_kind{gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
    gqe::query_stream::default_stream())};

  EXPECT_THROW(std::ignore = gqe::qep::to_table_view(null_slot), std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::to_table_view(null_owned), std::logic_error);
  EXPECT_FALSE(gqe::qep::to_table_view(wrong_kind).has_value());
}

TEST(QepIndirectTableSource, AcceptsOwnedBorrowedSlicedNullableAndZeroRowColumns)
{
  auto owned = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{
                                             {1, 2, 3}, {true, false, true}}.release()},
                                           gqe::query_stream::default_stream());
  int32_column_wrapper owner{0, 10, 20, 30};
  auto sliced   = cudf::slice(owner, {1, 4}).front();
  auto borrowed = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});

  auto relation = gqe::qep::indirect_table_source_state::make({owned, borrowed});
  EXPECT_EQ(relation->row_count(), 3);
  EXPECT_EQ(relation->num_columns(), 2);
  auto view = relation->contiguous_table_view();
  EXPECT_EQ(view.column(0).null_count(), 1);
  EXPECT_EQ(view.column(1).head(), sliced.head());
  EXPECT_EQ(view.column(1).offset(), sliced.offset());

  auto empty_owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{
      cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64})},
    gqe::query_stream::default_stream());
  auto empty_borrowed_owner = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto empty_borrowed       = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column_view{empty_borrowed_owner->view()});
  auto empty_relation = gqe::qep::indirect_table_source_state::make({empty_owned, empty_borrowed});
  EXPECT_EQ(empty_relation->row_count(), 0);
  EXPECT_EQ(empty_relation->num_columns(), 2);
}

TEST(QepIndirectTableSource, RejectsEmptyNullNonConcreteAndMismatchedColumns)
{
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make({}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make({nullptr}),
               std::logic_error);
  EXPECT_THROW(
    std::ignore = gqe::qep::indirect_table_source_state::make({gqe::qep::make_shared_state(
      gqe::qep::state_kind::cudf_column{nullptr}, gqe::query_stream::default_stream())}),
    std::logic_error);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make(
                 {gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1})}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_source_state::make(
                 {owned_int64({1, 2}), owned_int64({3})}),
               std::invalid_argument);
}

TEST(QepIndirectGatherSource, RejectsNullHandles)
{
  auto relation = gqe::qep::indirect_table_source_state::make({owned_int64({1})});
  EXPECT_THROW(std::ignore = gqe::qep::indirect_gather_source_state::make(nullptr, positions({0})),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_gather_source_state::make(relation, nullptr),
               std::invalid_argument);
}

TEST(QepIndirectColumnState, ValidatesIndexAndSharesRelationGatherAndPositionLifetimes)
{
  bool is_first_source_destroyed  = false;
  bool is_second_source_destroyed = false;
  auto first_source =
    gqe::test::track_lifetime(owned_int64({10, 20, 30}), is_first_source_destroyed);
  auto second_source =
    gqe::test::track_lifetime(owned_int64({100, 200, 300}), is_second_source_destroyed);
  auto map = positions({2, 0});
  std::weak_ptr<gqe::qep::position_list_state const> map_weak{map};
  auto const* map_identity = map.get();

  auto relation = gqe::qep::indirect_table_source_state::make({first_source, second_source});
  auto gather   = gqe::qep::indirect_gather_source_state::make(relation, map);
  std::weak_ptr<gqe::qep::indirect_table_source_state const> relation_weak{relation};
  std::weak_ptr<gqe::qep::indirect_gather_source_state const> gather_weak{gather};

  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(nullptr, 0),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(gather, -1),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::indirect_column_state::make(gather, 2),
               std::invalid_argument);

  auto first  = indirect(gather, 0);
  auto second = indirect(gather, 1);
  first_source.reset();
  second_source.reset();
  map.reset();
  relation.reset();
  gather.reset();

  EXPECT_FALSE(is_first_source_destroyed);
  EXPECT_FALSE(is_second_source_destroyed);
  EXPECT_FALSE(map_weak.expired());
  EXPECT_FALSE(relation_weak.expired());
  EXPECT_FALSE(gather_weak.expired());
  {
    auto const& first_state  = *std::get<gqe::qep::state_kind::indirect_column>(*first).state();
    auto const& second_state = *std::get<gqe::qep::state_kind::indirect_column>(*second).state();
    EXPECT_EQ(first_state.gather_source(), second_state.gather_source());
    EXPECT_EQ(first_state.gather_source()->position_list(),
              second_state.gather_source()->position_list());
    EXPECT_EQ(first_state.source_column_index(), 0);
    EXPECT_EQ(second_state.source_column_index(), 1);
    EXPECT_EQ(first_state.gather_source()->relation_source()->num_columns(), 2);
    EXPECT_EQ(first_state.gather_source()->position_list().get(), map_identity);
  }
  first.reset();
  second.reset();
  EXPECT_TRUE(is_first_source_destroyed);
  EXPECT_TRUE(is_second_source_destroyed);
  EXPECT_TRUE(map_weak.expired());
  EXPECT_TRUE(relation_weak.expired());
  EXPECT_TRUE(gather_weak.expired());
}

TEST(QepIndirectColumnState, PublicWrapperOnlyExposesAnImmutableHandle)
{
  using wrapper = gqe::qep::state_kind::indirect_column;
  using handle  = std::remove_cvref_t<decltype(std::declval<wrapper const&>().state())>;
  static_assert(!std::is_aggregate_v<wrapper>);
  static_assert(std::is_const_v<typename handle::element_type>);
  EXPECT_THROW(std::ignore = wrapper{nullptr}, std::invalid_argument);
}

TEST(QepIndirectColumnState, BorrowedSlicedSourcePreservesViewWhileExternalOwnerLives)
{
  int64_column_wrapper owner{0, 10, 20, 30};
  auto sliced = cudf::slice(owner, {1, 4}).front();
  auto source = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{sliced});
  auto gather = one_column_gather(source, positions({2, 0}), cudf::out_of_bounds_policy::NULLIFY);
  source.reset();

  // `owner` must remain alive while the borrowed view is used. Destroying it before inspecting the
  // view would make the access undefined, so external ownership remains an API contract.
  auto view = gather->relation_source()->contiguous_table_view().column(0);
  EXPECT_EQ(view.head(), sliced.head());
  EXPECT_EQ(view.offset(), sliced.offset());
  EXPECT_EQ(gather->bounds_policy(), cudf::out_of_bounds_policy::NULLIFY);
}

TEST(QepIndirectTableView, RecognizesIndirectMixedAndMaskedShapesByIdentity)
{
  auto gather    = one_column_gather(owned_int64({10, 20, 30}), positions({1, 0}));
  auto projected = indirect(gather);

  gqe::qep::state_container indirect_only{projected};
  auto one = gqe::qep::indirect_table_view::try_from(indirect_only);
  ASSERT_TRUE(one.has_value());
  EXPECT_EQ(one->row_count, 2);
  EXPECT_FALSE(one->mask.has_value());
  ASSERT_EQ(one->indirect_columns.size(), 1);
  EXPECT_EQ(one->indirect_columns.front().output_index, 0);
  EXPECT_EQ(one->indirect_columns.front().gather_source, gather);
  EXPECT_EQ(one->indirect_columns.front().source_column_index, 0);

  gqe::qep::state_container mixed{owned_int64({1, 2}), projected, owned_int64({3, 4})};
  auto three = gqe::qep::indirect_table_view::try_from(mixed);
  ASSERT_TRUE(three.has_value());
  EXPECT_EQ(three->columns.size(), 3);
  EXPECT_EQ(three->indirect_columns.front().output_index, 1);

  gqe::qep::state_container masked{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
      gqe::query_stream::default_stream()),
    projected,
    owned_int64({5, 6})};
  auto masked_view = gqe::qep::indirect_table_view::try_from(masked);
  ASSERT_TRUE(masked_view.has_value());
  ASSERT_TRUE(masked_view->mask.has_value());
  EXPECT_EQ(masked_view->mask->type().id(), cudf::type_id::BOOL8);
  ASSERT_EQ(masked_view->indirect_columns.size(), 1);
  EXPECT_EQ(masked_view->indirect_columns.front().output_index, 0);

  bool_column_wrapper borrowed_mask_owner({true, false, true, true}, {true, true, false, true});
  auto const borrowed_mask = cudf::slice(borrowed_mask_owner, {1, 3}).front();
  gqe::qep::state_container borrowed_masked{
    gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask_view{borrowed_mask}), projected};
  auto borrowed_masked_view = gqe::qep::indirect_table_view::try_from(borrowed_masked);
  ASSERT_TRUE(borrowed_masked_view.has_value());
  ASSERT_TRUE(borrowed_masked_view->mask.has_value());
  EXPECT_EQ(borrowed_masked_view->mask->head(), borrowed_mask.head());
  EXPECT_EQ(borrowed_masked_view->mask->null_mask(), borrowed_mask.null_mask());
  EXPECT_EQ(borrowed_masked_view->mask->offset(), borrowed_mask.offset());
  EXPECT_EQ(borrowed_masked_view->mask->null_count(), borrowed_mask.null_count());
  EXPECT_EQ(borrowed_masked_view->indirect_columns.front().output_index, 0);
}

TEST(QepIndirectTableView, NonIndirectShapesReturnNullopt)
{
  gqe::qep::state_container empty;
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(empty).has_value());

  gqe::qep::state_container plain{owned_int64({1})};
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(plain).has_value());

  gqe::qep::state_container raw_positions{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{positions({0})})};
  EXPECT_FALSE(gqe::qep::indirect_table_view::try_from(raw_positions).has_value());
}

TEST(QepIndirectTableView, RejectsMalformedClaimedShapes)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20}), positions({1, 0})));

  gqe::qep::state_container null_slot{projected, nullptr};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(null_slot), std::logic_error);

  gqe::qep::state_container mismatched_direct{projected, owned_int64({1})};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(mismatched_direct),
               std::logic_error);

  gqe::qep::state_container mismatched_mask{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
      gqe::query_stream::default_stream()),
    projected};
  EXPECT_THROW(std::ignore = gqe::qep::indirect_table_view::try_from(mismatched_mask),
               std::logic_error);
}

TEST(QepIndirectTableView, LogicalGeometryUsesPositionListInsteadOfSourceRows)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20, 30}), positions({2, 0})));
  gqe::qep::state_container table{projected};
  auto view = gqe::qep::indirect_table_view::try_from(table);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->row_count, 2);
  EXPECT_EQ(view->indirect_columns.front().gather_source->relation_source()->row_count(), 3);
  EXPECT_EQ(gqe::qep::get_row_count(table), 2);
}

TEST(QepIndirectTableSlice, RejectsRangesOutsideLogicalRows)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20, 30}), positions({2, 0})));
  gqe::qep::state_container table{projected};
  auto view = gqe::qep::indirect_table_view::try_from(table);
  ASSERT_TRUE(view.has_value());

  auto slice = [&](cudf::size_type begin, cudf::size_type end) {
    return gqe::qep::slice_indirect_table(*view,
                                          begin,
                                          end,
                                          gqe::query_stream::default_stream(),
                                          cudf::get_current_device_resource_ref());
  };
  EXPECT_THROW(std::ignore = slice(-1, 0), std::invalid_argument);
  EXPECT_THROW(std::ignore = slice(2, 1), std::invalid_argument);
  EXPECT_THROW(std::ignore = slice(0, 3), std::invalid_argument);
}

TEST(QepIndirectState, GenericHelpersHandleIndirectAndPositionListStates)
{
  auto projected = indirect(one_column_gather(owned_int64({10, 20, 30}), positions({2, 0})));
  gqe::qep::state_container table{projected};
  EXPECT_FALSE(gqe::qep::to_table_view(table).has_value());

  gqe::qep::state_container raw_positions{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{positions({2, 0, 2})})};
  EXPECT_EQ(gqe::qep::get_row_count(raw_positions), 3);
  EXPECT_FALSE(gqe::qep::to_table_view(raw_positions).has_value());
}

TEST(QepIndirectState, UnionsAndDeduplicatesReachableDeallocationStreams)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const first_stream    = context.create_stream("first");
  auto const second_stream   = context.create_stream("second");
  auto const position_stream = context.create_stream("positions");
  std::array<int32_t, 2> const values{10, 20};
  std::array<int32_t, 2> const position_values{1, 0};
  auto first  = owned_int32(values, first_stream);
  auto second = owned_int32(values, second_stream);

  auto relation      = gqe::qep::indirect_table_source_state::make({first, second, first});
  auto position_list = positions(position_values, position_stream);
  auto gather        = gqe::qep::indirect_gather_source_state::make(relation, position_list);
  auto projected     = gqe::qep::indirect_column_state::make(gather, 0);

  ASSERT_EQ(relation->deallocation_streams().size(), 2);
  ASSERT_EQ(gather->deallocation_streams().size(), 3);
  ASSERT_EQ(projected.deallocation_streams().size(), 3);
  EXPECT_TRUE(contains_stream(projected.deallocation_streams(), first_stream));
  EXPECT_TRUE(contains_stream(projected.deallocation_streams(), second_stream));
  EXPECT_TRUE(contains_stream(projected.deallocation_streams(), position_stream));
}

TEST(QepIndirectTableSlice, IncludesSourceAndSlicedPositionListDeallocationStreams)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const relation_stream = context.create_stream("relation");
  auto const position_stream = context.create_stream("positions");
  auto const slice_stream    = context.create_stream("slice");
  std::array<int32_t, 3> const values{10, 20, 30};
  std::array<int32_t, 3> const position_values{2, 0, 1};
  auto relation_state = owned_int32(values, relation_stream);
  auto position_list  = positions(position_values, position_stream);
  auto relation       = gqe::qep::indirect_table_source_state::make({relation_state});
  auto gather         = gqe::qep::indirect_gather_source_state::make(relation, position_list);
  gqe::qep::state_container input{gqe::qep::indirect_column_state::make(gather, 0)};
  auto view = gqe::qep::indirect_table_view::try_from(input);
  ASSERT_TRUE(view.has_value());

  auto sliced = gqe::qep::slice_indirect_table(
    *view, 0, 2, slice_stream, cudf::get_current_device_resource_ref());
  ASSERT_EQ(sliced.size(), 1);
  ASSERT_EQ(sliced.front().deallocation_streams().size(), 3);
  EXPECT_TRUE(contains_stream(sliced.front().deallocation_streams(), relation_stream));
  EXPECT_TRUE(contains_stream(sliced.front().deallocation_streams(), position_stream));
  EXPECT_TRUE(contains_stream(sliced.front().deallocation_streams(), slice_stream));
}
