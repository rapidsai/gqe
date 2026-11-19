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

#include <gqe/qep/shapes/logical_table.hpp>

#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

/** @brief Build an owned INT64 QEP column state. */
gqe::qep::shared_state owned_int64(std::initializer_list<int64_t> values)
{
  return gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values).release()},
    gqe::query_stream::default_stream());
}

/** @brief Build a typed immutable position-list handle. */
gqe::qep::position_list_handle make_positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

/** @brief Build one indirect logical column over one source state. */
gqe::qep::shared_state make_indirect(gqe::qep::shared_state source,
                                     gqe::qep::position_list_handle positions)
{
  auto relation =
    gqe::qep::indirect_table_source_state::make(gqe::qep::state_container{std::move(source)});
  auto gather =
    gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(positions));
  return gqe::qep::indirect_column_state::make(std::move(gather), 0);
}

TEST(QepLogicalTableView, RecognizesOwnedBorrowedSlicedNullableAndIndirectColumns)
{
  auto owned = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{{1, 2}, {true, false}}.release()},
    gqe::query_stream::default_stream());

  string_column_wrapper borrowed_owner{"zero", "un", "二", "three"};
  auto borrowed_slice = cudf::slice(borrowed_owner, {1, 3}).front();
  auto borrowed =
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{borrowed_slice});

  auto positions         = make_positions({2, 0});
  auto indirect          = make_indirect(owned_int64({10, 20, 30}), positions);
  auto const* descriptor = std::get_if<gqe::qep::state_kind::indirect_column>(indirect.get());
  ASSERT_NE(descriptor, nullptr);
  auto gather_source = descriptor->state()->gather_source();

  auto mask = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false}.release()},
    gqe::query_stream::default_stream());
  gqe::qep::state_container input{mask, owned, borrowed, indirect};

  auto logical = gqe::qep::logical_table_view::try_from(input);
  ASSERT_TRUE(logical.has_value());
  EXPECT_EQ(logical->num_columns(), 3);
  EXPECT_EQ(logical->row_count(), 2);
  ASSERT_TRUE(logical->valid_mask().has_value());
  EXPECT_EQ(logical->valid_mask()->head(),
            std::get<gqe::qep::state_kind::valid_mask>(*mask).column().view().head());
  ASSERT_NE(logical->valid_mask_state(), nullptr);
  EXPECT_EQ(*logical->valid_mask_state(), mask);

  EXPECT_EQ(logical->column_state(0), owned);
  EXPECT_EQ(logical->column_state(1), borrowed);
  EXPECT_EQ(logical->column_state(2), indirect);
  EXPECT_EQ(logical->column_type(0).id(), cudf::type_id::INT64);
  EXPECT_EQ(logical->column_type(1).id(), cudf::type_id::STRING);
  EXPECT_EQ(logical->column_type(2).id(), cudf::type_id::INT64);
  auto const borrowed_view = gqe::qep::to_column_view(logical->column_state(1));
  ASSERT_TRUE(borrowed_view.has_value());
  EXPECT_EQ(borrowed_view->head(), borrowed_slice.head());
  auto const* indirect_descriptor = gqe::qep::try_get_indirect_column(logical->column_state(2));
  ASSERT_NE(indirect_descriptor, nullptr);
  EXPECT_EQ(indirect_descriptor, descriptor->state().get());
  EXPECT_EQ(indirect_descriptor->gather_source(), gather_source);
  EXPECT_EQ(indirect_descriptor->gather_source()->position_list(), positions);
}

TEST(QepLogicalTableView, UsesPositionListGeometryIncludingZeroRows)
{
  auto indirect = make_indirect(owned_int64({1, 2, 3}), make_positions({}));
  gqe::qep::state_container input{indirect};
  auto logical = gqe::qep::logical_table_view::try_from(input);
  ASSERT_TRUE(logical.has_value());
  EXPECT_EQ(logical->row_count(), 0);
  EXPECT_EQ(logical->num_columns(), 1);
}

TEST(QepLogicalTableView, UsesEmptyDirectColumnGeometry)
{
  gqe::qep::state_container empty_direct{
    gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{int64_column_wrapper{}.release()},
                                gqe::query_stream::default_stream())};
  auto direct = gqe::qep::logical_table_view::try_from(empty_direct);
  ASSERT_TRUE(direct.has_value());
  EXPECT_EQ(direct->row_count(), 0);
  EXPECT_EQ(direct->num_columns(), 1);
}

TEST(QepLogicalTableView, ReturnsNulloptForAbsentCountOnlyAndNonRelationalShapes)
{
  EXPECT_FALSE(gqe::qep::logical_table_view::try_from({}).has_value());

  auto count_only = gqe::qep::make_row_count_container(7);
  EXPECT_FALSE(gqe::qep::logical_table_view::try_from(count_only).has_value());

  gqe::qep::state_container positions_only{
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{make_positions({0})})};
  EXPECT_FALSE(gqe::qep::logical_table_view::try_from(positions_only).has_value());
}

TEST(QepLogicalTableView, RejectsMalformedClaimedRelationsAndGeometry)
{
  gqe::qep::state_container null_slot{owned_int64({1}), nullptr};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(null_slot), std::logic_error);

  gqe::qep::state_container null_owned{gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{nullptr}, gqe::query_stream::default_stream())};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(null_owned), std::logic_error);

  gqe::qep::state_container unsupported_trailing{
    owned_int64({1}), gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1})};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(unsupported_trailing),
               std::logic_error);

  gqe::qep::state_container mismatched{owned_int64({1, 2}), owned_int64({3})};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(mismatched), std::logic_error);

  gqe::qep::state_container mismatched_mask{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
      gqe::query_stream::default_stream()),
    owned_int64({1, 2})};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(mismatched_mask),
               std::logic_error);

  gqe::qep::state_container lone_mask{gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
    gqe::query_stream::default_stream())};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(lone_mask), std::logic_error);

  gqe::qep::state_container masked_positions{
    gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true}.release()},
      gqe::query_stream::default_stream()),
    gqe::qep::make_shared_state(gqe::qep::state_kind::position_list{make_positions({0})})};
  EXPECT_THROW(std::ignore = gqe::qep::logical_table_view::try_from(masked_positions),
               std::logic_error);
}

TEST(QepLogicalTableView, CheckedAccessRejectsInvalidLogicalIndices)
{
  gqe::qep::state_container input{owned_int64({1})};
  auto logical = gqe::qep::logical_table_view::try_from(input);
  ASSERT_TRUE(logical.has_value());

  EXPECT_THROW(std::ignore = logical->column_state(-1), std::out_of_range);
  EXPECT_THROW(std::ignore = logical->column_type(1), std::out_of_range);
  EXPECT_THROW(std::ignore = logical->column_state(1), std::out_of_range);
}

TEST(QepIndirectColumnProbe, BorrowsDescriptorAndRejectsOtherKindsAndNullSlots)
{
  auto owned    = owned_int64({10, 20});
  auto borrowed = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column_view{*gqe::qep::to_column_view(owned)});
  auto indirect          = make_indirect(owned, make_positions({1, 0}));
  auto const& descriptor = std::get<gqe::qep::state_kind::indirect_column>(*indirect).state();
  std::weak_ptr<gqe::qep::indirect_column_state const> lifetime = descriptor;

  auto const* probe = gqe::qep::try_get_indirect_column(indirect);
  ASSERT_EQ(probe, descriptor.get());
  EXPECT_EQ(probe->source_column_index(), 0);
  EXPECT_EQ(gqe::qep::try_get_indirect_column(owned), nullptr);
  EXPECT_EQ(gqe::qep::try_get_indirect_column(borrowed), nullptr);
  EXPECT_EQ(gqe::qep::try_get_indirect_column(
              gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{2})),
            nullptr);
  EXPECT_THROW(std::ignore = gqe::qep::try_get_indirect_column(nullptr), std::logic_error);

  indirect.reset();
  EXPECT_TRUE(lifetime.expired());
}
