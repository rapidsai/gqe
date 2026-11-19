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

#include <gqe/qep/state.hpp>

#include "lifetime_probe.hpp"

#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

[[nodiscard]] std::vector<cudaStream_t> raw_streams(std::span<gqe::query_stream const> streams)
{
  std::vector<cudaStream_t> result;
  result.reserve(streams.size());
  for (auto const& stream : streams) {
    result.push_back(stream.ref().get());
  }
  return result;
}

TEST(QepSharedStateTest, SupportsDefaultNullCopyAndMoveSemantics)
{
  qep::shared_state empty;
  qep::shared_state null{nullptr};
  EXPECT_EQ(empty, nullptr);
  EXPECT_EQ(null, nullptr);
  EXPECT_EQ(empty, null);

  bool is_destroyed = false;
  auto state =
    gqe::test::track_lifetime(qep::make_shared_state(state_kind::row_count{3}), is_destroyed);
  auto copy  = state;
  auto moved = std::move(copy);
  EXPECT_NE(state, nullptr);
  EXPECT_EQ(copy, nullptr);
  EXPECT_EQ(state.get(), moved.get());
  EXPECT_FALSE(is_destroyed);

  state.reset();
  EXPECT_FALSE(is_destroyed);
  moved.reset();
  EXPECT_TRUE(is_destroyed);
}

TEST(QepSharedStateTest, RequiresExplicitDeallocationStreamForDirectlyOwnedStorage)
{
  auto column = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  EXPECT_THROW((void)qep::make_shared_state(state_kind::cudf_column{std::move(column)}),
               std::invalid_argument);

  auto mask = cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8});
  EXPECT_THROW(
    (void)qep::state_container_builder().add_state(state_kind::valid_mask{std::move(mask)}),
    std::invalid_argument);
}

TEST(QepSharedStateTest, EmptyOwnedAndBorrowedStatesExposeTheirDeallocationMetadata)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const column_stream = context.create_stream("column");
  auto const mask_stream   = context.create_stream("mask");
  auto owned               = qep::make_shared_state(
    state_kind::cudf_column{cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32})},
    column_stream);
  auto borrowed = qep::make_shared_state(
    state_kind::cudf_column_view{std::get<state_kind::cudf_column>(*owned).column->view()});
  EXPECT_EQ(raw_streams(owned.deallocation_streams()),
            std::vector<cudaStream_t>{column_stream.ref().get()});
  EXPECT_TRUE(borrowed.deallocation_streams().empty());

  auto mask = qep::make_shared_state(
    state_kind::valid_mask{cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8})},
    mask_stream);
  auto borrowed_mask =
    qep::make_shared_state(state_kind::valid_mask_view{*qep::to_valid_mask_view(mask)});
  EXPECT_EQ(raw_streams(mask.deallocation_streams()),
            std::vector<cudaStream_t>{mask_stream.ref().get()});
  EXPECT_TRUE(borrowed_mask.deallocation_streams().empty());

  std::array<cudf::data_type, 2> const types{cudf::data_type{cudf::type_id::INT32},
                                             cudf::data_type{cudf::type_id::STRING}};
  auto empty_table = qep::make_empty_state_container(types);
  ASSERT_EQ(empty_table.size(), types.size());
  for (auto const& state : empty_table) {
    EXPECT_TRUE(state.deallocation_streams().empty());
  }
}

TEST(QepSharedStateTest, StateVariantAccessIsConstWhileOwnedPointeesFollowTheDocumentedContract)
{
  auto state = qep::make_shared_state(state_kind::row_count{3});

  static_assert(std::is_const_v<std::remove_reference_t<decltype(*state)>>);
  static_assert(std::is_same_v<decltype(state.get()), state_kind::type const*>);
  EXPECT_EQ(std::get<state_kind::row_count>(*state).value, 3);
}

TEST(QepSharedStateTest, AnchoredStateReportsTheUnionOfBothDeallocationStreamSets)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const first_stream  = context.create_stream("first");
  auto const second_stream = context.create_stream("second");
  auto first               = qep::make_shared_state(
    state_kind::cudf_column{cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32})},
    first_stream);
  auto second = qep::make_shared_state(
    state_kind::cudf_column{cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32})},
    second_stream);

  auto anchored = qep::make_anchored_state(first, second);
  EXPECT_EQ(raw_streams(anchored.deallocation_streams()),
            (std::vector<cudaStream_t>{first_stream.ref().get(), second_stream.ref().get()}));
}

}  // namespace
