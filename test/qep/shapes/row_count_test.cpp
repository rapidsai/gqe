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

#include <gqe/qep/shapes/row_count.hpp>

#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <stdexcept>
#include <tuple>
#include <utility>

using gqe::qep::make_row_count_container;
using gqe::qep::make_shared_state;
using gqe::qep::state_container;
using gqe::qep::state_container_view;
using gqe::qep::try_row_count;
namespace state_kind = gqe::qep::state_kind;

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

/**
 * @brief The round trip: a container built by `make_row_count_container(N)` is detected as
 *        count-only and yields back `N`.
 */
TEST(TryRowCount, MakeRowCountContainerRoundTrips)
{
  auto const container = make_row_count_container(5);
  EXPECT_THAT(try_row_count(state_container_view(container)), ::testing::Optional(5));
}

/**
 * @brief An empty container has no leading slot and is not count-only.
 */
TEST(TryRowCount, EmptyContainerReturnsNullopt)
{
  state_container empty;
  EXPECT_THAT(try_row_count(state_container_view(empty)), ::testing::Eq(std::nullopt));
}

/**
 * @brief A single column slot is a regular (column-bearing) shape, not count-only.
 */
TEST(TryRowCount, SingleColumnContainerReturnsNullopt)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  EXPECT_THAT(try_row_count(state_container_view(c)), ::testing::Eq(std::nullopt));
}

/**
 * @brief Count-only is a *lone* `row_count` slot: a `row_count` followed by columns is a regular
 *        table (with a leading count hint), not count-only.
 */
TEST(TryRowCount, RowCountFollowedByColumnReturnsNullopt)
{
  state_container c;
  c.push_back(make_shared_state(state_kind::row_count{3}));
  c.push_back(make_shared_state(state_kind::cudf_column{int64_column_wrapper{1, 2, 3}.release()}));
  EXPECT_THAT(try_row_count(state_container_view(c)), ::testing::Eq(std::nullopt));
}

/**
 * @brief A null slot in an otherwise count-only-shaped container is a caller bug, not a shape
 *        mismatch: fail loudly rather than silently report "not count-only".
 */
TEST(TryRowCount, NullSlotThrows)
{
  state_container c;
  c.push_back(nullptr);
  EXPECT_THROW(std::ignore = try_row_count(state_container_view(c)), std::logic_error);
}
