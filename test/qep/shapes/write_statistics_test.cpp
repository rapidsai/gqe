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

#include <gqe/qep/shapes/write_statistics.hpp>

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/types.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace {

TEST(TryWriteStatistics, MakeWriteStatisticsContainerRoundTrips)
{
  auto const data_types     = std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}};
  auto statistics           = gqe::table_statistics{7, data_types};
  statistics.num_row_groups = 2;
  auto& compression_statistics = std::get<gqe::fixed_width_compression_statistics>(
    statistics.column_stats.front().compression_stats);
  compression_statistics.compressed_size   = 17;
  compression_statistics.uncompressed_size = 31;

  auto const container = gqe::qep::make_write_statistics_container(std::move(statistics));
  auto const result    = gqe::qep::try_write_statistics(container);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->num_rows, 7);
  EXPECT_EQ(result->num_row_groups, 2);
  EXPECT_EQ(result->num_columns, 1);
  auto const& result_compression_statistics = std::get<gqe::fixed_width_compression_statistics>(
    result->column_stats.front().compression_stats);
  EXPECT_EQ(result_compression_statistics.compressed_size, 17);
  EXPECT_EQ(result_compression_statistics.uncompressed_size, 31);
}

TEST(TryWriteStatistics, OtherShapesReturnNullopt)
{
  auto empty = gqe::qep::state_container{};
  EXPECT_THAT(gqe::qep::try_write_statistics(empty), testing::Eq(std::nullopt));

  auto row_count = gqe::qep::make_row_count_container(1);
  EXPECT_THAT(gqe::qep::try_write_statistics(row_count), testing::Eq(std::nullopt));

  auto statistics = gqe::qep::make_write_statistics_container(gqe::table_statistics{});
  statistics.push_back(gqe::qep::make_shared_state(gqe::qep::state_kind::row_count{1}));
  EXPECT_THAT(gqe::qep::try_write_statistics(statistics), testing::Eq(std::nullopt));
}

TEST(TryWriteStatistics, NullSlotThrows)
{
  auto container = gqe::qep::state_container{nullptr};
  EXPECT_THROW(std::ignore = gqe::qep::try_write_statistics(container), std::logic_error);
}

}  // namespace
