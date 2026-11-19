/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/rpc/serialization/statistics.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

/**
 * @brief Build a message declaring one column count while carrying another.
 *
 * `num_columns` and `column_stats` are independent wire fields, so the two can disagree.
 *
 * @param declared_columns Value of the `num_columns` field.
 * @param carried_columns Number of `column_stats` entries to add.
 * @return The constructed message.
 */
gqe::proto::TableStatistics make_table_statistics(uint64_t declared_columns, int carried_columns)
{
  gqe::proto::TableStatistics pb;
  pb.set_num_rows(100);
  pb.set_num_row_groups(2);
  pb.set_num_columns(declared_columns);
  for (int i = 0; i < carried_columns; ++i) {
    auto* col = pb.add_column_stats();
    col->set_column_id(i);
    col->mutable_compression_stats()->mutable_fixed_width();
  }
  return pb;
}

}  // namespace

/**
 * @brief A consistent message deserializes with every field carried across.
 */
TEST(StatisticsSerdesTest, DeserializeAcceptsConsistentColumnCounts)
{
  auto const out = gqe::rpc::deserialize_table_statistics(make_table_statistics(2, 2));
  EXPECT_EQ(out.num_rows, 100);
  EXPECT_EQ(out.num_row_groups, 2u);
  EXPECT_EQ(out.num_columns, 2u);
  EXPECT_EQ(out.column_stats.size(), 2u);
}

/**
 * @brief Declaring no columns is consistent, not a violation.
 *
 * An empty row group carries its row and row-group counts and nothing per column.
 */
TEST(StatisticsSerdesTest, DeserializeAcceptsNoColumns)
{
  auto const out = gqe::rpc::deserialize_table_statistics(make_table_statistics(0, 0));
  EXPECT_EQ(out.num_columns, 0u);
  EXPECT_TRUE(out.column_stats.empty());
}

/**
 * @brief Declaring more columns than are carried must be rejected.
 *
 * This is the case that motivates the guard: consumers index `column_stats` with
 * `num_columns`, so admitting such a message defers the fault to an unrelated caller.
 */
TEST(StatisticsSerdesTest, DeserializeRejectsTooFewColumnStats)
{
  EXPECT_THROW(std::ignore = gqe::rpc::deserialize_table_statistics(make_table_statistics(3, 1)),
               std::logic_error);
}

/**
 * @brief Carrying more columns than are declared must also be rejected.
 *
 * Rejected rather than truncated, because a sender that disagrees with itself has a bug and
 * the surplus entries would be silently dropped.
 */
TEST(StatisticsSerdesTest, DeserializeRejectsTooManyColumnStats)
{
  EXPECT_THROW(std::ignore = gqe::rpc::deserialize_table_statistics(make_table_statistics(1, 3)),
               std::logic_error);
}

/**
 * @brief A round trip preserves the invariant.
 *
 * This is the path every task manager response takes, so it must hold by construction rather
 * than by the sender happening to be consistent.
 */
TEST(StatisticsSerdesTest, RoundTripPreservesColumnCounts)
{
  std::vector<cudf::data_type> const column_types = {cudf::data_type(cudf::type_id::INT32),
                                                     cudf::data_type(cudf::type_id::STRING)};
  gqe::table_statistics const original(100, column_types);

  auto const serialized = gqe::rpc::serialize_table_statistics(original);
  auto const out        = gqe::rpc::deserialize_table_statistics(serialized);
  EXPECT_EQ(out.num_columns, original.num_columns);
  EXPECT_EQ(out.column_stats.size(), original.column_stats.size());
  EXPECT_EQ(out.num_rows, original.num_rows);
}
