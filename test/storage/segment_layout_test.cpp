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

#include <gqe/storage/segment_layout.hpp>

#include <gqe/storage/shared_segment.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

constexpr std::size_t segment_bytes = std::size_t{8} << 20;

constexpr auto policy = gqe::segment_layout_policy_type::single_arena_unplaced;

}  // namespace

TEST(SegmentLayoutTest, PolicyNameRoundTrip)
{
  for (auto candidate : {policy, gqe::segment_layout_policy_type::per_node_arenas_bound_overflow}) {
    EXPECT_EQ(gqe::from_string<gqe::segment_layout_policy_type>(gqe::to_string(candidate)),
              candidate);
  }
  EXPECT_EQ(gqe::from_string<gqe::segment_layout_policy_type>("PER_NODE_ARENAS_BOUND_OVERFLOW"),
            gqe::segment_layout_policy_type::per_node_arenas_bound_overflow);
  EXPECT_EQ(gqe::from_string<gqe::segment_layout_policy_type>("SINGLE_ARENA_UNPLACED"), policy);
  EXPECT_THROW(std::ignore = gqe::from_string<gqe::segment_layout_policy_type>("bogus"),
               std::invalid_argument);
  EXPECT_EQ(gqe::to_string(gqe::segment_layout_policy_type::num_policies), "unknown");
}

TEST(SegmentLayoutTest, RejectsUnusableSizes)
{
  auto const page_bytes   = gqe::page_kind{gqe::page_kind::system_default}.size();
  auto const header_bytes = gqe::shared_segment_manager::header_bytes(page_bytes, 1, 1);

  EXPECT_THROW(std::ignore = gqe::plan_segment_layout(policy, header_bytes), std::runtime_error);
  EXPECT_THROW(std::ignore = gqe::plan_segment_layout(policy, header_bytes + page_bytes - 1),
               std::runtime_error);
  EXPECT_NO_THROW(std::ignore = gqe::plan_segment_layout(policy, header_bytes + page_bytes));
  EXPECT_THROW(std::ignore = gqe::plan_segment_layout(gqe::segment_layout_policy_type::num_policies,
                                                      segment_bytes),
               std::invalid_argument);
}

TEST(SegmentLayoutTest, SparseNodesUseNodeIdsAndNearestFirstOrder)
{
  std::array const nodes{0, 2, 5};
  // Node 2 is closer to 5 than to 0. Node 1 has no arena and prefers 2.
  auto distance = [](int from, int to) {
    if (from == to) { return 10; }
    if ((from == 2 && to == 5) || (from == 5 && to == 2)) { return 20; }
    if (from == 1 && to == 2) { return 15; }
    return 30;
  };
  auto const layout = gqe::detail::plan_bound_arenas(segment_bytes, nodes, 6, distance);
  ASSERT_EQ(layout.arenas.size(), 3);
  ASSERT_EQ(layout.fallback_order_by_node.size(), 6);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(layout.arenas[i].mode, gqe::arena_placement::bind);
    for (int node = 0; node < 6; ++node) {
      EXPECT_EQ(layout.arenas[i].nodes.contains(node), node == nodes[i]);
    }
  }
  EXPECT_EQ(layout.fallback_order_by_node[0], (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(layout.fallback_order_by_node[1], (std::vector<int>{1, 0, 2}));
  EXPECT_EQ(layout.fallback_order_by_node[2], (std::vector<int>{1, 2, 0}));
  EXPECT_EQ(layout.fallback_order_by_node[5], (std::vector<int>{2, 1, 0}));
  // Remote distance ties retain arena order.
  EXPECT_EQ(layout.fallback_order_by_node[3], (std::vector<int>{0, 1, 2}));
}

TEST(SegmentLayoutTest, ArenasFillMappingWithAlignedStartsAndLastRemainder)
{
  std::array const nodes{0, 1, 2};
  auto const page   = gqe::page_kind{gqe::page_kind::system_default}.size();
  auto const header = gqe::shared_segment_manager::header_bytes(page, 3, 3);
  auto const bytes  = header + 10 * page + 17;
  auto const layout = gqe::detail::plan_bound_arenas(
    bytes, nodes, 3, [](int from, int to) { return from == to ? 10 : 20; });
  ASSERT_EQ(layout.arenas.size(), 3);
  EXPECT_EQ(layout.arenas[0].bytes, 3 * page);
  EXPECT_EQ(layout.arenas[1].bytes, 3 * page);
  EXPECT_EQ(layout.arenas[2].bytes, 4 * page + 17);
  std::size_t offset = header;
  for (auto const& arena : layout.arenas) {
    EXPECT_EQ(offset % page, 0);
    offset += arena.bytes;
  }
  EXPECT_EQ(offset, bytes);
}

TEST(SegmentLayoutTest, BoundPolicyRejectsEmptyTopologyAndInsufficientSpace)
{
  std::array const nodes{0, 2};
  auto const distance = [](int from, int to) { return from == to ? 10 : 20; };
  auto const page     = gqe::page_kind{gqe::page_kind::system_default}.size();
  auto const header   = gqe::shared_segment_manager::header_bytes(page, 2, 3);
  EXPECT_THROW(std::ignore = gqe::detail::plan_bound_arenas(segment_bytes, {}, 1, distance),
               std::runtime_error);
  EXPECT_THROW(std::ignore = gqe::detail::plan_bound_arenas(header, nodes, 3, distance),
               std::runtime_error);
  EXPECT_THROW(
    std::ignore = gqe::detail::plan_bound_arenas(header + 2 * page - 1, nodes, 3, distance),
    std::runtime_error);
  auto const layout = gqe::detail::plan_bound_arenas(header + 2 * page, nodes, 3, distance);
  ASSERT_EQ(layout.arenas.size(), 2);
  EXPECT_EQ(layout.arenas[0].bytes, page);
  EXPECT_EQ(layout.arenas[1].bytes, page);
}

TEST(SegmentLayoutTest, LocalArenaPrecedesRemoteDistanceTies)
{
  std::array const nodes{0, 1, 2};
  auto const layout =
    gqe::detail::plan_bound_arenas(segment_bytes, nodes, 3, [](int, int) { return 10; });
  EXPECT_EQ(layout.fallback_order_by_node[0], (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(layout.fallback_order_by_node[1], (std::vector<int>{1, 0, 2}));
  EXPECT_EQ(layout.fallback_order_by_node[2], (std::vector<int>{2, 0, 1}));
}

TEST(SegmentLayoutTest, SingleUsableNodeHasOneArenaForEveryRequester)
{
  std::array const nodes{2};
  auto const layout = gqe::detail::plan_bound_arenas(
    segment_bytes, nodes, 3, [](int from, int to) { return from == to ? 10 : 20; });
  ASSERT_EQ(layout.arenas.size(), 1);
  ASSERT_EQ(layout.fallback_order_by_node.size(), 3);
  EXPECT_TRUE(layout.arenas[0].nodes.contains(2));
  for (auto const& row : layout.fallback_order_by_node) {
    EXPECT_EQ(row, (std::vector<int>{0}));
  }
  auto const page = gqe::page_kind{layout.pages}.size();
  EXPECT_EQ(layout.arenas[0].bytes,
            segment_bytes - gqe::shared_segment_manager::header_bytes(page, 1, 3));
}
