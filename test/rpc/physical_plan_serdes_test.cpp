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

#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe::rpc {
namespace {

std::shared_ptr<physical::read_relation> make_read()
{
  return std::make_shared<physical::read_relation>(
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::string>{"a"},
    "t",
    nullptr,
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}});
}

std::shared_ptr<physical::broadcast_join_relation> make_broadcast_join(bool perfect_hashing,
                                                                       bool use_hash_map_cache,
                                                                       bool use_mark_join,
                                                                       bool use_like_shift_and)
{
  return std::make_shared<physical::broadcast_join_relation>(
    make_read(),
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    join_type_type::inner,
    nullptr,
    std::vector<cudf::size_type>{},
    physical::broadcast_policy::right,
    unique_keys_policy::none,
    perfect_hashing,
    nullptr,
    nullptr,
    use_hash_map_cache,
    use_mark_join,
    use_like_shift_and);
}

std::shared_ptr<physical::shuffle_join_relation> make_shuffle_join(bool perfect_hashing,
                                                                   bool use_like_shift_and)
{
  return std::make_shared<physical::shuffle_join_relation>(
    make_read(),
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    join_type_type::inner,
    nullptr,
    std::vector<cudf::size_type>{},
    unique_keys_policy::none,
    perfect_hashing,
    use_like_shift_and);
}

std::shared_ptr<physical::filter_relation> make_filter(bool use_like_shift_and)
{
  return std::make_shared<physical::filter_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    nullptr,
    std::vector<cudf::size_type>{},
    use_like_shift_and);
}

std::shared_ptr<physical::project_relation> make_project(bool use_like_shift_and)
{
  return std::make_shared<physical::project_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::unique_ptr<expression>>{},
    use_like_shift_and);
}

std::shared_ptr<physical::concatenate_sort_relation> make_sort(bool use_like_shift_and)
{
  return std::make_shared<physical::concatenate_sort_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::unique_ptr<expression>>{},
    std::vector<cudf::order>{},
    std::vector<cudf::null_order>{},
    use_like_shift_and);
}

std::shared_ptr<physical::concatenate_aggregate_relation> make_aggregate(bool perfect_hashing,
                                                                         bool use_like_shift_and)
{
  return std::make_shared<physical::concatenate_aggregate_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::unique_ptr<expression>>{},
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>>{},
    nullptr,
    perfect_hashing,
    use_like_shift_and);
}

std::shared_ptr<physical::window_relation> make_window(bool use_like_shift_and)
{
  return std::make_shared<physical::window_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    cudf::aggregation::Kind::SUM,
    std::vector<std::unique_ptr<expression>>{},
    std::vector<std::unique_ptr<expression>>{},
    std::vector<std::unique_ptr<expression>>{},
    std::vector<std::unique_ptr<expression>>{},
    std::vector<cudf::order>{},
    window_frame_bound::type{window_frame_bound::unbounded{}},
    window_frame_bound::type{window_frame_bound::unbounded{}},
    use_like_shift_and);
}

}  // namespace

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripUseHashMapCache_True)
{
  auto original = make_broadcast_join(false, true, false, false);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_TRUE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripUseMarkJoin_True)
{
  auto original = make_broadcast_join(false, false, true, false);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_TRUE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_broadcast_join(false, false, false, true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_TRUE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripPerfectHashing_True)
{
  auto original = make_broadcast_join(true, false, false, false);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_TRUE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripAllFalseDefault)
{
  auto original = make_broadcast_join(false, false, false, false);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, ShuffleJoin_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_shuffle_join(false, true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* j       = dynamic_cast<physical::shuffle_join_relation*>(restored.get());
  ASSERT_NE(j, nullptr);
  EXPECT_TRUE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, Filter_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_filter(true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* f       = dynamic_cast<physical::filter_relation*>(restored.get());
  ASSERT_NE(f, nullptr);
  EXPECT_TRUE(f->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, Project_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_project(true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* p       = dynamic_cast<physical::project_relation*>(restored.get());
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(p->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, Sort_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_sort(true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* s       = dynamic_cast<physical::concatenate_sort_relation*>(restored.get());
  ASSERT_NE(s, nullptr);
  EXPECT_TRUE(s->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, Aggregate_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_aggregate(false, true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* a       = dynamic_cast<physical::concatenate_aggregate_relation*>(restored.get());
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->use_like_shift_and());
  EXPECT_FALSE(a->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, Window_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_window(true);
  auto restored = deserialize_physical_plan(serialize_physical_plan(original.get()));
  auto* w       = dynamic_cast<physical::window_relation*>(restored.get());
  ASSERT_NE(w, nullptr);
  EXPECT_TRUE(w->use_like_shift_and());
}

}  // namespace gqe::rpc
