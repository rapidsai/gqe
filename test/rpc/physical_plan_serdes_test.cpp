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

#include <gqe/expression/literal.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace gqe::rpc {
namespace {

std::shared_ptr<physical::read_relation> make_read(int num_columns = 1)
{
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> data_types;
  for (int i = 0; i < num_columns; ++i) {
    column_names.push_back("col" + std::to_string(i));
    data_types.emplace_back(cudf::type_id::INT64);
  }
  return std::make_shared<physical::read_relation>(
    std::vector<std::shared_ptr<physical::relation>>{},
    std::move(column_names),
    "t",
    nullptr,
    std::move(data_types));
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

proto::PhysicalRelation serialize_plan(std::shared_ptr<physical::relation> root)
{
  return serialize_physical_plan(physical::make_identified_plan(std::move(root)));
}

template <typename DecimalT>
std::shared_ptr<physical::filter_relation> make_decimal_filter(typename DecimalT::rep rep,
                                                               int32_t scale)
{
  using rep_t = typename DecimalT::rep;
  return std::make_shared<physical::filter_relation>(
    make_read(),
    std::vector<std::shared_ptr<physical::relation>>{},
    std::make_unique<literal_expression<DecimalT>>(
      DecimalT{numeric::scaled_integer<rep_t>{rep, static_cast<numeric::scale_type>(scale)}}),
    std::vector<cudf::size_type>{});
}

template <typename DecimalT>
void expect_decimal_round_trip(typename DecimalT::rep rep, int32_t scale)
{
  auto original = make_decimal_filter<DecimalT>(rep, scale);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* f       = dynamic_cast<physical::filter_relation*>(restored.root().get());
  ASSERT_NE(f, nullptr);
  auto* lit = dynamic_cast<literal_expression<DecimalT>*>(f->condition_unsafe());
  ASSERT_NE(lit, nullptr);
  EXPECT_EQ(lit->value().value(), rep);
  EXPECT_EQ(lit->data_type({}).scale(), scale);
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

std::shared_ptr<physical::bounded_hash_aggregate_relation> make_bounded_hash_aggregate(
  std::vector<cudf::size_type> key_indices,
  std::vector<physical::bounded_hash_aggregate_relation::value_type> values,
  int64_t group_count_bound,
  std::optional<double> desired_load_factor = std::nullopt)
{
  return std::make_shared<physical::bounded_hash_aggregate_relation>(make_read(/*num_columns=*/4),
                                                                     std::move(key_indices),
                                                                     std::move(values),
                                                                     group_count_bound,
                                                                     desired_load_factor);
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
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_TRUE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripUseMarkJoin_True)
{
  auto original = make_broadcast_join(false, false, true, false);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_TRUE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_broadcast_join(false, false, false, true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_TRUE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripPerfectHashing_True)
{
  auto original = make_broadcast_join(true, false, false, false);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_TRUE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BroadcastJoin_RoundTripAllFalseDefault)
{
  auto original = make_broadcast_join(false, false, false, false);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_FALSE(j->use_hash_map_cache());
  EXPECT_FALSE(j->use_mark_join());
  EXPECT_FALSE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, ShuffleJoin_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_shuffle_join(false, true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* j       = dynamic_cast<physical::shuffle_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);
  EXPECT_TRUE(j->use_like_shift_and());
  EXPECT_FALSE(j->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, Filter_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_filter(true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* f       = dynamic_cast<physical::filter_relation*>(restored.root().get());
  ASSERT_NE(f, nullptr);
  EXPECT_TRUE(f->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, Project_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_project(true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* p       = dynamic_cast<physical::project_relation*>(restored.root().get());
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(p->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, Sort_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_sort(true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* s       = dynamic_cast<physical::concatenate_sort_relation*>(restored.root().get());
  ASSERT_NE(s, nullptr);
  EXPECT_TRUE(s->use_like_shift_and());
}

TEST(PhysicalPlanSerdesTest, TopK_RoundTrip)
{
  auto original = std::make_shared<physical::top_k_relation>(
    make_read(), 0, cudf::order::DESCENDING, cudf::null_order::BEFORE, 7, 13);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* top_k   = dynamic_cast<physical::top_k_relation*>(restored.root().get());
  ASSERT_NE(top_k, nullptr);
  EXPECT_EQ(top_k->key_index(), 0);
  EXPECT_EQ(top_k->column_order(), cudf::order::DESCENDING);
  EXPECT_EQ(top_k->null_precedence(), cudf::null_order::BEFORE);
  EXPECT_EQ(top_k->offset(), 7);
  EXPECT_EQ(top_k->count(), 13);
}

TEST(PhysicalPlanSerdesTest, Aggregate_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_aggregate(false, true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* a       = dynamic_cast<physical::concatenate_aggregate_relation*>(restored.root().get());
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(a->use_like_shift_and());
  EXPECT_FALSE(a->perfect_hashing());
}

TEST(PhysicalPlanSerdesTest, BoundedHashAggregate_RoundTripDefaultLoadFactor)
{
  auto original =
    make_bounded_hash_aggregate({0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* a       = dynamic_cast<physical::bounded_hash_aggregate_relation*>(restored.root().get());
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->group_count_bound(), 42);
  EXPECT_FALSE(a->desired_load_factor().has_value());
  EXPECT_EQ(a->key_indices(), std::vector<cudf::size_type>{0});
  ASSERT_EQ(a->values().size(), 1);
  EXPECT_EQ(a->values()[0].kind, cudf::aggregation::SUM);
  EXPECT_EQ(a->values()[0].value_index, 1);
}

TEST(PhysicalPlanSerdesTest, BoundedHashAggregate_RoundTripExplicitLoadFactor)
{
  auto original = make_bounded_hash_aggregate(
    {0}, {{cudf::aggregation::SUM, 1, gqe::nullability::nullable}}, 42, 0.7);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* a       = dynamic_cast<physical::bounded_hash_aggregate_relation*>(restored.root().get());
  ASSERT_NE(a, nullptr);
  ASSERT_TRUE(a->desired_load_factor().has_value());
  EXPECT_DOUBLE_EQ(*a->desired_load_factor(), 0.7);
}

TEST(PhysicalPlanSerdesTest, BoundedHashAggregate_RoundTripValueNotNull)
{
  auto original =
    make_bounded_hash_aggregate({0},
                                {{cudf::aggregation::SUM, 1, nullability::not_null},
                                 {cudf::aggregation::MIN, 2, gqe::nullability::nullable}},
                                42);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* a       = dynamic_cast<physical::bounded_hash_aggregate_relation*>(restored.root().get());
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(a->values().size(), 2);
  EXPECT_EQ(a->values()[0].value_nullability, nullability::not_null);
  EXPECT_EQ(a->values()[1].value_nullability, nullability::nullable);
}

/**
 * The relation models grouped aggregation in general; only the QEP lowering restricts the
 * shape. Wider shapes must round-trip, so the planned composite-key and multi-aggregate
 * extensions need no wire-format change. COUNT_ALL covers the unset value-index arm.
 */
TEST(PhysicalPlanSerdesTest, BoundedHashAggregate_RoundTripMultiKeyMultiValue)
{
  auto original =
    make_bounded_hash_aggregate({0, 1},
                                {{cudf::aggregation::SUM, 2, gqe::nullability::nullable},
                                 {cudf::aggregation::MIN, 3, gqe::nullability::nullable},
                                 {cudf::aggregation::COUNT_ALL, {}, gqe::nullability::nullable}},
                                42);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* a       = dynamic_cast<physical::bounded_hash_aggregate_relation*>(restored.root().get());
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->key_indices(), (std::vector<cudf::size_type>{0, 1}));
  ASSERT_EQ(a->values().size(), 3);
  EXPECT_EQ(a->values()[0].kind, cudf::aggregation::SUM);
  EXPECT_EQ(a->values()[0].value_index, 2);
  EXPECT_EQ(a->values()[1].kind, cudf::aggregation::MIN);
  EXPECT_EQ(a->values()[1].value_index, 3);
  EXPECT_EQ(a->values()[2].kind, cudf::aggregation::COUNT_ALL);
  EXPECT_EQ(a->values()[2].value_index, std::nullopt);
}

TEST(PhysicalPlanSerdesTest, Window_RoundTripUseLikeShiftAnd_True)
{
  auto original = make_window(true);
  auto restored = deserialize_physical_plan(serialize_plan(original));
  auto* w       = dynamic_cast<physical::window_relation*>(restored.root().get());
  ASSERT_NE(w, nullptr);
  EXPECT_TRUE(w->use_like_shift_and());
}

// A relation reached from more than one parent must survive the round trip as one
// relation. `task_graph_builder` caches the tasks it generates against the relation
// pointer, so a relation restored once per parent generates its tasks once per parent
// and the work is done twice.
TEST(PhysicalPlanSerdesTest, SharedChild_RestoredAsOneRelation)
{
  auto shared   = make_read();
  auto original = std::make_shared<physical::broadcast_join_relation>(
    shared,
    shared,
    std::vector<std::shared_ptr<physical::relation>>{},
    join_type_type::inner,
    nullptr,
    std::vector<cudf::size_type>{},
    physical::broadcast_policy::right,
    unique_keys_policy::none,
    false,
    nullptr,
    nullptr,
    false,
    false,
    false);

  auto identified = physical::make_identified_plan(original);
  auto serialized = serialize_physical_plan(identified);
  EXPECT_NE(serialized.node_id(), 0);

  auto const& serialized_join = serialized.broadcast_join();
  EXPECT_NE(serialized_join.left().node_id(), 0);
  ASSERT_EQ(serialized_join.right().relation_case(), proto::PhysicalRelation::kReferenceRel);
  EXPECT_EQ(serialized_join.right().reference_rel().node_id(), serialized_join.left().node_id());

  EXPECT_EQ(identified.node_id(original.get()), serialized.node_id());
  EXPECT_EQ(identified.node_id(shared.get()), serialized_join.left().node_id());

  auto restored = deserialize_physical_plan(serialized);
  auto* j       = dynamic_cast<physical::broadcast_join_relation*>(restored.root().get());
  ASSERT_NE(j, nullptr);

  auto children = j->children_unsafe();
  ASSERT_EQ(children.size(), 2u);
  EXPECT_EQ(children[0], children[1]);
  ASSERT_EQ(restored.node_ids().size(), 2u);
  EXPECT_EQ(restored.node_id(restored.root().get()), serialized.node_id());
  EXPECT_EQ(restored.node_id(children[0]), serialized_join.left().node_id());
}

TEST(PhysicalPlanSerdesTest, SerializationPreservesIdentifiedPlanNodeIds)
{
  auto root   = make_filter(false);
  auto* child = root->children_unsafe().front();
  physical::identified_plan identified{root, {{root.get(), 17}, {child, 29}}};

  auto serialized = serialize_physical_plan(identified);

  EXPECT_EQ(serialized.node_id(), 17);
  EXPECT_EQ(serialized.filter().child().node_id(), 29);
}

TEST(PhysicalPlanSerdesTest, SerializationRejectsMissingNodeIds)
{
  auto root = make_filter(false);
  physical::identified_plan identified{root, {{root.get(), 17}}};

  EXPECT_THROW((void)serialize_physical_plan(identified), std::logic_error);
}

TEST(PhysicalPlanSerdesTest, DuplicateNodeIdRejected)
{
  proto::PhysicalRelation plan;
  plan.set_node_id(1);
  auto* union_all = plan.mutable_union_all();
  union_all->mutable_left()->set_node_id(2);
  union_all->mutable_left()->mutable_read();
  union_all->mutable_right()->set_node_id(2);
  union_all->mutable_right()->mutable_read();

  EXPECT_THROW((void)deserialize_physical_plan(plan), std::logic_error);
}

TEST(PhysicalPlanSerdesTest, NonPositiveNodeIdRejected)
{
  proto::PhysicalRelation missing_node_id;
  missing_node_id.mutable_read();
  EXPECT_THROW((void)deserialize_physical_plan(missing_node_id), std::logic_error);

  proto::PhysicalRelation negative_node_id;
  negative_node_id.set_node_id(-1);
  negative_node_id.mutable_read();
  EXPECT_THROW((void)deserialize_physical_plan(negative_node_id), std::logic_error);
}

TEST(PhysicalPlanSerdesTest, ReachableRelationWithoutNodeIdRejected)
{
  proto::PhysicalRelation plan;
  plan.set_node_id(1);
  plan.mutable_filter()->mutable_child()->mutable_read();

  EXPECT_THROW((void)deserialize_physical_plan(plan), std::logic_error);
}

TEST(PhysicalPlanSerdesTest, MissingNodeIdLookupRejected)
{
  auto original  = make_read();
  auto restored  = deserialize_physical_plan(serialize_plan(original));
  auto unrelated = make_read();

  EXPECT_THROW((void)restored.node_id(unrelated.get()), std::logic_error);
}

TEST(PhysicalPlanSerdesTest, InvalidIdentifiedPlanRejected)
{
  EXPECT_THROW((void)physical::identified_plan(nullptr, {}), std::invalid_argument);

  auto root = make_read();
  EXPECT_THROW((void)physical::identified_plan(root, {{root.get(), 0}}), std::invalid_argument);

  auto unrelated = make_read();
  EXPECT_THROW((void)physical::identified_plan(root, {{unrelated.get(), 1}}),
               std::invalid_argument);

  auto filter = make_filter(false);
  EXPECT_THROW((void)physical::identified_plan(
                 filter, {{filter.get(), 1}, {filter->children_unsafe().front(), 1}}),
               std::invalid_argument);
}

TEST(PhysicalPlanSerdesTest, Filter_RoundTripDecimal32Literal)
{
  expect_decimal_round_trip<numeric::decimal32>(12345, -2);
}

TEST(PhysicalPlanSerdesTest, Filter_RoundTripDecimal64Literal)
{
  expect_decimal_round_trip<numeric::decimal64>(-987654321012345, -6);
}

// The representation exceeds 64 bits, so an encoding that truncates to a narrower
// width fails here rather than silently returning a different value.
TEST(PhysicalPlanSerdesTest, Filter_RoundTripDecimal128Literal)
{
  expect_decimal_round_trip<numeric::decimal128>((static_cast<__int128_t>(1) << 80) + 7, -10);
}

// A scale of zero is stored in the protobuf `oneof`, where presence is tracked
// separately from the value.
TEST(PhysicalPlanSerdesTest, Filter_RoundTripDecimalScaleZero)
{
  expect_decimal_round_trip<numeric::decimal64>(42, 0);
}

TEST(PhysicalPlanSerdesTest, DataType_DecimalWithoutScaleRejected)
{
  proto::DataType dt;
  dt.set_type_id(static_cast<::proto::DataType>(static_cast<int>(cudf::type_id::DECIMAL64)));
  EXPECT_THROW((void)deserialize_data_type(dt), std::invalid_argument);
}

}  // namespace gqe::rpc
