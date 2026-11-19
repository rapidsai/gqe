/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/optimizer/statistics.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace {

/** @brief Build `a == c` (left col 0 == right col 0). */
std::unique_ptr<gqe::expression> equality_condition()
{
  return std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::column_reference_expression>(2));
}

/** @brief Build `(a == c) AND (b > d)` — mixed equality + inequality. */
std::unique_ptr<gqe::expression> mixed_condition()
{
  auto eq =
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2));
  auto gt = std::make_shared<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(1),
    std::make_shared<gqe::column_reference_expression>(3));
  return std::make_unique<gqe::logical_and_expression>(eq, gt);
}

}  // namespace

/**
 * @brief Tests the hash-map-cache gating logic in plan_join.cpp.
 *
 * The optimizer now decides `use_hash_map_cache` based on both the user flag
 * and whether the join condition is equality-only. Mark-join on
 * left_semi/left_anti continues to force the cache on regardless of the flag.
 */
class PlanJoinHashMapCacheTest : public gqe::test::BaseFixture {
 protected:
  /**
   * @brief Register `left_table(a INT64, b INT32)` and
   *        `right_table(c INT64, d INT32)` in the catalog.
   *
   * @param left_rows  Optional row-count seed for `left_table` (>0 sets the
   *                   table_statistics).
   * @param right_rows Optional row-count seed for `right_table`.
   */
  void register_tables(int64_t left_rows = 0, int64_t right_rows = 0)
  {
    catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());

    catalog->register_table(
      "left_table",
      {{"a", cudf::data_type(cudf::type_id::INT64)}, {"b", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      std::vector<std::vector<std::string>>{});

    catalog->register_table(
      "right_table",
      {{"c", cudf::data_type(cudf::type_id::INT64)}, {"d", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      std::vector<std::vector<std::string>>{});

    if (left_rows > 0) catalog->statistics("left_table")->add_rows(left_rows);
    if (right_rows > 0) catalog->statistics("right_table")->add_rows(right_rows);
  }

  /**
   * @brief Build a logical join over `left_table` / `right_table` and lower it
   *        through `physical_plan_builder` with the given parameters.
   *
   * @param condition Join predicate (ownership transferred).
   * @param join_type Logical join type.
   * @param params    Optimization parameters that drive the gating decisions.
   * @return The resulting `broadcast_join_relation` (downcast from the
   *         physical plan root), or `nullptr` if the lowering produced a
   *         different relation type.
   */
  std::shared_ptr<gqe::physical::broadcast_join_relation const> build_physical_join(
    std::unique_ptr<gqe::expression> condition,
    gqe::join_type_type join_type,
    gqe::optimization_parameters const& params)
  {
    auto left_read = std::make_unique<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::vector<std::string>{"a", "b"},
      std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT64),
                                   cudf::data_type(cudf::type_id::INT32)},
      "left_table",
      nullptr);

    auto right_read = std::make_unique<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::vector<std::string>{"c", "d"},
      std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT64),
                                   cudf::data_type(cudf::type_id::INT32)},
      "right_table",
      nullptr);

    std::vector<cudf::size_type> proj = {0, 1, 2, 3};
    auto join                         = std::make_shared<gqe::logical::join_relation>(
      std::move(left_read),
      std::move(right_read),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(condition),
      join_type,
      proj);

    gqe::physical_plan_builder builder(catalog.get(), &params);
    auto physical = builder.build(join.get());
    return std::dynamic_pointer_cast<gqe::physical::broadcast_join_relation const>(physical);
  }

  std::unique_ptr<gqe::catalog> catalog;
};

/** Equality-only condition + flag enabled → cache enabled. */
TEST_F(PlanJoinHashMapCacheTest, EqualityOnlyWithFlagEnablesCache)
{
  register_tables();
  auto params                    = gqe::make_optimization_parameters(true);
  params.join_use_hash_map_cache = true;
  params.join_use_mark_join      = false;

  auto join = build_physical_join(equality_condition(), gqe::join_type_type::inner, params);
  ASSERT_NE(join, nullptr);
  EXPECT_TRUE(join->use_hash_map_cache());
  EXPECT_FALSE(join->use_mark_join());
}

/**
 * Mixed (non-equality) condition + flag enabled → cache disabled.
 * The optimizer now refuses to set `use_hash_map_cache` when the executor
 * would discard it at runtime anyway.
 */
TEST_F(PlanJoinHashMapCacheTest, MixedConditionWithFlagDisablesCache)
{
  register_tables();
  auto params                    = gqe::make_optimization_parameters(true);
  params.join_use_hash_map_cache = true;
  params.join_use_mark_join      = false;

  auto join = build_physical_join(mixed_condition(), gqe::join_type_type::inner, params);
  ASSERT_NE(join, nullptr);
  EXPECT_FALSE(join->use_hash_map_cache());
  EXPECT_FALSE(join->use_mark_join());
}

/** Flag off + equality-only → cache still off. */
TEST_F(PlanJoinHashMapCacheTest, FlagOffKeepsCacheOff)
{
  register_tables();
  auto params                    = gqe::make_optimization_parameters(true);
  params.join_use_hash_map_cache = false;
  params.join_use_mark_join      = false;

  auto join = build_physical_join(equality_condition(), gqe::join_type_type::inner, params);
  ASSERT_NE(join, nullptr);
  EXPECT_FALSE(join->use_hash_map_cache());
}

/**
 * Mark-join on left_semi (broadcast-left, equality-only) forces the cache on
 * even when the user flag is off.
 */
TEST_F(PlanJoinHashMapCacheTest, MarkJoinForcesCacheOnEvenWithFlagOff)
{
  // Need left_num_rows < right_num_rows so plan_join picks broadcast_policy::left,
  // which is the precondition for use_mark_join.
  register_tables(/*left_rows=*/10, /*right_rows=*/1000);
  auto params                    = gqe::make_optimization_parameters(true);
  params.join_use_hash_map_cache = false;
  params.join_use_mark_join      = true;

  auto join = build_physical_join(equality_condition(), gqe::join_type_type::left_semi, params);
  ASSERT_NE(join, nullptr);
  EXPECT_TRUE(join->use_mark_join());
  EXPECT_TRUE(join->use_hash_map_cache());
}

/**
 * Mark-join overrides the equality-only gate: even a mixed condition gets the
 * cache when mark-join is in effect (mark-join is its own self-contained cache
 * path).
 */
TEST_F(PlanJoinHashMapCacheTest, MarkJoinOverridesEqualityGate)
{
  register_tables(/*left_rows=*/10, /*right_rows=*/1000);
  auto params                    = gqe::make_optimization_parameters(true);
  params.join_use_hash_map_cache = false;
  params.join_use_mark_join      = true;

  auto join = build_physical_join(mixed_condition(), gqe::join_type_type::left_semi, params);
  ASSERT_NE(join, nullptr);
  EXPECT_TRUE(join->use_mark_join());
  EXPECT_TRUE(join->use_hash_map_cache());
}
