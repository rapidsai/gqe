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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

class JoinPerfectHashTest : public gqe::test::BaseFixture {
 protected:
  /**
   * @brief Register `left_table {a INT64, b INT32}` and `right_table {c INT64, d INT32}` with the
   * given unique key-sets, then build an optimizer running uniqueness_propagation +
   * join_perfect_hash.
   *
   * @param left_unique_keys  Unique key-sets (column-name groups) to register on "left_table".
   * @param right_unique_keys Unique key-sets (column-name groups) to register on "right_table".
   */
  void initialize_optimizer(std::vector<std::vector<std::string>> left_unique_keys,
                            std::vector<std::vector<std::string>> right_unique_keys)
  {
    catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());

    catalog->register_table(
      "left_table",
      {{"a", cudf::data_type(cudf::type_id::INT64)}, {"b", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      std::move(left_unique_keys));

    catalog->register_table(
      "right_table",
      {{"c", cudf::data_type(cudf::type_id::INT64)}, {"d", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      std::move(right_unique_keys));

    gqe::optimizer::optimization_configuration cfg(
      {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
       gqe::optimizer::logical_optimization_rule_type::join_perfect_hash},
      {});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&cfg, catalog.get());
  }

  /**
   * @brief Convenience overload registering a single-column unique key ("a" / "c") per side.
   *
   * @param right_unique Whether to register a unique key on "right_table".c
   * @param left_unique  Whether to register a unique key on "left_table".a
   */
  void initialize_optimizer(bool right_unique = true, bool left_unique = false)
  {
    initialize_optimizer(left_unique ? std::vector<std::vector<std::string>>{{"a"}}
                                     : std::vector<std::vector<std::string>>{},
                         right_unique ? std::vector<std::vector<std::string>>{{"c"}}
                                      : std::vector<std::vector<std::string>>{});
  }

  /**
   * @brief Build an inner join of left_table and right_table.
   *
   * The projection keeps all four columns.
   *
   * @param join_condition Join predicate (ownership transferred).
   * @param join_type      Logical join type (defaults to inner).
   * @return The constructed logical join relation.
   */
  std::unique_ptr<gqe::logical::relation> make_join(
    std::unique_ptr<gqe::expression> join_condition,
    gqe::join_type_type join_type = gqe::join_type_type::inner)
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

    return std::make_unique<gqe::logical::join_relation>(
      std::move(left_read),
      std::move(right_read),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(join_condition),
      join_type,
      proj);
  }

  std::unique_ptr<gqe::catalog> catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

/** EQUAL join on a right-unique column sets the right perfect-hash policy. */
TEST_F(JoinPerfectHashTest, EqualOnUniqueRhsSetsRightPolicy)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  // a (left col 0) == c (right col 0, global col 2)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::right);
}

/** EQUAL join on a left-unique column sets the left perfect-hash policy. */
TEST_F(JoinPerfectHashTest, EqualOnUniqueLhsSetsLeftPolicy)
{
  initialize_optimizer(/*right_unique=*/false, /*left_unique=*/true);

  // a (left col 0) == c (right col 0, global col 2)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::left);
}

/** When both sides have unique keys on the joined columns, either-policy is set. */
TEST_F(JoinPerfectHashTest, EqualOnUniqueBothSidesSetsEitherPolicy)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/true);

  // a (left col 0) == c (right col 0, global col 2)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::either);
}

/** When neither side has a unique key on the join column, policy remains none. */
TEST_F(JoinPerfectHashTest, NoUniqueKeyNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/false, /*left_unique=*/false);

  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::none);
}

/**
 * NULL_EQUALS (IS NOT DISTINCT FROM) must NOT enable perfect hashing even when the key is unique.
 * A nullable UNIQUE column permits multiple NULL rows; perfect hashing on such a key would map
 * different NULL rows to the same slot and produce incorrect results.
 */
TEST_F(JoinPerfectHashTest, NullEqualsOnUniqueRhsNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::nulls_equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::none);
}

/** Non-equi predicate (a > c) should not trigger perfect hashing. */
TEST_F(JoinPerfectHashTest, NonEquiPredicateNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::greater_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::none);
}

/** Left outer join should not be eligible even with a unique RHS key. */
TEST_F(JoinPerfectHashTest, LeftJoinNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan = make_join(
    std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr), gqe::join_type_type::left);

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::none);
}

/** Composite equi-join where the right unique key is fully covered sets the right policy. */
TEST_F(JoinPerfectHashTest, CompositeEquiJoinFullyCoveredSetsRightPolicy)
{
  // left_table: no unique key; right_table: composite unique key on (c, d).
  initialize_optimizer(std::vector<std::vector<std::string>>{}, {{"c", "d"}});

  // a==c AND b==d
  auto lhs_a   = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_c   = std::make_shared<gqe::column_reference_expression>(2);
  auto lhs_b   = std::make_shared<gqe::column_reference_expression>(1);
  auto rhs_d   = std::make_shared<gqe::column_reference_expression>(3);
  auto cond_ac = std::make_unique<gqe::equal_expression>(lhs_a, rhs_c);
  auto cond_bd = std::make_unique<gqe::equal_expression>(lhs_b, rhs_d);
  auto cond = std::make_unique<gqe::logical_and_expression>(std::move(cond_ac), std::move(cond_bd));

  std::shared_ptr<gqe::logical::relation> plan = make_join(std::move(cond));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::right);
}

/**
 * Joining on only one column of a composite unique key does NOT qualify for perfect hashing,
 * because the key set is not fully covered by the equi-join keys.
 */
TEST_F(JoinPerfectHashTest, PartialCompositeUniqueKeyNoPolicySet)
{
  // right_table has a composite unique key on (c, d); we only join on c.
  initialize_optimizer(std::vector<std::vector<std::string>>{}, {{"c", "d"}});

  // Only a==c (does not cover d, which is part of the composite key)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::none);
}

/** join_perfect_hash must not interfere with the join_unique_keys policy when run together. */
TEST_F(JoinPerfectHashTest, BothRulesCoexistIndependently)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  // Also add join_unique_keys to the config.
  gqe::optimizer::optimization_configuration cfg(
    {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys,
     gqe::optimizer::logical_optimization_rule_type::join_perfect_hash},
    {});
  optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&cfg, catalog.get());

  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  // Both rules should independently set their policies.
  EXPECT_EQ(join->unique_keys_policy(), gqe::unique_keys_policy::right);
  EXPECT_EQ(join->perfect_hash_policy(), gqe::perfect_hash_policy::right);
}
