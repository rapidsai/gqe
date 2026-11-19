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

class JoinUniqueKeysTest : public gqe::test::BaseFixture {
 protected:
  // Register two tables:
  //   "left_table"  — cols {a INT64, b INT32}, no unique keys
  //   "right_table" — cols {c INT64, d INT32}, c is unique (col index 0)
  // Run both uniqueness_propagation (to seed traits from catalog) and join_unique_keys.
  void initialize_optimizer(bool right_unique = true, bool left_unique = false)
  {
    catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());

    catalog->register_table(
      "left_table",
      {{"a", cudf::data_type(cudf::type_id::INT64)}, {"b", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      left_unique ? std::vector<std::vector<std::string>>{{"a"}}
                  : std::vector<std::vector<std::string>>{});

    catalog->register_table(
      "right_table",
      {{"c", cudf::data_type(cudf::type_id::INT64)}, {"d", cudf::data_type(cudf::type_id::INT32)}},
      gqe::storage_kind::system_memory{},
      gqe::partitioning_schema_kind::none{},
      right_unique ? std::vector<std::vector<std::string>>{{"c"}}
                   : std::vector<std::vector<std::string>>{});

    gqe::optimizer::optimization_configuration cfg(
      {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
       gqe::optimizer::logical_optimization_rule_type::join_unique_keys},
      {});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&cfg, catalog.get());
  }

  // Build an inner join of left_table and right_table.
  // join_condition is passed in; projection keeps all four columns.
  std::unique_ptr<gqe::logical::relation> make_join(std::unique_ptr<gqe::expression> join_condition)
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

    // All four output columns: left {0,1}, right {2,3}
    std::vector<cudf::size_type> proj = {0, 1, 2, 3};

    return std::make_unique<gqe::logical::join_relation>(
      std::move(left_read),
      std::move(right_read),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(join_condition),
      gqe::join_type_type::inner,
      proj);
  }

  std::unique_ptr<gqe::catalog> catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

// EQUAL join on a right-unique column sets the right unique-keys policy.
TEST_F(JoinUniqueKeysTest, EqualOnUniqueRhsSetsRightPolicy)
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
  EXPECT_EQ(join->unique_keys_policy(), gqe::unique_keys_policy::right);
}

// NULL_EQUALS join on a right-unique column must NOT enable the unique-key optimization.
// A nullable SQL UNIQUE constraint allows multiple NULL rows; NULL_EQUALS (IS NOT DISTINCT FROM)
// would then match a probe NULL against multiple build NULLs, violating the unique-build-side
// assumption and producing incorrect results.
TEST_F(JoinUniqueKeysTest, NullEqualsOnUniqueRhsNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  // a (left col 0) IS NOT DISTINCT FROM c (right col 0, global col 2)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::nulls_equal_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->unique_keys_policy(), gqe::unique_keys_policy::none);
}

// When both sides have unique keys on the joined columns, either-policy is set.
TEST_F(JoinUniqueKeysTest, EqualOnUniqueLhsAndRhsSetsEitherPolicy)
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
  EXPECT_EQ(join->unique_keys_policy(), gqe::unique_keys_policy::either);
}

// Non-equi predicate (a > c) should not trigger the optimization.
TEST_F(JoinUniqueKeysTest, NonEquiPredicateNoPolicySet)
{
  initialize_optimizer(/*right_unique=*/true, /*left_unique=*/false);

  // a (left col 0) > c (right col 0, global col 2)
  auto lhs_expr = std::make_shared<gqe::column_reference_expression>(0);
  auto rhs_expr = std::make_shared<gqe::column_reference_expression>(2);
  std::shared_ptr<gqe::logical::relation> plan =
    make_join(std::make_unique<gqe::greater_expression>(lhs_expr, rhs_expr));

  auto optimized = optimizer->optimize(plan);
  auto* join     = dynamic_cast<gqe::logical::join_relation*>(optimized.get());
  ASSERT_NE(join, nullptr);
  EXPECT_EQ(join->unique_keys_policy(), gqe::unique_keys_policy::none);
}
