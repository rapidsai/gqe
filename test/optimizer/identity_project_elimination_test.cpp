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
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Build a read of the 5-column INT64 table `temp_table`.
 *
 * Matches the schema registered by the `IdentityProjectElimination` fixture.
 *
 * @return A `read_relation` over `temp_table`.
 */
std::shared_ptr<gqe::logical::read_relation> make_temp_read()
{
  return std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<std::string>{"column_0", "column_1", "column_2", "column_3", "column_4"},
    std::vector<cudf::data_type>(5, cudf::data_type(cudf::type_id::INT64)),
    "temp_table",
    nullptr);
}

/**
 * @brief Build a project over `child` with the given output expressions.
 *
 * @param child Input relation to project from.
 * @param output_expressions One expression per output column.
 * @return A `project_relation` with no subquery relations.
 */
std::shared_ptr<gqe::logical::project_relation> make_project(
  std::shared_ptr<gqe::logical::relation> child,
  std::vector<std::unique_ptr<gqe::expression>> output_expressions)
{
  return std::make_shared<gqe::logical::project_relation>(
    std::move(child),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(output_expressions));
}

/**
 * @brief Build an identity project over `child`, selecting bare column references `0..n-1`.
 *
 * @param child Input relation to project from.
 * @param num_columns Number of column references to emit; pass the child's column count for a
 * strict identity, or fewer to build a non-identity column drop.
 * @return A `project_relation` whose outputs are `col_ref(0) .. col_ref(num_columns - 1)`.
 */
std::shared_ptr<gqe::logical::project_relation> make_identity_project(
  std::shared_ptr<gqe::logical::relation> child, cudf::size_type num_columns)
{
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  for (cudf::size_type i = 0; i < num_columns; ++i) {
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(i));
  }
  return make_project(std::move(child), std::move(exprs));
}

/**
 * @brief Build `GROUP BY column_0, SUM(column_1)` over `child`.
 *
 * @param child Input relation to aggregate.
 * @return An `aggregate_relation` producing 2 output columns (one key, one measure).
 */
std::shared_ptr<gqe::logical::aggregate_relation> make_aggregate(
  std::shared_ptr<gqe::logical::relation> child)
{
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(1));
  return std::make_shared<gqe::logical::aggregate_relation>(
    std::move(child),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::move(keys),
    std::move(measures));
}

/**
 * @brief Build `ORDER BY column_0 ASC` over `child`.
 *
 * @param child Input relation to sort.
 * @return A `sort_relation`, which is width-preserving and so mirrors the child's column count.
 */
std::shared_ptr<gqe::logical::sort_relation> make_sort(
  std::shared_ptr<gqe::logical::relation> child)
{
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  return std::make_shared<gqe::logical::sort_relation>(
    std::move(child),
    std::vector<std::shared_ptr<gqe::logical::relation>>(),
    std::vector<cudf::order>{cudf::order::ASCENDING},
    std::vector<cudf::null_order>{cudf::null_order::AFTER},
    std::move(sort_exprs));
}

/**
 * @brief Build `ORDER BY column_0 ASC` over `child`, with `subquery` attached as a subquery
 * relation.
 *
 * Used to place a relation at a subquery root, a position that belongs to no parent's child list.
 *
 * @param child Input relation to sort.
 * @param subquery Relation to attach as the sort's single subquery.
 * @return A `sort_relation` holding one subquery.
 */
std::shared_ptr<gqe::logical::sort_relation> make_sort_with_subquery(
  std::shared_ptr<gqe::logical::relation> child, std::shared_ptr<gqe::logical::relation> subquery)
{
  std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
  sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  return std::make_shared<gqe::logical::sort_relation>(
    std::move(child),
    std::vector<std::shared_ptr<gqe::logical::relation>>{std::move(subquery)},
    std::vector<cudf::order>{cudf::order::ASCENDING},
    std::vector<cudf::null_order>{cudf::null_order::AFTER},
    std::move(sort_exprs));
}

}  // namespace

/**
 * @brief Test fixture for the `identity_project_elimination` optimizer rule.
 *
 * Configures a single-rule optimizer pipeline over a 5-column INT64 table `temp_table`
 * ({column_0 .. column_4}), so each test observes this rule in isolation.
 */
class IdentityProjectElimination : public gqe::test::BaseFixture {
 protected:
  IdentityProjectElimination()
  {
    _catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());
    _catalog->register_table("temp_table",
                             {{"column_0", cudf::data_type(cudf::type_id::INT64)},
                              {"column_1", cudf::data_type(cudf::type_id::INT64)},
                              {"column_2", cudf::data_type(cudf::type_id::INT64)},
                              {"column_3", cudf::data_type(cudf::type_id::INT64)},
                              {"column_4", cudf::data_type(cudf::type_id::INT64)}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});

    gqe::optimizer::optimization_configuration config(
      {gqe::optimizer::logical_optimization_rule_type::identity_project_elimination}, {});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&config, _catalog.get());
  }

  /**
   * @brief Number of times `identity_project_elimination` fired.
   *
   * @return The applied-rule count, which is at most 1 per `optimize()` call regardless of how
   * many projects were removed.
   */
  std::size_t rule_count()
  {
    return optimizer->get_rule_count(
      gqe::optimizer::logical_optimization_rule_type::identity_project_elimination);
  }

  std::unique_ptr<gqe::catalog> _catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

/**
 * @brief project([col_ref(0), col_ref(1)]) over an aggregate root -> project removed, aggregate
 * becomes the new root.
 */
TEST_F(IdentityProjectElimination, RemovesIdentityOverAggregate)
{
  auto aggregate = make_aggregate(make_temp_read());  // 2 output columns
  auto plan      = make_identity_project(aggregate, 2);

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  EXPECT_NE(dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get()), nullptr);
}

/**
 * @brief project([col_ref(0) .. col_ref(4)]) over a sort root -> project removed, sort becomes the
 * new root.
 */
TEST_F(IdentityProjectElimination, RemovesIdentityOverSort)
{
  auto sort = make_sort(make_temp_read());  // width-preserving -> 5 output columns
  auto plan = make_identity_project(sort, 5);

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  EXPECT_NE(dynamic_cast<gqe::logical::sort_relation*>(optimized.get()), nullptr);
}

/**
 * @brief project([col_ref(0) .. col_ref(4)]) directly over a read root -> project removed, read
 * becomes the new root.
 */
TEST_F(IdentityProjectElimination, RemovesIdentityAtRootOverRead)
{
  auto plan = make_identity_project(make_temp_read(), 5);

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  EXPECT_NE(dynamic_cast<gqe::logical::read_relation*>(optimized.get()), nullptr);
}

/**
 * @brief 4 stacked identity projects at the plan root -> all removed in a single pass.
 *
 * Exercises the root-handling loop in `apply()`. Depth is >2 deliberately: a 2-deep chain would
 * pass even if the rule only unwrapped one level.
 */
TEST_F(IdentityProjectElimination, CollapsesChainOfIdentityProjectsAtRoot)
{
  auto aggregate = make_aggregate(make_temp_read());  // 2 output columns
  std::shared_ptr<gqe::logical::relation> plan = aggregate;
  for (int i = 0; i < 4; ++i) {  // 4 stacked identity projects above the aggregate
    plan = make_identity_project(plan, 2);
  }

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  // Every project in the chain is gone: the aggregate itself is the new root.
  EXPECT_EQ(optimized.get(), static_cast<gqe::logical::relation*>(aggregate.get()));
}

/**
 * @brief 3 stacked identity projects between a sort root and an aggregate -> all removed, leaving
 * sort over aggregate.
 *
 * Exercises the per-child loop in the visitor, a separate code path from the root handling above.
 */
TEST_F(IdentityProjectElimination, CollapsesChainOfIdentityProjectsMidPlan)
{
  auto aggregate = make_aggregate(make_temp_read());  // 2 output columns
  std::shared_ptr<gqe::logical::relation> chain = aggregate;
  for (int i = 0; i < 3; ++i) {  // 3 stacked identity projects between sort and aggregate
    chain = make_identity_project(chain, 2);
  }
  auto plan = make_sort(chain);  // sort is the root, so the chain is not root-handled

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  auto* sort = dynamic_cast<gqe::logical::sort_relation*>(optimized.get());
  ASSERT_NE(sort, nullptr);
  // The whole chain collapsed: the sort's child is the aggregate, not a project.
  ASSERT_EQ(sort->children_size(), 1u);
  EXPECT_EQ(sort->children_unsafe()[0], static_cast<gqe::logical::relation*>(aggregate.get()));
}

/**
 * @brief An identity project at the root of a subquery relation -> removed.
 *
 * A subquery root belongs to no parent's child list, so it is reached by the visitor's separate
 * subquery pass rather than the child pass.
 */
TEST_F(IdentityProjectElimination, RemovesIdentityAtSubqueryRoot)
{
  auto subquery_child = make_temp_read();  // 5 output columns
  auto plan = make_sort_with_subquery(make_temp_read(), make_identity_project(subquery_child, 5));

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  auto* sort = dynamic_cast<gqe::logical::sort_relation*>(optimized.get());
  ASSERT_NE(sort, nullptr);
  // The subquery list keeps its size, so any relation_index into it stays valid.
  ASSERT_EQ(sort->subqueries_size(), 1u);
  EXPECT_EQ(sort->subqueries_unsafe()[0],
            static_cast<gqe::logical::relation*>(subquery_child.get()));
}

/**
 * @brief A chain of stacked identity projects at a subquery root -> all removed in a single pass.
 */
TEST_F(IdentityProjectElimination, CollapsesChainOfIdentityProjectsAtSubqueryRoot)
{
  auto subquery_child                              = make_temp_read();  // 5 output columns
  std::shared_ptr<gqe::logical::relation> subquery = subquery_child;
  for (int i = 0; i < 3; ++i) {  // 3 stacked identity projects at the subquery root
    subquery = make_identity_project(subquery, 5);
  }
  auto plan = make_sort_with_subquery(make_temp_read(), subquery);

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 1u);
  auto* sort = dynamic_cast<gqe::logical::sort_relation*>(optimized.get());
  ASSERT_NE(sort, nullptr);
  ASSERT_EQ(sort->subqueries_size(), 1u);
  EXPECT_EQ(sort->subqueries_unsafe()[0],
            static_cast<gqe::logical::relation*>(subquery_child.get()));
}

/**
 * @brief project([col_ref(1), col_ref(0)]) over an aggregate: right width but reordered, so not an
 * identity -> plan untouched, rule does not fire.
 */
TEST_F(IdentityProjectElimination, KeepsReorderingProjectOverAggregate)
{
  auto aggregate = make_aggregate(make_temp_read());  // 2 output columns
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
  auto plan = make_project(aggregate, std::move(exprs));

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 0u);
  EXPECT_NE(dynamic_cast<gqe::logical::project_relation*>(optimized.get()), nullptr);
}

/**
 * @brief project carrying a computed expression (`column_1 * 2`) is not an identity -> plan
 * untouched, rule does not fire.
 */
TEST_F(IdentityProjectElimination, KeepsProjectWithExpression)
{
  std::vector<std::unique_ptr<gqe::expression>> exprs;
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
  exprs.emplace_back(std::make_unique<gqe::multiply_expression>(
    std::make_shared<gqe::column_reference_expression>(1),
    std::make_shared<gqe::literal_expression<int64_t>>(2)));
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(2));
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(3));
  exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(4));
  auto plan = make_project(make_temp_read(), std::move(exprs));

  auto optimized = optimizer->optimize(plan);

  EXPECT_EQ(rule_count(), 0u);
  EXPECT_NE(dynamic_cast<gqe::logical::project_relation*>(optimized.get()), nullptr);
}

/**
 * @brief Exercise `project_relation::is_identity()` in isolation.
 *
 * A strict identity is bare column references `0..n-1` whose count matches the child's output
 * width; column drops, reorderings, and computed expressions must all be rejected.
 */
TEST(ProjectRelationIsIdentity, DetectsIdentityAndRejectsNonIdentity)
{
  // Identity: bare column references 0..4 matching the 5-column child width.
  EXPECT_TRUE(make_identity_project(make_temp_read(), 5)->is_identity());

  // Column drop: fewer columns than the child -> not identity.
  EXPECT_FALSE(make_identity_project(make_temp_read(), 3)->is_identity());

  // Reordering: same width but references out of order -> not identity.
  {
    std::vector<std::unique_ptr<gqe::expression>> exprs;
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(2));
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(3));
    exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(4));
    EXPECT_FALSE(make_project(make_temp_read(), std::move(exprs))->is_identity());
  }
}
