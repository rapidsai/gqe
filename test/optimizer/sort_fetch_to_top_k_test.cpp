/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/top_k.hpp>
#include <gqe/optimizer/estimator.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/optimizer/relation_traits.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/task_manager_context.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using logical_type  = gqe::logical::relation::relation_type;
using physical_type = gqe::physical::relation::relation_type;
using rule_type     = gqe::optimizer::logical_optimization_rule_type;

/**
 * @brief Test fixture for the `sort_fetch_to_top_k` optimizer rule.
 *
 * Configures a single-rule optimizer pipeline over a three-column INT32 table `t`.
 */
class SortFetchToTopKTest : public ::testing::Test {
 protected:
  SortFetchToTopKTest()
  {
    task_manager_context = std::make_unique<gqe::task_manager_context>();
    catalog              = std::make_unique<gqe::catalog>(task_manager_context.get());
    catalog->register_table("t",
                            {{"a", cudf::data_type(cudf::type_id::INT32)},
                             {"b", cudf::data_type(cudf::type_id::INT32)},
                             {"c", cudf::data_type(cudf::type_id::INT32)}},
                            gqe::storage_kind::system_memory{},
                            gqe::partitioning_schema_kind::none{});

    config    = gqe::optimizer::optimization_configuration({rule_type::sort_fetch_to_top_k}, {});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&config, catalog.get());
  }

  /**
   * @brief Build a logical read of columns {a, b, c} from table `t`.
   *
   * @return A read relation over table `t`
   */
  [[nodiscard]] std::shared_ptr<gqe::logical::read_relation> make_read() const
  {
    return std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::vector<std::string>{"a", "b", "c"},
      std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT32),
                                   cudf::data_type(cudf::type_id::INT32),
                                   cudf::data_type(cudf::type_id::INT32)},
      "t",
      nullptr);
  }

  /**
   * @brief Build a fetch over a descending sort with the specified keys and bounds.
   *
   * @param keys Expressions used to order the input rows
   * @param offset Number of ordered rows to skip
   * @param count Maximum number of rows to return after `offset`
   * @return A fetch relation over a sort relation
   */
  [[nodiscard]] std::shared_ptr<gqe::logical::fetch_relation> make_sort_fetch(
    std::vector<std::unique_ptr<gqe::expression>> keys,
    int64_t offset = 3,
    int64_t count  = 20) const
  {
    std::vector<cudf::order> orders(keys.size(), cudf::order::DESCENDING);
    std::vector<cudf::null_order> null_orders(keys.size(), cudf::null_order::BEFORE);
    auto sort = std::make_shared<gqe::logical::sort_relation>(
      make_read(),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(orders),
      std::move(null_orders),
      std::move(keys));
    return std::make_shared<gqe::logical::fetch_relation>(std::move(sort), offset, count);
  }

  /**
   * @brief Build column-reference expressions for the specified column indices.
   *
   * @param indices Zero-based column indices
   * @return Column-reference expressions in the same order as `indices`
   */
  [[nodiscard]] static std::vector<std::unique_ptr<gqe::expression>> column_keys(
    std::initializer_list<cudf::size_type> indices)
  {
    std::vector<std::unique_ptr<gqe::expression>> result;
    for (auto index : indices) {
      result.push_back(std::make_unique<gqe::column_reference_expression>(index));
    }
    return result;
  }

  std::unique_ptr<gqe::task_manager_context> task_manager_context;
  std::unique_ptr<gqe::catalog> catalog;
  gqe::optimizer::optimization_configuration config;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

TEST_F(SortFetchToTopKTest, RewritesRootAndLowersToPhysicalTopK)
{
  auto optimized = optimizer->optimize(make_sort_fetch(column_keys({1})));

  ASSERT_EQ(optimized->type(), logical_type::top_k);
  auto const* logical_top_k = static_cast<gqe::logical::top_k_relation const*>(optimized.get());
  auto const* key =
    dynamic_cast<gqe::column_reference_expression const*>(logical_top_k->key_unsafe());
  ASSERT_NE(key, nullptr);
  EXPECT_EQ(key->column_idx(), 1);
  EXPECT_EQ(logical_top_k->column_order(), cudf::order::DESCENDING);
  EXPECT_EQ(logical_top_k->null_precedence(), cudf::null_order::BEFORE);
  EXPECT_EQ(logical_top_k->offset(), 3);
  EXPECT_EQ(logical_top_k->count(), 20);
  EXPECT_EQ(gqe::estimator(catalog.get())(logical_top_k).num_rows, 20);
  EXPECT_EQ(logical_top_k->children_unsafe().front()->type(), logical_type::read);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 1u);

  gqe::physical_plan_builder builder(catalog.get());
  auto physical = builder.build(optimized.get());
  ASSERT_EQ(physical->type(), physical_type::top_k);
  auto const* physical_top_k = static_cast<gqe::physical::top_k_relation const*>(physical.get());
  EXPECT_EQ(physical_top_k->key_index(), 1);
  EXPECT_EQ(physical_top_k->offset(), 3);
  EXPECT_EQ(physical_top_k->count(), 20);
}

/**
 * @brief Rewrite an eligible Fetch-Sort below another relation.
 */
TEST_F(SortFetchToTopKTest, RewritesNestedPattern)
{
  std::vector<std::unique_ptr<gqe::expression>> outputs;
  outputs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  auto root = std::make_shared<gqe::logical::project_relation>(
    make_sort_fetch(column_keys({0})),
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(outputs));

  auto optimized = optimizer->optimize(root);

  ASSERT_EQ(optimized->type(), logical_type::project);
  EXPECT_EQ(optimized->children_unsafe().front()->type(), logical_type::top_k);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 1u);
}

/**
 * @brief Rewrite an eligible Fetch-Sort at the root of a subquery plan.
 */
TEST_F(SortFetchToTopKTest, RewritesSubqueryPattern)
{
  std::vector<std::unique_ptr<gqe::expression>> outputs;
  outputs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  auto root = std::make_shared<gqe::logical::project_relation>(
    make_read(),
    std::vector<std::shared_ptr<gqe::logical::relation>>{make_sort_fetch(column_keys({0}))},
    std::move(outputs));

  auto optimized = optimizer->optimize(root);

  ASSERT_EQ(optimized->subqueries_unsafe().size(), 1u);
  EXPECT_EQ(optimized->subqueries_unsafe().front()->type(), logical_type::top_k);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 1u);
}

/**
 * @brief Reuse one Top-K replacement for direct and nested edges to a shared Fetch node.
 */
TEST_F(SortFetchToTopKTest, ReusesReplacementForSharedFetch)
{
  auto shared_fetch = make_sort_fetch(column_keys({0}));
  std::vector<std::unique_ptr<gqe::expression>> outputs;
  outputs.push_back(std::make_unique<gqe::column_reference_expression>(0));
  outputs.push_back(std::make_unique<gqe::column_reference_expression>(1));
  outputs.push_back(std::make_unique<gqe::column_reference_expression>(2));
  auto nested_reference = std::make_shared<gqe::logical::project_relation>(
    shared_fetch, std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outputs));
  auto root = std::make_shared<gqe::logical::set_relation>(
    shared_fetch, std::move(nested_reference), gqe::logical::set_relation::set_union_all);

  auto optimized = optimizer->optimize(root);

  auto const children = optimized->children_safe();
  ASSERT_EQ(children.size(), 2u);
  ASSERT_EQ(children[0]->type(), logical_type::top_k);
  ASSERT_EQ(children[1]->type(), logical_type::project);
  EXPECT_EQ(children[0], children[1]->children_safe().front());
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 1u);
}

/**
 * @brief Repeat the visitor pass until independent Fetch-Sort patterns are rewritten.
 */
TEST_F(SortFetchToTopKTest, RewritesMultiplePatterns)
{
  auto root =
    std::make_shared<gqe::logical::set_relation>(make_sort_fetch(column_keys({0})),
                                                 make_sort_fetch(column_keys({1})),
                                                 gqe::logical::set_relation::set_union_all);

  auto optimized = optimizer->optimize(root);

  auto const children = optimized->children_safe();
  ASSERT_EQ(children.size(), 2u);
  EXPECT_EQ(children[0]->type(), logical_type::top_k);
  EXPECT_EQ(children[1]->type(), logical_type::top_k);
  EXPECT_NE(children[0], children[1]);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 1u);
}

TEST_F(SortFetchToTopKTest, LeavesMultiKeySortUnchanged)
{
  auto root      = make_sort_fetch(column_keys({0, 1}));
  auto optimized = optimizer->optimize(root);

  EXPECT_EQ(optimized.get(), root.get());
  EXPECT_EQ(optimized->type(), logical_type::fetch);
  EXPECT_EQ(optimized->children_unsafe().front()->type(), logical_type::sort);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 0u);
}

/**
 * @brief Leave a computed-key Sort and its Fetch unchanged.
 */
TEST_F(SortFetchToTopKTest, LeavesComplexKeySortUnchanged)
{
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));
  auto root      = make_sort_fetch(std::move(keys));
  auto optimized = optimizer->optimize(root);

  EXPECT_EQ(optimized, root);
  EXPECT_EQ(optimized->type(), logical_type::fetch);
  EXPECT_EQ(optimized->children_unsafe().front()->type(), logical_type::sort);
  EXPECT_EQ(optimizer->get_rule_count(rule_type::sort_fetch_to_top_k), 0u);
}

TEST_F(SortFetchToTopKTest, RejectsInvalidTopKParameters)
{
  auto make_top_k = [](std::shared_ptr<gqe::logical::relation> input,
                       std::unique_ptr<gqe::expression> key,
                       int64_t offset,
                       int64_t count) {
    return std::make_shared<gqe::logical::top_k_relation>(
      std::move(input),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(key),
      cudf::order::DESCENDING,
      cudf::null_order::BEFORE,
      offset,
      count);
  };

  EXPECT_THROW(make_top_k(nullptr, std::make_unique<gqe::column_reference_expression>(0), 0, 1),
               std::invalid_argument);
  EXPECT_THROW(make_top_k(make_read(), nullptr, 0, 1), std::invalid_argument);
  EXPECT_THROW(
    make_top_k(make_read(), std::make_unique<gqe::column_reference_expression>(0), -1, 1),
    std::invalid_argument);
  EXPECT_THROW(
    make_top_k(make_read(), std::make_unique<gqe::column_reference_expression>(0), 0, -1),
    std::invalid_argument);
}

TEST_F(SortFetchToTopKTest, PreservesRelationTraits)
{
  auto root = make_sort_fetch(column_keys({0}));
  gqe::optimizer::relation_properties properties;
  properties.add_unique_key({0});
  root->set_relation_traits(
    std::make_unique<gqe::optimizer::relation_traits>(std::move(properties)));

  auto optimized = optimizer->optimize(root);

  ASSERT_EQ(optimized->type(), logical_type::top_k);
  EXPECT_EQ(optimized->relation_traits(), root->relation_traits());
}

}  // namespace
