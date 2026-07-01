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
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

class AggregatePerfectHashTest : public gqe::test::BaseFixture {
 protected:
  /**
   * @brief Register a table and build an optimizer running uniqueness_propagation then
   * aggregate_perfect_hash (same ordering requirement as join_unique_keys).
   *
   * @param cols        Column descriptors for the single registered table.
   * @param unique_keys Composite unique key-sets, each a list of column names.
   */
  void initialize_optimizer(std::vector<gqe::column_traits> const& cols,
                            std::vector<std::vector<std::string>> unique_keys = {})
  {
    catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());
    catalog->register_table("t",
                            cols,
                            gqe::storage_kind::system_memory{},
                            gqe::partitioning_schema_kind::none{},
                            unique_keys);

    gqe::optimizer::optimization_configuration cfg(
      {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
       gqe::optimizer::logical_optimization_rule_type::aggregate_perfect_hash},
      {});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&cfg, catalog.get());
  }

  /**
   * @brief Build a logical plan: read "t", group by one key column, sum a measure column.
   *
   * @param cols        Column descriptors (must match what was registered).
   * @param key_col_idx Index of the group-by key column.
   */
  std::shared_ptr<gqe::logical::relation> make_aggregate(
    std::vector<gqe::column_traits> const& cols, cudf::size_type key_col_idx)
  {
    std::vector<std::string> col_names;
    std::vector<cudf::data_type> col_dtypes;
    for (auto const& col : cols) {
      col_names.push_back(col.name);
      col_dtypes.push_back(col.data_type);
    }

    auto read = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>{}, col_names, col_dtypes, "t", nullptr);

    std::vector<std::unique_ptr<gqe::expression>> keys;
    keys.push_back(std::make_unique<gqe::column_reference_expression>(key_col_idx));

    // SUM of the first non-key column as the measure
    cudf::size_type measure_col_idx = (key_col_idx == 0) ? 1 : 0;
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
    measures.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(measure_col_idx));

    return std::make_shared<gqe::logical::aggregate_relation>(
      std::move(read),
      std::vector<std::shared_ptr<gqe::logical::relation>>{},
      std::move(keys),
      std::move(measures));
  }

  std::unique_ptr<gqe::catalog> catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
};

/** Unique, fixed-width group-by key → perfect hash enabled. */
TEST_F(AggregatePerfectHashTest, UniqueFixedWidthKeySetsFlag)
{
  std::vector<gqe::column_traits> cols = {{"id", cudf::data_type(cudf::type_id::INT64)},
                                          {"val", cudf::data_type(cudf::type_id::INT32)}};
  initialize_optimizer(cols, {{"id"}});

  auto optimized = optimizer->optimize(make_aggregate(cols, /*key_col_idx=*/0));
  auto* agg      = dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get());
  ASSERT_NE(agg, nullptr);
  EXPECT_TRUE(agg->is_perfect_hashable());
}

/** Non-unique group-by key → perfect hash disabled. */
TEST_F(AggregatePerfectHashTest, NonUniqueKeyNoFlag)
{
  std::vector<gqe::column_traits> cols = {{"category", cudf::data_type(cudf::type_id::INT32)},
                                          {"val", cudf::data_type(cudf::type_id::INT64)}};
  initialize_optimizer(cols, {});  // no unique keys registered

  auto optimized = optimizer->optimize(make_aggregate(cols, /*key_col_idx=*/0));
  auto* agg      = dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get());
  ASSERT_NE(agg, nullptr);
  EXPECT_FALSE(agg->is_perfect_hashable());
}

/** Unique key but non-fixed-width type (STRING) → perfect hash disabled. */
TEST_F(AggregatePerfectHashTest, UniqueStringKeyNoFlag)
{
  std::vector<gqe::column_traits> cols = {{"name", cudf::data_type(cudf::type_id::STRING)},
                                          {"val", cudf::data_type(cudf::type_id::INT32)}};
  initialize_optimizer(cols, {{"name"}});

  auto optimized = optimizer->optimize(make_aggregate(cols, /*key_col_idx=*/0));
  auto* agg      = dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get());
  ASSERT_NE(agg, nullptr);
  EXPECT_FALSE(agg->is_perfect_hashable());
}

/** Composite unique key (a, b) fully covered by GROUP BY a, b → perfect hash enabled. */
TEST_F(AggregatePerfectHashTest, CompositeUniqueKeyCoveredSetsFlag)
{
  std::vector<gqe::column_traits> cols = {{"a", cudf::data_type(cudf::type_id::INT32)},
                                          {"b", cudf::data_type(cudf::type_id::INT32)},
                                          {"val", cudf::data_type(cudf::type_id::INT64)}};
  initialize_optimizer(cols, {{"a", "b"}});

  auto read = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::vector<std::string>{"a", "b", "val"},
    std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT32),
                                 cudf::data_type(cudf::type_id::INT32),
                                 cudf::data_type(cudf::type_id::INT64)},
    "t",
    nullptr);

  std::vector<std::unique_ptr<gqe::expression>> keys_ab;
  keys_ab.push_back(std::make_unique<gqe::column_reference_expression>(0));  // a
  keys_ab.push_back(std::make_unique<gqe::column_reference_expression>(1));  // b

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(2));

  auto plan = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(read),
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(keys_ab),
    std::move(measures));

  auto optimized = optimizer->optimize(plan);
  auto* agg      = dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get());
  ASSERT_NE(agg, nullptr);
  EXPECT_TRUE(agg->is_perfect_hashable());
}

/**
 * Non-column-reference grouping key (`GROUP BY a + b`) → perfect hash disabled. Only column-
 * reference keys are tracked against the input's unique-key sets, so a lone `a + b` key
 * contributes no columns and the unique key `{a, b}` is not covered.
 */
TEST_F(AggregatePerfectHashTest, NonColumnReferenceKeyNoFlag)
{
  std::vector<gqe::column_traits> cols = {{"a", cudf::data_type(cudf::type_id::INT32)},
                                          {"b", cudf::data_type(cudf::type_id::INT32)},
                                          {"val", cudf::data_type(cudf::type_id::INT64)}};
  initialize_optimizer(cols, {{"a", "b"}});

  auto read = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::vector<std::string>{"a", "b", "val"},
    std::vector<cudf::data_type>{cudf::data_type(cudf::type_id::INT32),
                                 cudf::data_type(cudf::type_id::INT32),
                                 cudf::data_type(cudf::type_id::INT64)},
    "t",
    nullptr);

  // GROUP BY a + b — a single binary-op expression rather than plain column references.
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::column_reference_expression>(1)));

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::SUM,
                        std::make_unique<gqe::column_reference_expression>(2));

  auto plan = std::make_shared<gqe::logical::aggregate_relation>(
    std::move(read),
    std::vector<std::shared_ptr<gqe::logical::relation>>{},
    std::move(keys),
    std::move(measures));

  auto optimized = optimizer->optimize(plan);
  auto* agg      = dynamic_cast<gqe::logical::aggregate_relation*>(optimized.get());
  ASSERT_NE(agg, nullptr);
  EXPECT_FALSE(agg->is_perfect_hashable());
}
