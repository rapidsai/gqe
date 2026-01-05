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
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/utility.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/logger.hpp>

#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;
class StringToIntLiteralTest : public testing::TestWithParam<relation_t> {
 protected:
  StringToIntLiteralTest()
  {
    // Register the test table in the catalog
    _catalog                    = std::make_unique<gqe::catalog>(&_task_manager_ctx);
    _test_table_name            = "test_table";
    std::string column_name     = "a";
    cudf::data_type column_type = cudf::data_type(cudf::type_id::INT8);
    _catalog->register_table(_test_table_name,
                             {{column_name, column_type}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});

    // Initialize the optimizer
    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::string_to_int_literal}, {});
    optimizer =
      std::make_unique<gqe::optimizer::logical_optimizer>(&logical_rule_config, _catalog.get());
  }

  void construct_test_plan(relation_t rel_type) { test_plan = _construct_plan(false, rel_type); }
  void construct_ref_plan(relation_t rel_type) { ref_plan = _construct_plan(true, rel_type); }

  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
  std::shared_ptr<gqe::logical::relation> test_plan;
  std::unique_ptr<gqe::logical::relation> ref_plan;

 private:
  /*
   * Constructs a plan with particular relation as the head
   */
  std::unique_ptr<gqe::logical::relation> _construct_plan(bool optimized, relation_t rel_type)
  {
    // Hand coded logical plan for testing
    auto column_reference_expr = std::make_shared<gqe::column_reference_expression>(0);
    auto literal_int_expr      = std::make_shared<gqe::literal_expression<int8_t>>(82);
    auto literal_string_expr   = std::make_shared<gqe::literal_expression<std::string>>("R");
    auto comparison_expr =
      std::make_shared<gqe::less_expression>(column_reference_expr, literal_string_expr);
    std::unique_ptr<gqe::expression> inner_expr;
    if (optimized) {
      inner_expr = std::make_unique<gqe::equal_expression>(column_reference_expr, literal_int_expr);
    } else {
      inner_expr =
        std::make_unique<gqe::equal_expression>(column_reference_expr, literal_string_expr);
    }

    std::vector<std::string> column_names     = {"a"};
    std::vector<cudf::data_type> column_types = {cudf::data_type(cudf::type_id::INT8)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
    auto read_rel = std::make_shared<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, _test_table_name, nullptr);

    switch (rel_type) {
      case relation_t::aggregate: {
        // Note that the inner expression does not make sense as a key or measure, and is only used
        // for testing purposes.
        std::vector<std::unique_ptr<gqe::expression>> keys;
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
        keys.push_back(inner_expr->clone());
        measures.push_back(
          std::make_pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>(
            cudf::aggregation::Kind::COUNT_ALL, std::move(inner_expr)));
        return std::make_unique<gqe::logical::aggregate_relation>(
          std::move(read_rel), std::move(subquery_relations), std::move(keys), std::move(measures));
      }
      case relation_t::filter: {
        std::vector<cudf::size_type> projection_indices = {0};
        return std::make_unique<gqe::logical::filter_relation>(std::move(read_rel),
                                                               std::move(subquery_relations),
                                                               std::move(inner_expr),
                                                               std::move(projection_indices));
      }
      case relation_t::join: {
        // Note that the inner expression does not make sense as a join condition, and is only used
        // for testing purposes.
        std::vector<cudf::size_type> projection_indices = {0};
        return std::make_unique<gqe::logical::join_relation>(read_rel,
                                                             read_rel,
                                                             std::move(subquery_relations),
                                                             std::move(inner_expr),
                                                             gqe::join_type_type::inner,
                                                             std::move(projection_indices));
      }
      case relation_t::project: {
        std::vector<std::unique_ptr<gqe::expression>> select_exprs;
        select_exprs.push_back(std::move(inner_expr));
        return std::make_unique<gqe::logical::project_relation>(
          std::move(read_rel), std::move(subquery_relations), std::move(select_exprs));
      }
      case relation_t::read: {
        // Note that the inner expression does not make sense as a partial filter, and is only used
        // for testing purposes.
        return std::make_unique<gqe::logical::read_relation>(std::move(subquery_relations),
                                                             column_names,
                                                             column_types,
                                                             "test_table",
                                                             std::move(inner_expr));
      }
      case relation_t::sort: {
        std::vector<std::unique_ptr<gqe::expression>> key_exprs;
        std::vector<cudf::order> column_orders         = {cudf::order::ASCENDING};
        std::vector<cudf::null_order> null_precedences = {cudf::null_order::AFTER};
        key_exprs.push_back(std::move(inner_expr));
        return std::make_unique<gqe::logical::sort_relation>(std::move(read_rel),
                                                             std::move(subquery_relations),
                                                             std::move(column_orders),
                                                             std::move(null_precedences),
                                                             std::move(key_exprs));
      }
      case relation_t::window: {
        std::vector<std::unique_ptr<gqe::expression>> args;
        std::vector<std::unique_ptr<gqe::expression>> order_bys;
        std::vector<std::unique_ptr<gqe::expression>> partition_bys;
        std::vector<cudf::order> orders = {cudf::order::ASCENDING};
        args.push_back(inner_expr->clone());
        order_bys.push_back(inner_expr->clone());
        partition_bys.push_back(std::move(inner_expr));
        return std::make_unique<gqe::logical::window_relation>(
          std::move(read_rel),
          std::move(subquery_relations),
          cudf::aggregation::Kind::COUNT_ALL,
          std::move(args),
          std::move(order_bys),
          std::move(partition_bys),
          std::move(orders),
          gqe::window_frame_bound::unbounded(),
          gqe::window_frame_bound::unbounded());
      }
      default: throw std::runtime_error("unsupported relation type");
    }
  }

  gqe::task_manager_context _task_manager_ctx;
  std::string _test_table_name;
  std::unique_ptr<gqe::catalog> _catalog;
};

TEST_P(StringToIntLiteralTest, TestTypes)
{
  // Construct test and ref plans
  relation_t rel_type = GetParam();
  construct_test_plan(rel_type);
  construct_ref_plan(rel_type);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

INSTANTIATE_TEST_SUITE_P(Relations,
                         StringToIntLiteralTest,
                         testing::Values(relation_t::aggregate,
                                         relation_t::project,
                                         relation_t::filter,
                                         relation_t::join,
                                         relation_t::read,
                                         relation_t::sort,
                                         relation_t::window));

// Test that the partial filter of a read relation is evaluated on the base table schema and not the
// projected columns. Construct a read relation with the following properties:
// - Schema of base table has 3 columns with indexes 0, 1, 2 (counting from zero).
// - The read relation projects a single column
// - The partial filter is evaluated on column 2 (counting from zero).
// This ensures that the column index of the partial filter is bigger than the number of
// projected columns (2 > 1).
TEST(StringToIntLiteralTest, transformPartialFilterInReadRelation)
{
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::string_to_int_literal}, {});
  gqe::task_manager_context task_manager_ctx;
  gqe::catalog catalog{&task_manager_ctx};
  auto optimizer =
    std::make_unique<gqe::optimizer::logical_optimizer>(&logical_rule_config, &catalog);

  // Input expression has a single-char string literal
  constexpr cudf::size_type column_idx = 2;
  const std::string stringLiteral      = "R";
  constexpr int8_t int8Literal         = 82;  // Corresponds to "R"
  auto partial_filter                  = std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::column_reference_expression>(column_idx),
    std::make_shared<gqe::literal_expression<std::string>>(stringLiteral));

  // Expected expression replaces single-char string literal with int8
  auto expected = std::make_unique<gqe::equal_expression>(
    std::make_shared<gqe::column_reference_expression>(column_idx),
    std::make_shared<gqe::literal_expression<std::int8_t>>(int8Literal));

  // Register the base table
  const std::string table_name                  = "table";
  const std::vector<gqe::column_traits> columns = {{"col1", cudf::data_type(cudf::type_id::INT8)},
                                                   {"col2", cudf::data_type(cudf::type_id::INT8)},
                                                   {"col3", cudf::data_type(cudf::type_id::INT8)}};
  catalog.register_table(
    table_name, columns, gqe::storage_kind::system_memory{}, gqe::partitioning_schema_kind::none{});

  // Create read relation
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  std::vector<std::string> projected_column_names     = {"a"};
  std::vector<cudf::data_type> projected_column_types = {cudf::data_type(cudf::type_id::INT8)};
  auto read_relation = std::make_shared<gqe::logical::read_relation>(subquery_relations,
                                                                     projected_column_names,
                                                                     projected_column_types,
                                                                     table_name,
                                                                     std::move(partial_filter));

  // Optimize the read relation
  auto optimized_relation =
    dynamic_cast<gqe::logical::read_relation*>(optimizer->optimize(read_relation).get());

  // Check that partial filter has been replaced
  auto actual = optimized_relation->partial_filter_unsafe();
  EXPECT_EQ(*expected, *actual);
}
