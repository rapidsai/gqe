/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using relation_t = gqe::logical::relation::relation_type;

class RewriteRuleTest : public testing::TestWithParam<relation_t> {
 protected:
  RewriteRuleTest()
  {
    // Register the test table in the catalog
    _catalog                    = std::make_unique<gqe::catalog>(&_task_manager_ctx);
    _test_table_name            = "test_table";
    std::string column_name     = "a";
    cudf::data_type column_type = cudf::data_type(cudf::type_id::INT32);
    _catalog->register_table(_test_table_name,
                             {{column_name, column_type}},
                             gqe::storage_kind::system_memory{},
                             gqe::partitioning_schema_kind::none{});

    // Initialize the optimizer
    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::not_not_rewrite}, {});
    optimizer =
      std::make_unique<gqe::optimizer::logical_optimizer>(&logical_rule_config, _catalog.get());
  }

  void construct_test_plan(relation_t rel_type) { test_plan = _construct_plan(false, rel_type); }
  void construct_ref_plan(relation_t rel_type) { ref_plan = _construct_plan(true, rel_type); }

  void construct_test_plan() { test_plan = _construct_plan(false); }
  void construct_ref_plan() { ref_plan = _construct_plan(true); }

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
    auto literal_one_expr = std::make_shared<gqe::literal_expression<int32_t>>(1);
    auto literal_two_expr = std::make_shared<gqe::literal_expression<int32_t>>(2);
    auto comparison_expr =
      std::make_shared<gqe::less_expression>(literal_one_expr, literal_two_expr);
    std::unique_ptr<gqe::expression> inner_expr;
    if (optimized) {
      inner_expr = comparison_expr->clone();  // not-not removed
    } else {
      inner_expr = std::make_unique<gqe::not_expression>(
        std::make_shared<gqe::not_expression>(comparison_expr));  // not-not
    }
    std::vector<std::string> column_names = {"a"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_shared<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table", nullptr);

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

  /*
   * Constructs a general plan:
   *       J
   *      / \
   *     J   P
   *     /\  |
   *    P  P R
   *    |  |
   *    R  R
   */
  std::unique_ptr<gqe::logical::relation> _construct_plan(bool optimized)
  {
    // Hand coded logical plan for testing
    auto literal_one_expr = std::make_shared<gqe::literal_expression<int32_t>>(1);
    auto literal_two_expr = std::make_shared<gqe::literal_expression<int32_t>>(2);
    auto comparison_expr =
      std::make_shared<gqe::less_expression>(literal_one_expr, literal_two_expr);
    std::unique_ptr<gqe::expression> inner_expr;
    if (optimized) {
      inner_expr = comparison_expr->clone();  // not-not removed
    } else {
      inner_expr = std::make_unique<gqe::not_expression>(
        std::make_shared<gqe::not_expression>(comparison_expr));  // not-not
    }
    std::vector<std::string> column_names = {"a"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_shared<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table", nullptr);

    std::vector<std::unique_ptr<gqe::expression>> select_exprs_1;
    select_exprs_1.push_back(inner_expr->clone());
    std::vector<std::unique_ptr<gqe::expression>> select_exprs_2;
    select_exprs_2.push_back(inner_expr->clone());
    std::vector<std::unique_ptr<gqe::expression>> select_exprs_3;
    select_exprs_3.push_back(inner_expr->clone());
    auto project_1 = std::make_shared<gqe::logical::project_relation>(
      read_rel, subquery_relations, std::move(select_exprs_1));
    auto project_2 = std::make_shared<gqe::logical::project_relation>(
      read_rel, subquery_relations, std::move(select_exprs_2));
    auto project_3 = std::make_shared<gqe::logical::project_relation>(
      read_rel, subquery_relations, std::move(select_exprs_3));

    std::vector<cudf::size_type> projection_indices = {0};
    auto join_1 = std::make_shared<gqe::logical::join_relation>(project_1,
                                                                project_2,
                                                                subquery_relations,
                                                                inner_expr->clone(),
                                                                gqe::join_type_type::inner,
                                                                projection_indices);
    auto join_2 = std::make_unique<gqe::logical::join_relation>(join_1,
                                                                project_3,
                                                                subquery_relations,
                                                                inner_expr->clone(),
                                                                gqe::join_type_type::inner,
                                                                projection_indices);

    return join_2;
  }

  gqe::task_manager_context _task_manager_ctx;
  std::string _test_table_name;
  std::unique_ptr<gqe::catalog> _catalog;
};

TEST_P(RewriteRuleTest, TestTypes)
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

INSTANTIATE_TEST_SUITE_P(NotNotRelations,
                         RewriteRuleTest,
                         testing::Values(relation_t::aggregate,
                                         relation_t::project,
                                         relation_t::filter,
                                         relation_t::join,
                                         relation_t::read,
                                         relation_t::sort,
                                         relation_t::window));

TEST_F(RewriteRuleTest, NotNotGeneral)
{
  // Construct test and ref plans
  construct_test_plan();
  construct_ref_plan();

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}
