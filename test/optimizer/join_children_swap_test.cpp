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

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf_test/base_fixture.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class JoinChildrenSwapTest : public ::testing::Test {
 protected:
  void initialize_optimizer(gqe::optimizer::optimization_configuration rule_config)
  {
    // Create tables
    constexpr cudf::size_type num_rows        = 100;
    constexpr cudf::size_type num_files_large = 10;

    std::vector<std::unique_ptr<cudf::column>> columns_ints;
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.1, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    auto table_ints = std::make_unique<cudf::table>(std::move(columns_ints));

    std::vector<std::unique_ptr<cudf::column>> columns_mixed;
    columns_mixed.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.1, -30, 30));
    columns_mixed.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_mixed.push_back(generate_fixed_width_column<double>(num_rows, 0.01, -30, 30));
    auto table_mixed = std::make_unique<cudf::table>(std::move(columns_mixed));

    // Write tables
    auto file_dir_small = temp_env->get_temp_dir() + "small";
    std::filesystem::create_directory(file_dir_small);
    write_table_to_file(table_ints->view(),
                        get_column_names(table_ints->num_columns()),
                        file_dir_small + "/table.parquet");

    auto file_dir_large = temp_env->get_temp_dir() + "large";
    std::filesystem::create_directory(file_dir_large);
    for (cudf::size_type file_number = 0; file_number < num_files_large; file_number++) {
      write_table_to_file(table_mixed->view(),
                          get_column_names(table_mixed->num_columns()),
                          file_dir_large + "/table" + std::to_string(file_number) + ".parquet");
    }

    catalog.register_table(
      "big_table",
      {{"column_0", cudf::data_type(cudf::type_id::INT64)},
       {"column_1", cudf::data_type(cudf::type_id::INT64)},
       {"column_2", cudf::data_type(cudf::type_id::FLOAT64)}},
      gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(file_dir_large)},
      gqe::partitioning_schema_kind::automatic{});
    catalog.register_table(
      "small_table",
      {{"column_0", cudf::data_type(cudf::type_id::INT64)},
       {"column_1", cudf::data_type(cudf::type_id::INT64)},
       {"column_2", cudf::data_type(cudf::type_id::INT64)}},
      gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(file_dir_small)},
      gqe::partitioning_schema_kind::automatic{});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&rule_config, &catalog);
  }

  std::shared_ptr<gqe::logical::relation> construct_test_plan(bool colref_only, bool nested)
  {
    return _construct_plan(false, colref_only, nested);
  }

  std::shared_ptr<gqe::logical::relation> construct_optimized_plan(bool colref_only, bool nested)
  {
    return _construct_plan(true, colref_only, nested);
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::catalog catalog{&task_manager_ctx};
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;

 private:
  std::unique_ptr<gqe::logical::relation> _construct_plan(bool optimized,
                                                          bool colref_only,
                                                          bool nested)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names_small = {"column_0", "column_1"};
    auto column_types_small                     = {cudf::data_type(cudf::type_id::INT64),
                                                   cudf::data_type(cudf::type_id::INT64)};
    auto small_table                            = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      column_names_small,
      column_types_small,
      "small_table",
      nullptr);
    std::vector<std::string> column_names_large = {"column_0", "column_2"};
    auto data_types_large                       = {cudf::data_type(cudf::type_id::INT64),
                                                   cudf::data_type(cudf::type_id::FLOAT64)};
    auto big_table                              = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      column_names_large,
      data_types_large,
      "big_table",
      nullptr);

    std::unique_ptr<gqe::logical::join_relation> join0;
    std::unique_ptr<gqe::logical::join_relation> join1;

    if (optimized) {
      std::shared_ptr<gqe::expression> condition_lhs =
        std::make_shared<gqe::column_reference_expression>(0);
      if (!colref_only) {
        condition_lhs = std::make_shared<gqe::add_expression>(
          condition_lhs, std::make_shared<gqe::literal_expression<int32_t>>(1));
      }
      // Always put the smaller relation on the right
      join0 = std::make_unique<gqe::logical::join_relation>(
        std::move(big_table),
        small_table,
        std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
        std::make_unique<gqe::equal_expression>(
          condition_lhs, std::make_shared<gqe::column_reference_expression>(2)),
        gqe::join_type_type::inner,
        std::vector<cudf::size_type>({2, 3, 1}));  // from {0, 1, 3}
      if (nested) {
        join1 = std::make_unique<gqe::logical::join_relation>(
          std::move(join0),
          small_table,
          std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
          std::make_unique<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(3)),
          gqe::join_type_type::inner,
          std::vector<cudf::size_type>({4, 0, 2}));
        return join1;
      }
    } else {
      std::shared_ptr<gqe::expression> condition_rhs =
        std::make_shared<gqe::column_reference_expression>(2);
      if (!colref_only) {
        condition_rhs = std::make_shared<gqe::add_expression>(
          condition_rhs, std::make_shared<gqe::literal_expression<int32_t>>(1));
      }
      // Put smaller relation on the left
      join0 = std::make_unique<gqe::logical::join_relation>(
        small_table,
        std::move(big_table),
        std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
        std::make_unique<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(0), condition_rhs),
        gqe::join_type_type::inner,
        std::vector<cudf::size_type>({0, 1, 3}));  // {2, 3, 1} when swapped
      if (nested) {
        join1 = std::make_unique<gqe::logical::join_relation>(
          small_table,
          std::move(join0),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),  // subquery_relations
          std::make_unique<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(0),
            std::make_shared<gqe::column_reference_expression>(3)),
          gqe::join_type_type::inner,
          std::vector<cudf::size_type>({1, 2, 4}));
        return join1;
      }
    }
    return join0;
  }
};

TEST_F(JoinChildrenSwapTest, Simple)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::join_children_swap}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  bool nested      = false;
  bool colref_only = true;
  auto test_plan   = construct_test_plan(colref_only, nested);
  auto ref_plan    = construct_optimized_plan(colref_only, nested);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(
    optimizer->get_rule_count(gqe::optimizer::logical_optimization_rule_type::join_children_swap),
    1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(JoinChildrenSwapTest, ConditionWithAdd)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::join_children_swap}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  bool nested      = false;
  bool colref_only = false;
  auto test_plan   = construct_test_plan(colref_only, nested);
  auto ref_plan    = construct_optimized_plan(colref_only, nested);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(
    optimizer->get_rule_count(gqe::optimizer::logical_optimization_rule_type::join_children_swap),
    1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(JoinChildrenSwapTest, Nested)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::join_children_swap}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  bool nested      = true;
  bool colref_only = true;
  auto test_plan   = construct_test_plan(colref_only, nested);
  auto ref_plan    = construct_optimized_plan(colref_only, nested);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(
    optimizer->get_rule_count(gqe::optimizer::logical_optimization_rule_type::join_children_swap),
    1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(JoinChildrenSwapTest, Optimized)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::join_children_swap}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  bool nested      = false;
  bool colref_only = true;
  auto test_plan   = construct_optimized_plan(colref_only, nested);
  auto ref_plan    = construct_optimized_plan(colref_only, nested);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(
    optimizer->get_rule_count(gqe::optimizer::logical_optimization_rule_type::join_children_swap),
    0);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}
