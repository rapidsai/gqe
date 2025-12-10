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

#include "../utility.hpp"

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

typedef ::testing::Types<gqe::logical::filter_relation, gqe::logical::join_relation> ChildTypes;

template <typename T>
class ProjectionPushdown : public ::testing::Test {
 protected:
  enum class test_type {
    no_opt,
    col_ref_only,
    col_ref_with_reordering,
    mixed_expression,
    transformation_direction_up
  };

  void initialize_optimizer(gqe::optimizer::optimization_configuration rule_config)
  {
    // Create tables
    constexpr cudf::size_type num_rows = 100;

    std::vector<std::unique_ptr<cudf::column>> columns_ints;
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
    columns_ints.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));

    auto table_ints = std::make_unique<cudf::table>(std::move(columns_ints));

    // Write tables
    auto file_dir_temp = temp_env->get_temp_dir() + "temp";
    std::filesystem::create_directory(file_dir_temp);
    write_table_to_file(table_ints->view(),
                        get_column_names(table_ints->num_columns()),
                        file_dir_temp + "/table.parquet");

    catalog.register_table(
      "temp_table",
      {{"column_0", cudf::data_type(cudf::type_id::INT64)},
       {"column_1", cudf::data_type(cudf::type_id::INT64)},
       {"column_2", cudf::data_type(cudf::type_id::INT64)},
       {"column_3", cudf::data_type(cudf::type_id::INT64)},
       {"column_4", cudf::data_type(cudf::type_id::INT64)}},
      gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(file_dir_temp)},
      gqe::partitioning_schema_kind::automatic{});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&rule_config, &catalog);
  }

  std::shared_ptr<gqe::logical::relation> construct_test_plan(test_type test)
  {
    return _construct_plan(false, test);
  }

  std::shared_ptr<gqe::logical::relation> construct_optimized_plan(test_type test)
  {
    return _construct_plan(true, test);
  }

  void projection_pushdown_helper(test_type test)
  {
    gqe::optimizer::optimization_configuration logical_rule_config(
      {gqe::optimizer::logical_optimization_rule_type::projection_pushdown}, {});

    initialize_optimizer(logical_rule_config);

    // Construct test and ref plans
    auto test_plan = construct_test_plan(test);

    // Optimize
    assert(optimizer);

    auto optimized_plan = optimizer->optimize(test_plan);

    // Test
    if (test == test_type::no_opt) {
      EXPECT_EQ(optimizer->get_rule_count(
                  gqe::optimizer::logical_optimization_rule_type::projection_pushdown),
                0);
      EXPECT_EQ(*test_plan, *optimized_plan);
    } else {
      auto ref_plan = construct_optimized_plan(test);
      EXPECT_EQ(optimizer->get_rule_count(
                  gqe::optimizer::logical_optimization_rule_type::projection_pushdown),
                1);
      EXPECT_EQ(*ref_plan, *optimized_plan);
    }
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::catalog catalog{&task_manager_ctx};
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;

 private:
  /*
    Constructs and returns a unique pointer to a filter_relation.
    The filter applies a 'Column(1) >= Column(2)' condition on the 'table' and outputs only the
    specified projection indices.
  */
  template <typename U                                                                       = T,
            typename std::enable_if_t<std::is_same_v<U, gqe::logical::filter_relation>, int> = 0>
  std::unique_ptr<U> _construct_child(std::shared_ptr<gqe::logical::read_relation> table,
                                      std::vector<cudf::size_type> projection_indices)
  {
    return std::make_unique<U>(std::move(table),
                               std::vector<std::shared_ptr<gqe::logical::relation>>(),
                               std::make_unique<gqe::greater_equal_expression>(
                                 std::make_shared<gqe::column_reference_expression>(1),
                                 std::make_shared<gqe::column_reference_expression>(2)),
                               projection_indices);
  }

  /*
    Constructs and returns a unique pointer to a join_relation.
    This performs an inner join between 'table' and 'table', with the condition 'Column(0) ==
    Column(4)' and outputs only the specified projection indices.
  */
  template <typename U                                                                     = T,
            typename std::enable_if_t<std::is_same_v<U, gqe::logical::join_relation>, int> = 0>
  std::unique_ptr<U> _construct_child(std::shared_ptr<gqe::logical::read_relation> temp_table,
                                      std::vector<cudf::size_type> projection_indices)
  {
    return std::make_unique<U>(temp_table,
                               temp_table,
                               std::vector<std::shared_ptr<gqe::logical::relation>>(),
                               std::make_unique<gqe::equal_expression>(
                                 std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(4)),
                               gqe::join_type_type::inner,
                               projection_indices);
  }

  std::unique_ptr<gqe::logical::relation> _construct_plan(bool optimized, test_type test)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names_temp = {
      "column_0", "column_1", "column_2", "column_3", "column_4"};
    auto column_types_temp = {cudf::data_type(cudf::type_id::INT64),
                              cudf::data_type(cudf::type_id::INT64),
                              cudf::data_type(cudf::type_id::INT64),
                              cudf::data_type(cudf::type_id::INT64),
                              cudf::data_type(cudf::type_id::INT64)};
    auto temp_table        = std::make_shared<gqe::logical::read_relation>(
      std::vector<std::shared_ptr<gqe::logical::relation>>(),
      column_names_temp,
      column_types_temp,
      "temp_table",
      nullptr);

    if (test == test_type::no_opt) {
      auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 1, 2}));
      std::vector<std::unique_ptr<gqe::expression>> project_exprs;

      project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

      project_exprs.emplace_back(std::make_unique<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<int64_t>>(2)));

      project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

      auto project = std::make_unique<gqe::logical::project_relation>(
        std::move(child),
        std::vector<std::shared_ptr<gqe::logical::relation>>(),
        std::move(project_exprs));
      return project;
    }

    if (test == test_type::col_ref_only) {
      if (!optimized) {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 1, 2}));
        std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
        col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(child),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(col_0_exprs));

        return project;
      } else {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0}));
        return child;
      }
    }

    if (test == test_type::col_ref_with_reordering) {
      if (!optimized) {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 1, 2}));
        std::vector<std::unique_ptr<gqe::expression>> col_exprs;
        col_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));
        col_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(child),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(col_exprs));

        return project;
      } else {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({1, 0}));
        return child;
      }
    }

    if (test == test_type::mixed_expression) {
      if (!optimized) {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 2, 3, 4}));
        std::vector<std::unique_ptr<gqe::expression>> project_exprs;

        project_exprs.emplace_back(std::make_unique<gqe::if_then_else_expression>(
          std::make_shared<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(2),
            std::make_shared<gqe::literal_expression<int64_t>>(25)),
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(3)));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(child),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_exprs));
        return project;
      } else {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 3, 4}));
        std::vector<std::unique_ptr<gqe::expression>> project_exprs;

        project_exprs.emplace_back(std::make_unique<gqe::if_then_else_expression>(
          std::make_shared<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::literal_expression<int64_t>>(25)),
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(2)));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(child),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_exprs));
        return project;
      }
    }

    /* P (CRs only)     P (CRs only)
       |                |
       P (CRs only)  -> F          ->  F
       |
       F
    */
    if (test == test_type::transformation_direction_up) {
      if (!optimized) {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0, 1, 2}));

        std::vector<std::unique_ptr<gqe::expression>> project_0_exprs;
        project_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
        project_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

        std::vector<std::unique_ptr<gqe::expression>> project_exprs;
        project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project_0 = std::make_unique<gqe::logical::project_relation>(
          std::move(child),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_0_exprs));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(project_0),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_exprs));

        return project;
      }

      else {
        auto child = _construct_child(temp_table, std::vector<cudf::size_type>({0}));

        return child;
      }
    }

    else {
      throw std::invalid_argument("Unknown test type");
    }
  }
};

TYPED_TEST_SUITE(ProjectionPushdown, ChildTypes);

TYPED_TEST(ProjectionPushdown, ColRefOnly)
{
  auto test_type = ProjectionPushdown<TypeParam>::test_type::col_ref_only;

  this->projection_pushdown_helper(test_type);
}

TYPED_TEST(ProjectionPushdown, ColRefWithReordering)
{
  auto test_type = ProjectionPushdown<TypeParam>::test_type::col_ref_with_reordering;

  this->projection_pushdown_helper(test_type);
}

TYPED_TEST(ProjectionPushdown, MixedExpression)
{
  auto test_type = ProjectionPushdown<TypeParam>::test_type::mixed_expression;

  this->projection_pushdown_helper(test_type);
}

TYPED_TEST(ProjectionPushdown, TransformationDirectionUp)
{
  auto test_type = ProjectionPushdown<TypeParam>::test_type::transformation_direction_up;

  this->projection_pushdown_helper(test_type);
}

TYPED_TEST(ProjectionPushdown, NoOpt)
{
  auto test_type = ProjectionPushdown<TypeParam>::test_type::no_opt;

  this->projection_pushdown_helper(test_type);
}
