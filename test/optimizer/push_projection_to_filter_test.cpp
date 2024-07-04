/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

class PushProjectionToFilter : public ::testing::Test {
 public:
  enum class test_type {
    no_opt,
    col_ref_only,
    col_ref_with_reordering,
    mixed_expression,
    transformation_direction_up
  };

 protected:
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

  gqe::catalog catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;

 private:
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
      auto filter = std::make_unique<gqe::logical::filter_relation>(
        std::move(temp_table),
        std::vector<std::shared_ptr<gqe::logical::relation>>(),
        std::make_unique<gqe::greater_equal_expression>(
          std::make_shared<gqe::column_reference_expression>(1),
          std::make_shared<gqe::column_reference_expression>(2)),
        std::vector<cudf::size_type>({0, 1, 2}));
      std::vector<std::unique_ptr<gqe::expression>> project_exprs;

      project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

      project_exprs.emplace_back(std::make_unique<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::literal_expression<int64_t>>(2)));

      project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

      auto project = std::make_unique<gqe::logical::project_relation>(
        std::move(filter),
        std::vector<std::shared_ptr<gqe::logical::relation>>(),
        std::move(project_exprs));
      return project;
    }

    if (test == test_type::col_ref_only) {
      if (!optimized) {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0, 1, 2}));
        std::vector<std::unique_ptr<gqe::expression>> col_0_exprs;
        col_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(filter),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(col_0_exprs));

        return project;
      } else {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0}));
        return filter;
      }
    }

    if (test == test_type::col_ref_with_reordering) {
      if (!optimized) {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0, 1, 2}));
        std::vector<std::unique_ptr<gqe::expression>> col_exprs;
        col_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));
        col_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(filter),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(col_exprs));

        return project;
      } else {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({1, 0}));
        return filter;
      }
    }

    if (test == test_type::mixed_expression) {
      if (!optimized) {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0, 2, 3, 4}));
        std::vector<std::unique_ptr<gqe::expression>> project_exprs;

        project_exprs.emplace_back(std::make_unique<gqe::if_then_else_expression>(
          std::make_shared<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(2),
            std::make_shared<gqe::literal_expression<int64_t>>(25)),
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(3)));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(filter),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_exprs));
        return project;
      } else {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0, 3, 4}));
        std::vector<std::unique_ptr<gqe::expression>> project_exprs;

        project_exprs.emplace_back(std::make_unique<gqe::if_then_else_expression>(
          std::make_shared<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::literal_expression<int64_t>>(25)),
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(2)));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(filter),
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
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0, 1, 2}));

        std::vector<std::unique_ptr<gqe::expression>> project_0_exprs;
        project_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));
        project_0_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(1));

        std::vector<std::unique_ptr<gqe::expression>> project_exprs;
        project_exprs.emplace_back(std::make_unique<gqe::column_reference_expression>(0));

        auto project_0 = std::make_unique<gqe::logical::project_relation>(
          std::move(filter),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_0_exprs));

        auto project = std::make_unique<gqe::logical::project_relation>(
          std::move(project_0),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::move(project_exprs));

        return project;
      }

      else {
        auto filter = std::make_unique<gqe::logical::filter_relation>(
          std::move(temp_table),
          std::vector<std::shared_ptr<gqe::logical::relation>>(),
          std::make_unique<gqe::greater_equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>({0}));

        return filter;
      }
    }

    else {
      throw std::invalid_argument("Unknown test type");
    }
  }
};

TEST_F(PushProjectionToFilter, ColRefOnly)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  auto test_plan = construct_test_plan(test_type::col_ref_only);
  auto ref_plan  = construct_optimized_plan(test_type::col_ref_only);

  // Optimize
  assert(optimizer);

  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(optimizer->get_rule_count(
              gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter),
            1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(PushProjectionToFilter, ColRefWithReordering)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  auto test_plan = construct_test_plan(test_type::col_ref_with_reordering);
  auto ref_plan  = construct_optimized_plan(test_type::col_ref_with_reordering);

  // Optimize
  assert(optimizer);

  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(optimizer->get_rule_count(
              gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter),
            1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(PushProjectionToFilter, MixedExpressions)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  auto test_plan = construct_test_plan(test_type::mixed_expression);
  auto ref_plan  = construct_optimized_plan(test_type::mixed_expression);

  // Optimize
  assert(optimizer);

  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(optimizer->get_rule_count(
              gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter),
            1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(PushProjectionToFilter, TransformationDirectionUp)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  auto test_plan = construct_test_plan(test_type::transformation_direction_up);
  auto ref_plan  = construct_optimized_plan(test_type::transformation_direction_up);

  // Optimize
  assert(optimizer);

  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(optimizer->get_rule_count(
              gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter),
            1);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(PushProjectionToFilter, NoOpt)
{
  // Initialize and create optimizer
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter}, {});
  initialize_optimizer(logical_rule_config);

  // Construct test and ref plans
  auto test_plan = construct_test_plan(test_type::no_opt);
  auto ref_plan  = construct_optimized_plan(test_type::no_opt);

  // Optimize
  assert(optimizer);

  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(optimizer->get_rule_count(
              gqe::optimizer::logical_optimization_rule_type::push_projection_to_filter),
            0);
  EXPECT_EQ(*ref_plan, *optimized_plan);
}
