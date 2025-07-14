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
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/optimizer/relation_traits.hpp>
#include <gqe/optimizer/rules/uniqueness_propagation.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cassert>
#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class UniquenessPropagationTest : public ::testing::Test {
 protected:
  void initialize_optimizer()
  {
    gqe::optimizer::optimization_configuration rule_config(
      {gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation}, {});
    using col_prop = gqe::column_traits::column_property;
    catalog.register_table(
      "test_table1",
      {{"t1_c1_unique", cudf::data_type(cudf::type_id::INT64), {col_prop::unique}},
       {"t1_c2", cudf::data_type(cudf::type_id::INT64), {}},
       {"t1_c3", cudf::data_type(cudf::type_id::INT32), {}},
       {"t1_c4_unique", cudf::data_type(cudf::type_id::INT64), {col_prop::unique}}},
      gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(temp_env->get_temp_dir())},
      gqe::partitioning_schema_kind::automatic{});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&rule_config, &catalog);

    catalog.register_table(
      "test_table2",
      {{"t2_c1_unique", cudf::data_type(cudf::type_id::INT64), {col_prop::unique}},
       {"t2_c2_unique", cudf::data_type(cudf::type_id::INT32), {col_prop::unique}}},
      gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(temp_env->get_temp_dir())},
      gqe::partitioning_schema_kind::automatic{});
    optimizer = std::make_unique<gqe::optimizer::logical_optimizer>(&rule_config, &catalog);
  }

  std::vector<std::shared_ptr<gqe::logical::relation>> empty_relations() { return {}; }

  std::unique_ptr<gqe::logical::relation> construct_read_one_unique(bool optimized)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names = {"t1_c1_unique", "t1_c3"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_unique<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table1", nullptr);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      _add_column_uniqueness(read_rel.get(), {0});  // only column `t1_c1_unique` is unique
    }
    return read_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_read_all_unique(bool optimized)
  {
    // Hand coded logical plan for testing
    std::vector<std::string> column_names = {"t2_c1_unique", "t2_c2_unique"};
    auto column_types                     = {cudf::data_type(cudf::type_id::INT64),
                                             cudf::data_type(cudf::type_id::INT32)};
    std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

    auto read_rel = std::make_unique<gqe::logical::read_relation>(
      subquery_relations, column_names, column_types, "test_table2", nullptr);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Both columns are unique
      _add_column_uniqueness(read_rel.get(), {0, 1});
    }
    return read_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_agg(bool optimized)
  {
    auto read_rel = construct_read_one_unique(optimized);
    std::vector<std::unique_ptr<gqe::expression>> keys;
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;

    keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto agg_rel = std::make_unique<gqe::logical::aggregate_relation>(
      std::move(read_rel), empty_relations(), std::move(keys), std::move(measures));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      _add_column_uniqueness(agg_rel.get(), {0});  // Only the key column in
                                                   // unique
    }

    return agg_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_unique_RHS(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_1 = std::make_shared<gqe::column_reference_expression>(1);  // not unique
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto cond  = std::make_unique<gqe::equal_expression>(col_1, col_2);  // unique RHS if optimized
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // LHS columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_compound_condition(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_0  = std::make_shared<gqe::column_reference_expression>(0);  // unique if optimized
    auto col_1  = std::make_shared<gqe::column_reference_expression>(1);  // not unique
    auto col_2  = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto col_3  = std::make_shared<gqe::column_reference_expression>(3);  // unique if optimized
    auto cond_0 = std::make_shared<gqe::equal_expression>(col_1, col_2);  // unique RHS if optimized
    auto cond_1 = std::make_shared<gqe::equal_expression>(col_1, col_3);  // unique RHS if optimized
    auto cond_0_and_1 = std::make_shared<gqe::logical_and_expression>(cond_0, cond_1);
    auto cond_2 =
      std::make_shared<gqe::equal_expression>(col_0, col_2);  // unique LHS & RHS if optimized
    auto cond_0_and_1_or_2 = std::make_unique<gqe::logical_or_expression>(cond_0_and_1, cond_2);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond_0_and_1_or_2),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // LHS columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_join_unique_LHS_RHS(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    auto col_0 = std::make_shared<gqe::column_reference_expression>(0);  // unique if optimized
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);  // unique if optimized
    auto cond =
      std::make_unique<gqe::equal_expression>(col_0, col_2);  // unique LHS & RHS if optimized
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(read_rel_0),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // All columns get to keep their uniqueness
      _add_column_uniqueness(join_rel.get(), {0, 2, 3});
    }

    return join_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_fetch(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    auto fetch_rel  = std::make_unique<gqe::logical::fetch_relation>(std::move(read_rel_0), 0, 10);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Only column 0 is unique
      _add_column_uniqueness(fetch_rel.get(), {0});
    }

    return fetch_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_project(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> output_expressions;
    // Switch column order in the output
    output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
    output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
    auto project_rel = std::make_unique<gqe::logical::project_relation>(
      std::move(read_rel_0), empty_relations(), std::move(output_expressions));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Column 1 is unique due to column order switch
      _add_column_uniqueness(project_rel.get(), {1});
    }

    return project_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_set(
    bool optimized, gqe::logical::set_relation::set_operator_type set_op)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    auto read_rel_1 = construct_read_all_unique(optimized);  // uniqueness: true, true

    auto set_rel = std::make_unique<gqe::logical::set_relation>(
      std::move(read_rel_0), std::move(read_rel_1), set_op);

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      if (set_op == gqe::logical::set_relation::set_intersect ||
          set_op == gqe::logical::set_relation::set_minus) {
        _add_column_uniqueness(set_rel.get(), {0});
      }
    }

    return set_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_sort(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> sort_exprs;
    sort_exprs.push_back(std::make_unique<gqe::column_reference_expression>(1));

    auto sort_rel = std::make_unique<gqe::logical::sort_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      std::vector<cudf::null_order>({cudf::null_order::BEFORE}),
      std::move(sort_exprs));

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Column 1 is unique due to column order switch
      _add_column_uniqueness(sort_rel.get(), {0});
    }

    return sort_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_window(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false
    std::vector<std::unique_ptr<gqe::expression>> arguments;
    std::vector<std::unique_ptr<gqe::expression>> order_by;
    std::vector<std::unique_ptr<gqe::expression>> partition_by;

    arguments.push_back(std::make_unique<gqe::column_reference_expression>(0));
    order_by.push_back(std::make_unique<gqe::column_reference_expression>(0));
    partition_by.push_back(std::make_unique<gqe::column_reference_expression>(1));

    auto window_rel = std::make_unique<gqe::logical::window_relation>(
      std::move(read_rel_0),
      empty_relations(),
      cudf::aggregation::Kind::RANK,
      std::move(arguments),
      std::move(order_by),
      std::move(partition_by),
      std::vector<cudf::order>({cudf::order::ASCENDING}),
      gqe::window_frame_bound::unbounded(),
      gqe::window_frame_bound::unbounded());

    if (optimized) {
      // Optimized plan should have uniqueness info in the plan
      // Passthrough uniqueness from input
      _add_column_uniqueness(window_rel.get(), {0});
    }

    return window_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_write(bool optimized)
  {
    auto read_rel_0 = construct_read_one_unique(optimized);  // uniqueness: true, false

    std::vector<std::string> col_names = {"t1_c1_unique", "t1_c3"};
    auto col_types = {cudf::data_type(cudf::type_id::INT64), cudf::data_type(cudf::type_id::INT32)};
    auto write_rel = std::make_unique<gqe::logical::write_relation>(
      std::move(read_rel_0), std::move(col_names), std::move(col_types), "test_table1");

    if (optimized) { _add_column_uniqueness(write_rel.get(), {0}); }

    return write_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_plan_filter(bool optimized)
  {
    auto read_rel_0 = construct_read_all_unique(optimized);  // uniqueness: true, false
    // project subset of columns
    std::vector<cudf::size_type> filter_projection_indices = {1};
    auto filter_rel = std::make_unique<gqe::logical::filter_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(filter_projection_indices));

    if (optimized) { _add_column_uniqueness(filter_rel.get(), {0}); }

    return filter_rel;
  }

  std::unique_ptr<gqe::logical::relation> construct_three_level_plan(bool optimized)
  {
    // fetch(join(filter(read0), read1))
    auto read_rel_0 = construct_read_one_unique(optimized);
    auto read_rel_1 = construct_read_all_unique(optimized);

    // project all columns
    std::vector<cudf::size_type> filter_projection_indices(read_rel_0->num_columns());
    std::iota(filter_projection_indices.begin(), filter_projection_indices.end(), 0);

    auto filter_rel = std::make_unique<gqe::logical::filter_relation>(
      std::move(read_rel_0),
      empty_relations(),
      std::make_unique<gqe::literal_expression<bool>>(true),
      std::move(filter_projection_indices));
    if (optimized) _add_column_uniqueness(filter_rel.get(), {0});

    auto col_0 = std::make_shared<gqe::column_reference_expression>(0);
    auto col_2 = std::make_shared<gqe::column_reference_expression>(2);
    auto cond  = std::make_unique<gqe::equal_expression>(col_0, col_2);
    std::vector<cudf::size_type> projection_indices = {0, 1, 2, 3};

    auto join_rel = std::make_unique<gqe::logical::join_relation>(std::move(filter_rel),
                                                                  std::move(read_rel_1),
                                                                  empty_relations(),
                                                                  std::move(cond),
                                                                  gqe::join_type_type::inner,
                                                                  projection_indices);
    if (optimized) _add_column_uniqueness(join_rel.get(), {0, 2, 3});

    auto fetch_rel = std::make_unique<gqe::logical::fetch_relation>(std::move(join_rel), 0, 10);
    if (optimized) _add_column_uniqueness(fetch_rel.get(), {0, 2, 3});

    return fetch_rel;
  }

  gqe::catalog catalog;
  std::unique_ptr<gqe::optimizer::logical_optimizer> optimizer;
  std::shared_ptr<gqe::logical::relation> test_plan;
  std::unique_ptr<gqe::logical::relation> ref_plan;

 private:
  void _add_column_uniqueness(gqe::logical::relation* rel, std::vector<cudf::size_type> col_indices)
  {
    gqe::optimizer::relation_properties props;
    for (auto idx : col_indices) {
      props.add_column_property(idx, gqe::optimizer::column_property::property_id::unique);
    }
    auto traits = std::make_unique<gqe::optimizer::relation_traits>(props);
    rel->set_relation_traits(std::move(traits));
  }
};

TEST_F(UniquenessPropagationTest, SimpleRead)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans

  test_plan = construct_read_one_unique(false);
  ref_plan  = construct_read_one_unique(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleAgg)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_agg(false);
  ref_plan  = construct_plan_agg(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleJoinUniqueRHS)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_unique_RHS(false);
  ref_plan  = construct_plan_join_unique_RHS(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleJoinUniqueLhsRhs)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_unique_LHS_RHS(false);
  ref_plan  = construct_plan_join_unique_LHS_RHS(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, CompoundConditionJoin)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_join_compound_condition(false);
  ref_plan  = construct_plan_join_compound_condition(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleFetch)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_fetch(false);
  ref_plan  = construct_plan_fetch(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleFilter)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_filter(false);
  ref_plan  = construct_plan_filter(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleProject)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_project(false);
  ref_plan  = construct_plan_project(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleIntersect)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_intersect);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_intersect);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleMinus)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_minus);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_minus);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleUnion)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_union);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_union);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleUnionAll)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_set(false, gqe::logical::set_relation::set_union_all);
  ref_plan  = construct_plan_set(true, gqe::logical::set_relation::set_union_all);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleSort)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_sort(false);
  ref_plan  = construct_plan_sort(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleWindow)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_window(false);
  ref_plan  = construct_plan_window(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, SimpleWrite)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_plan_write(false);
  ref_plan  = construct_plan_write(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}

TEST_F(UniquenessPropagationTest, ComplexPlan)
{
  // Initialize and create optimizer
  initialize_optimizer();

  // Construct test and ref plans
  test_plan = construct_three_level_plan(false);
  ref_plan  = construct_three_level_plan(true);

  // Optimize
  assert(optimizer);
  auto optimized_plan = optimizer->optimize(test_plan);

  // Test
  EXPECT_EQ(*ref_plan, *optimized_plan);
}
