/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/utility/tpch.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace std::string_literals;

class ColumnNameAssignmentTest : public ::testing::Test {
 protected:
  ColumnNameAssignmentTest()
  {
    // Register all TPC-H tables (same as integration test)
    auto const& table_definitions = gqe::utility::tpch::table_definitions();
    for (auto const& [name, definition] : table_definitions) {
      catalog.register_table(name,
                             definition,
                             gqe::storage_kind::parquet_file{{"/" + name}},
                             gqe::partitioning_schema_kind::automatic{});
    }
  }
  std::string test_resource_dir = std::string(TEST_RESOURCE_DIR);
  gqe::catalog catalog;
};

TEST_F(ColumnNameAssignmentTest, SimpleFilterWithSubstrait)
{
  /*
  Test for column name assignment in filter conditions using substrait data.
  Uses a simple SQL query:

  SELECT
    l_orderkey
  FROM
    lineitem
  WHERE
    l_quantity < 24

  This test checks that filter conditions show proper column names (l_quantity)
  instead of generic column references (column_reference(4)) after running
  the column name assignment optimization rule.
  */

  // Read and parse substrait file into logical plan
  std::string const substrait_file =
    test_resource_dir +
    "/substrait_plan/unittest/substrait_SELECT_l_orderkey_FROM_lineitem_WHERE_l_quantity_lt_24.bin";

  gqe::substrait_parser parser(&catalog);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);

  // Run optimizer with only column name assignment rule
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::column_name_assignment});
  gqe::optimizer::logical_optimizer logical_optimizer(&logical_rule_config, &catalog);
  auto optimized_plan = logical_optimizer.optimize(query_plan[0]);

  // Verify that we have a filter relation in the optimized plan
  ASSERT_NE(optimized_plan, nullptr);

  // The plan should be: Project -> Filter -> Read
  auto project_rel = dynamic_cast<gqe::logical::project_relation*>(optimized_plan.get());
  ASSERT_NE(project_rel, nullptr) << "Expected top-level project relation";

  auto children = project_rel->children_unsafe();
  ASSERT_EQ(children.size(), 1) << "Project should have one child";

  auto filter_rel = dynamic_cast<gqe::logical::filter_relation*>(children[0]);
  ASSERT_NE(filter_rel, nullptr) << "Expected filter relation as child of project";

  // Get the filter condition and check its string representation
  auto condition = filter_rel->condition();
  ASSERT_NE(condition, nullptr) << "Filter should have a condition";

  // Check that column names are properly assigned throughout the plan
  std::string plan_str = optimized_plan->to_string();

  // 1. Project output expressions should use proper names
  bool has_project_outputs =
    plan_str.find("\"output expressions\" : [\"l_orderkey\"]") != std::string::npos;

  // 2. Filter condition should use proper names
  bool has_filter_condition =
    plan_str.find("\"condition\" : \"l_quantity < literal(double 24.000000)\"") !=
    std::string::npos;

  // 3. Partial filter should use proper names (this was the main fix!)
  bool has_partial_filter =
    plan_str.find("\"partial filter\" : \"l_quantity < literal(double 24.000000)\"") !=
    std::string::npos;

  // 4. Ensure no generic column_reference strings remain anywhere
  bool has_generic_refs = plan_str.find("column_reference(") != std::string::npos;

  GQE_LOG_TRACE("After column name assignment for simple filter: \n {}",
                optimized_plan->to_string());

  // Print debug information if test fails
  if (!has_project_outputs || !has_filter_condition || !has_partial_filter || has_generic_refs) {
    std::cout << "=== COLUMN NAME ASSIGNMENT TEST DEBUG ===" << std::endl;
    std::cout << "Full plan: " << plan_str << std::endl;
    std::cout << "Has project outputs 'l_orderkey': " << (has_project_outputs ? "YES" : "NO")
              << std::endl;
    std::cout << "Has filter condition 'l_quantity': " << (has_filter_condition ? "YES" : "NO")
              << std::endl;
    std::cout << "Has partial filter 'l_quantity': " << (has_partial_filter ? "YES" : "NO")
              << std::endl;
    std::cout << "Has generic column_reference: " << (has_generic_refs ? "YES" : "NO") << std::endl;
    std::cout << "Has generic column_reference: " << (has_generic_refs ? "YES" : "NO") << std::endl;
    std::cout << "=== END DEBUG ===" << std::endl;
  }

  // All checks should pass
  EXPECT_TRUE(has_project_outputs) << "Project should use column name 'l_orderkey'";
  EXPECT_TRUE(has_filter_condition) << "Filter condition should use column name 'l_quantity'";
  EXPECT_TRUE(has_partial_filter) << "Partial filter should use column name 'l_quantity'";
  EXPECT_FALSE(has_generic_refs) << "Should not contain any generic column_reference() expressions";
}

TEST_F(ColumnNameAssignmentTest, FilterSelectAggregateWithSubstrait)
{
  /*
  Test for column name assignment across filter, select, and aggregate operations.
  Uses a substrait file with a more complex query similar to TPC-H patterns:

  This test checks that column names are properly assigned across:
  1. Filter conditions (WHERE clause)
  2. Project expressions (SELECT clause)
  3. Aggregate expressions (GROUP BY and aggregate functions)

  This mimics the structure of TPC-H Query 6 to test the interaction between
  filter, project, and aggregate operations.
  */

  // Try to find a substrait file with aggregation, or use a simpler one and build the plan manually
  // For now, let's build a logical plan manually that represents:
  // SELECT l_returnflag, sum(l_extendedprice * l_discount)
  // FROM lineitem
  // WHERE l_quantity < 24
  // GROUP BY l_returnflag

  // Create a read relation for lineitem
  std::vector<std::string> column_names = {
    "l_orderkey", "l_quantity", "l_extendedprice", "l_discount", "l_returnflag"};
  std::vector<cudf::data_type> data_types = {
    cudf::data_type(cudf::type_id::INT32),    // l_orderkey
    cudf::data_type(cudf::type_id::FLOAT64),  // l_quantity
    cudf::data_type(cudf::type_id::FLOAT64),  // l_extendedprice
    cudf::data_type(cudf::type_id::FLOAT64),  // l_discount
    cudf::data_type(cudf::type_id::INT8)      // l_returnflag
  };

  auto read_rel = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    column_names,
    data_types,
    "lineitem",
    nullptr  // no partial filter
  );

  // Create filter: l_quantity < 24
  auto l_quantity_ref =
    std::make_shared<gqe::column_reference_expression>(1);  // column index 1 = l_quantity
  auto literal_24       = std::make_shared<gqe::literal_expression<double>>(24.0);
  auto filter_condition = std::make_unique<gqe::less_expression>(l_quantity_ref, literal_24);

  // Filter projects all columns (no projection)
  std::vector<cudf::size_type> filter_projection_indices = {0, 1, 2, 3, 4};
  auto filter_rel = std::make_shared<gqe::logical::filter_relation>(
    read_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(filter_condition),
    filter_projection_indices);

  // Create aggregate: GROUP BY l_returnflag, SUM(l_extendedprice * l_discount)
  // Key: l_returnflag (column index 4)
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(4));  // l_returnflag

  // Measure: SUM(l_extendedprice * l_discount) - columns 2 and 3
  auto l_extendedprice_ref = std::make_shared<gqe::column_reference_expression>(2);
  auto l_discount_ref      = std::make_shared<gqe::column_reference_expression>(3);
  auto multiply_expr =
    std::make_unique<gqe::multiply_expression>(l_extendedprice_ref, l_discount_ref);

  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> measures;
  measures.emplace_back(cudf::aggregation::Kind::SUM, std::move(multiply_expr));

  auto aggregate_rel = std::make_shared<gqe::logical::aggregate_relation>(
    filter_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(keys),
    std::move(measures));

  // Create project: SELECT l_returnflag, sum(l_extendedprice * l_discount)
  std::vector<std::unique_ptr<gqe::expression>> output_expressions;
  output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(
    0));  // l_returnflag (now column 0 in aggregate output)
  output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(
    1));  // sum result (column 1 in aggregate output)

  auto project_rel = std::make_shared<gqe::logical::project_relation>(
    aggregate_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(output_expressions));

  // Run optimizer with only column name assignment rule
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::column_name_assignment});
  gqe::optimizer::logical_optimizer logical_optimizer(&logical_rule_config, &catalog);
  auto optimized_plan = logical_optimizer.optimize(project_rel);

  // Verify the optimized plan has proper column names
  ASSERT_NE(optimized_plan, nullptr);

  // Get the string representation of the entire plan
  std::string plan_str = optimized_plan->to_string();

  // Check that we see proper column names instead of generic column_reference
  // Test specific components individually

  // 1. Project output expressions should use proper names
  bool has_project_outputs =
    plan_str.find(
      "\"output expressions\" : [\"l_returnflag\", \"sum(l_extendedprice * l_discount)\"]") !=
    std::string::npos;

  // 2. Aggregate key expressions should use proper names
  bool has_aggregate_keys =
    plan_str.find("\"key expressions\" : [\"l_returnflag\"]") != std::string::npos;

  // 3. Aggregate measures should use proper names
  bool has_aggregate_measures =
    plan_str.find("\"measures\" : [\"SUM(l_extendedprice * l_discount)\"]") != std::string::npos;

  // 4. Filter condition should use proper names
  bool has_filter_condition =
    plan_str.find("\"condition\" : \"l_quantity < literal(double 24.000000)\"") !=
    std::string::npos;

  // 5. Ensure no generic column_reference strings remain anywhere
  bool has_generic_refs = plan_str.find("column_reference(") != std::string::npos;

  GQE_LOG_TRACE("After column name assignment for filter select aggregate: \n {}",
                optimized_plan->to_string());

  // Print debug information if test fails
  if (!has_project_outputs || !has_aggregate_keys || !has_aggregate_measures ||
      !has_filter_condition || has_generic_refs) {
    std::cout << "=== COMPONENT-SPECIFIC TEST DEBUG ===" << std::endl;
    std::cout << "Full plan: " << plan_str << std::endl;
    std::cout << "Has project outputs: " << (has_project_outputs ? "YES" : "NO") << std::endl;
    std::cout << "Has aggregate keys: " << (has_aggregate_keys ? "YES" : "NO") << std::endl;
    std::cout << "Has aggregate measures: " << (has_aggregate_measures ? "YES" : "NO") << std::endl;
    std::cout << "Has filter condition: " << (has_filter_condition ? "YES" : "NO") << std::endl;
    std::cout << "Has generic column_reference: " << (has_generic_refs ? "YES" : "NO") << std::endl;
    std::cout << "=== END DEBUG ===" << std::endl;
  }

  // Test each component individually
  EXPECT_TRUE(has_project_outputs) << "Project output expressions should use proper column names";
  EXPECT_TRUE(has_aggregate_keys) << "Aggregate key expressions should use proper column names";
  EXPECT_TRUE(has_aggregate_measures) << "Aggregate measures should use proper column names";
  EXPECT_TRUE(has_filter_condition) << "Filter condition should use proper column names";
  EXPECT_FALSE(has_generic_refs)
    << "Plan should not contain 'column_reference(' after column name assignment";
}

TEST_F(ColumnNameAssignmentTest, JoinWithSubstrait)
{
  /*
  Test for column name assignment in join operations.
  Builds a logical plan that represents:

  SELECT
    c_custkey, o_orderkey, o_totalprice
  FROM
    customer c
  INNER JOIN
    orders o
  ON
    c.c_custkey = o.o_custkey
  WHERE
    c.c_acctbal > 1000.0

  This test checks that:
  1. Join conditions use proper column names from left and right tables
  2. Output column names are correctly formed from both tables
  3. Filter conditions on joined data use proper column names
  4. No generic column_reference expressions remain after optimization
  */

  // Create read relation for customer table (left side)
  std::vector<std::string> customer_column_names = {
    "c_custkey", "c_name", "c_acctbal", "c_mktsegment"};
  std::vector<cudf::data_type> customer_data_types = {
    cudf::data_type(cudf::type_id::INT32),    // c_custkey
    cudf::data_type(cudf::type_id::STRING),   // c_name
    cudf::data_type(cudf::type_id::FLOAT64),  // c_acctbal
    cudf::data_type(cudf::type_id::STRING)    // c_mktsegment
  };

  auto customer_read = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    customer_column_names,
    customer_data_types,
    "customer",
    nullptr  // no partial filter
  );

  // Create read relation for orders table (right side)
  std::vector<std::string> orders_column_names = {
    "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"};
  std::vector<cudf::data_type> orders_data_types = {
    cudf::data_type(cudf::type_id::INT32),    // o_orderkey
    cudf::data_type(cudf::type_id::INT32),    // o_custkey
    cudf::data_type(cudf::type_id::FLOAT64),  // o_totalprice
    cudf::data_type(cudf::type_id::STRING)    // o_orderdate
  };

  auto orders_read = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    orders_column_names,
    orders_data_types,
    "orders",
    nullptr  // no partial filter
  );

  // Create join condition: c.c_custkey = o.o_custkey
  // Left side: c_custkey (column 0 in customer)
  auto left_custkey_ref = std::make_shared<gqe::column_reference_expression>(0);  // c_custkey
  // Right side: o_custkey (column 1 in orders, but becomes column 5 in joined output: 0,1,2,3 from
  // customer + 0,1,2,3 from orders)
  auto right_custkey_ref =
    std::make_shared<gqe::column_reference_expression>(5);  // o_custkey (4 customer cols + 1)
  auto join_condition =
    std::make_unique<gqe::equal_expression>(left_custkey_ref, right_custkey_ref);

  // Projection indices for join output: select c_custkey, o_orderkey, o_totalprice
  // Full output would be: [c_custkey, c_name, c_acctbal, c_mktsegment, o_orderkey, o_custkey,
  // o_totalprice, o_orderdate]
  //                       [   0    ,   1   ,    2     ,      3       ,     4     ,    5     , 6 ,
  //                       7     ]
  std::vector<cudf::size_type> join_projection_indices = {
    0, 4, 6};  // c_custkey, o_orderkey, o_totalprice

  auto join_rel = std::make_shared<gqe::logical::join_relation>(
    customer_read,
    orders_read,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(join_condition),
    gqe::join_type_type::inner,
    join_projection_indices);

  // Create filter: c_acctbal > 1000.0
  // We need c_acctbal in the join projection to filter on it
  join_projection_indices = {0, 2, 4, 6};  // c_custkey, c_acctbal, o_orderkey, o_totalprice

  // Recreate join with updated projection
  left_custkey_ref  = std::make_shared<gqe::column_reference_expression>(0);
  right_custkey_ref = std::make_shared<gqe::column_reference_expression>(5);
  join_condition    = std::make_unique<gqe::equal_expression>(left_custkey_ref, right_custkey_ref);

  join_rel = std::make_shared<gqe::logical::join_relation>(
    customer_read,
    orders_read,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(join_condition),
    gqe::join_type_type::inner,
    join_projection_indices);

  // Create filter: c_acctbal > 1000.0 (column 1 in join output: c_custkey, c_acctbal, o_orderkey,
  // o_totalprice)
  auto c_acctbal_ref    = std::make_shared<gqe::column_reference_expression>(1);  // c_acctbal
  auto literal_1000     = std::make_shared<gqe::literal_expression<double>>(1000.0);
  auto filter_condition = std::make_unique<gqe::greater_expression>(c_acctbal_ref, literal_1000);

  // Filter projects all columns from join (no additional projection)
  std::vector<cudf::size_type> filter_projection_indices = {0, 1, 2, 3};
  auto filter_rel = std::make_shared<gqe::logical::filter_relation>(
    join_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(filter_condition),
    filter_projection_indices);

  // Create final project: SELECT c_custkey, o_orderkey, o_totalprice (remove c_acctbal)
  std::vector<std::unique_ptr<gqe::expression>> output_expressions;
  output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));  // c_custkey
  output_expressions.push_back(
    std::make_unique<gqe::column_reference_expression>(2));  // o_orderkey
  output_expressions.push_back(
    std::make_unique<gqe::column_reference_expression>(3));  // o_totalprice

  auto project_rel = std::make_shared<gqe::logical::project_relation>(
    filter_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(output_expressions));

  // Run optimizer with only column name assignment rule
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::column_name_assignment});
  gqe::optimizer::logical_optimizer logical_optimizer(&logical_rule_config, &catalog);
  auto optimized_plan = logical_optimizer.optimize(project_rel);

  // Verify the optimized plan has proper column names
  ASSERT_NE(optimized_plan, nullptr);

  // Get the string representation of the entire plan
  std::string plan_str = optimized_plan->to_string();

  // Check that we see proper column names instead of generic column_reference

  // 1. Project output expressions should use proper names
  bool has_project_outputs =
    plan_str.find("\"output expressions\" : [\"c_custkey\", \"o_orderkey\", \"o_totalprice\"]") !=
    std::string::npos;

  // 2. Filter condition should use proper names
  bool has_filter_condition =
    plan_str.find("\"condition\" : \"c_acctbal > literal(double 1000.000000)\"") !=
    std::string::npos;

  // 3. Join condition should use proper names
  bool has_join_condition =
    plan_str.find("\"condition\" : \"c_custkey = o_custkey\"") != std::string::npos;

  // 4. Join projection should use proper names (check for presence of expected column names)
  bool has_join_output = plan_str.find("c_custkey") != std::string::npos &&
                         plan_str.find("o_orderkey") != std::string::npos &&
                         plan_str.find("o_totalprice") != std::string::npos;

  // 5. Ensure no generic column_reference strings remain anywhere
  bool has_generic_refs = plan_str.find("column_reference(") != std::string::npos;

  GQE_LOG_TRACE("After column name assignment for join: \n {}", optimized_plan->to_string());

  // Print debug information if test fails
  if (!has_project_outputs || !has_filter_condition || !has_join_condition || !has_join_output ||
      has_generic_refs) {
    std::cout << "=== JOIN TEST DEBUG ===" << std::endl;
    std::cout << "Full plan: " << plan_str << std::endl;
    std::cout << "Has project outputs: " << (has_project_outputs ? "YES" : "NO") << std::endl;
    std::cout << "Has filter condition: " << (has_filter_condition ? "YES" : "NO") << std::endl;
    std::cout << "Has join condition: " << (has_join_condition ? "YES" : "NO") << std::endl;
    std::cout << "Has join output: " << (has_join_output ? "YES" : "NO") << std::endl;
    std::cout << "Has generic column_reference: " << (has_generic_refs ? "YES" : "NO") << std::endl;
    std::cout << "=== END DEBUG ===" << std::endl;
  }

  // Test each component individually
  EXPECT_TRUE(has_project_outputs) << "Project output expressions should use proper column names";
  EXPECT_TRUE(has_filter_condition) << "Filter condition should use proper column names";
  EXPECT_TRUE(has_join_condition) << "Join condition should use proper column names";
  EXPECT_TRUE(has_join_output) << "Join should output columns with proper names";
  EXPECT_FALSE(has_generic_refs)
    << "Plan should not contain 'column_reference(' after column name assignment";
}

TEST_F(ColumnNameAssignmentTest, UnionAllWithColumnNames)
{
  /*
  Test for column name assignment in union all operations.

  Creates two simple read relations with the same schema and unions them.
  This test verifies that set operations preserve column names from their inputs
  instead of generating generic column names like "col_default_0".

  Before the fix, set operations were in the default case and generated generic names.
  After the fix, they should preserve the input column structure.
  */

  // Create first table: customer with (c_custkey, c_name)
  std::vector<std::string> customer_column_names   = {"c_custkey", "c_name"};
  std::vector<cudf::data_type> customer_data_types = {
    cudf::data_type(cudf::type_id::INT64),  // c_custkey
    cudf::data_type(cudf::type_id::STRING)  // c_name
  };

  auto customer_read_rel = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    customer_column_names,
    customer_data_types,
    "customer",
    nullptr  // no partial filter
  );

  // Create second table: supplier with same schema (s_suppkey, s_name)
  // Note: For union to work, both sides must have identical schemas
  std::vector<std::string> supplier_column_names   = {"s_suppkey", "s_name"};
  std::vector<cudf::data_type> supplier_data_types = {
    cudf::data_type(cudf::type_id::INT64),  // s_suppkey
    cudf::data_type(cudf::type_id::STRING)  // s_name
  };

  auto supplier_read_rel = std::make_shared<gqe::logical::read_relation>(
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    supplier_column_names,
    supplier_data_types,
    "supplier",
    nullptr  // no partial filter
  );

  // Create union all operation
  auto union_rel = std::make_shared<gqe::logical::set_relation>(
    customer_read_rel, supplier_read_rel, gqe::logical::set_relation::set_union_all);

  // Create project to select columns from union result
  std::vector<std::unique_ptr<gqe::expression>> output_expressions;
  output_expressions.push_back(
    std::make_unique<gqe::column_reference_expression>(0));  // key column
  output_expressions.push_back(
    std::make_unique<gqe::column_reference_expression>(1));  // name column

  auto project_rel = std::make_shared<gqe::logical::project_relation>(
    union_rel,
    std::vector<std::shared_ptr<gqe::logical::relation>>{},  // subqueries
    std::move(output_expressions));

  // Run optimizer with only column name assignment rule
  gqe::optimizer::optimization_configuration logical_rule_config(
    {gqe::optimizer::logical_optimization_rule_type::column_name_assignment});
  gqe::optimizer::logical_optimizer logical_optimizer(&logical_rule_config, &catalog);
  auto optimized_plan = logical_optimizer.optimize(project_rel);

  // Verify the optimized plan has proper column names
  ASSERT_NE(optimized_plan, nullptr);

  // Get the string representation of the entire plan
  std::string plan_str = optimized_plan->to_string();

  // Check that union preserves the left-hand side column names
  // The union should use column names from the left input (customer)
  bool has_proper_column_names =
    plan_str.find("c_custkey") != std::string::npos && plan_str.find("c_name") != std::string::npos;

  // Ensure no generic column names were generated for the set operation
  bool has_generic_set_names = plan_str.find("col_default_") != std::string::npos;

  // Ensure no generic column_reference strings remain anywhere
  bool has_generic_refs = plan_str.find("column_reference(") != std::string::npos;

  GQE_LOG_TRACE("After column name assignment for union all: \n {}", optimized_plan->to_string());

  // Print debug information if test fails
  if (!has_proper_column_names || has_generic_set_names || has_generic_refs) {
    std::cout << "=== UNION ALL TEST DEBUG ===" << std::endl;
    std::cout << "Full plan: " << plan_str << std::endl;
    std::cout << "Has proper column names (c_custkey, c_name): "
              << (has_proper_column_names ? "YES" : "NO") << std::endl;
    std::cout << "Has generic set names (col_default_): " << (has_generic_set_names ? "YES" : "NO")
              << std::endl;
    std::cout << "Has generic column_reference: " << (has_generic_refs ? "YES" : "NO") << std::endl;
    std::cout << "=== END DEBUG ===" << std::endl;
  }

  // All checks should pass
  EXPECT_TRUE(has_proper_column_names) << "Union should preserve column names from left input";
  EXPECT_FALSE(has_generic_set_names) << "Union should not generate generic col_default_ names";
  EXPECT_FALSE(has_generic_refs)
    << "Plan should not contain 'column_reference(' after column name assignment";
}