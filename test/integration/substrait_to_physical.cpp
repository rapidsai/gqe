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
#include <gqe/expression/literal.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/tpch.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <vector>

static std::string const test_resource_dir = std::string(TEST_RESOURCE_DIR);

class SubstraitToPhysical : public ::testing::Test {
 protected:
  SubstraitToPhysical()
  {
    // Register all tables
    auto const& table_definitions = gqe::utility::tpch::table_definitions();
    for (auto const& [name, definition] : table_definitions) {
      catalog.register_table(name,
                             definition,
                             gqe::storage_kind::parquet_file{{"/" + name}},
                             gqe::partitioning_schema_kind::automatic{});
    }
  }

  std::string const test_resource_dir = std::string(TEST_RESOURCE_DIR);
  gqe::task_manager_context task_manager_ctx;
  gqe::catalog catalog{&task_manager_ctx};
};

TEST_F(SubstraitToPhysical, SubstraitFilterToPhysicalReadRelationPartialFilter)
{
  /*
  Substrait file generated as described in section "Running DataFusion-Substrait producer from GQE"
  of GQE README.md with the following input SQL string:

  SELECT
    l_orderkey
  FROM
    lineitem
  WHERE
    l_quantity < 24

  */

  // Read and parse substrait file into logical plan
  std::string const substrait_file =
    test_resource_dir +
    "/substrait_plan/unittest/substrait_SELECT_l_orderkey_FROM_lineitem_WHERE_l_quantity_lt_24.bin";
  std::cout << "Parsing Substrait plan at " << substrait_file << std::endl;
  gqe::substrait_parser parser(&catalog);
  std::vector<std::shared_ptr<gqe::logical::relation>> query_plan =
    parser.from_file(substrait_file);

  // Build physical plan from logical plan
  gqe::physical_plan_builder plan_builder(&catalog);
  auto physical_plan = plan_builder.build(query_plan[0].get());

  // Traverse physical plan and look for partial filter expression
  std::queue<gqe::physical::relation*> q;
  q.push(physical_plan.get());
  std::vector<gqe::expression*> partial_filter_expr;
  // BFS traversal
  while (q.size()) {
    gqe::physical::relation* pr = q.front();
    q.pop();

    gqe::physical::read_relation* read_rel = dynamic_cast<gqe::physical::read_relation* const>(pr);
    if (read_rel) {
      gqe::expression* pf = read_rel->partial_filter_unsafe();
      if (pf) partial_filter_expr.push_back(pf);
    }
    for (auto child : pr->children_unsafe())
      q.push(child);
    for (auto subquery : pr->subqueries_unsafe())
      q.push(subquery);
  }

  // Verify partial filter found from traversal
  ASSERT_EQ(partial_filter_expr.size(), 1);
  // construct expected partial filter expression: l_quantity (i.e. col 4) < 24
  auto l_quantity = std::dynamic_pointer_cast<gqe::expression>(
    std::make_shared<gqe::column_reference_expression>(4));
  auto double_literal_24 = std::dynamic_pointer_cast<gqe::expression>(
    std::make_shared<gqe::literal_expression<double>>(24.0));
  auto partial_filter_expr_expected =
    std::make_unique<gqe::less_expression>(l_quantity, double_literal_24);
  EXPECT_EQ(*partial_filter_expr[0], *partial_filter_expr_expected);
}
