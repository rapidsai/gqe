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

// Unit tests for parsed_query::output_column_names: verifies the query's output column names
// (Substrait RelRoot.names) are captured during parsing so results can be labelled with the
// query's real column names / aliases instead of generic placeholders.

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/tpch.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

class SubstraitOutputNamesTest : public gqe::test::BaseFixture {
 protected:
  SubstraitOutputNamesTest()
  {
    // Register all TPC-H tables so the parser can resolve read-relation column types.
    auto const opms               = gqe::make_optimization_parameters(true);
    auto const& table_definitions = gqe::utility::tpch::table_definitions(opms);
    for (auto const& [name, definition] : table_definitions) {
      catalog.register_table(name,
                             definition.columns,
                             gqe::storage_kind::parquet_file{{"/" + name}},
                             gqe::partitioning_schema_kind::automatic{},
                             definition.unique_keys);
    }
  }
  std::string test_resource_dir = std::string(TEST_RESOURCE_DIR);
  gqe::catalog catalog{get_task_manager_ctx()};
};

// The fixture plan is `SELECT l_orderkey FROM lineitem WHERE l_quantity < 24`, whose single
// output column is carried in the plan's RelRoot.names. Parsing must expose it verbatim.
TEST_F(SubstraitOutputNamesTest, CapturesRootOutputName)
{
  std::string const substrait_file =
    test_resource_dir +
    "/substrait_plan/unittest/substrait_SELECT_l_orderkey_FROM_lineitem_WHERE_l_quantity_lt_24.bin";

  auto const opms = gqe::make_optimization_parameters(true);
  gqe::substrait_parser parser(&catalog, &opms);
  auto query_plan = parser.from_file(substrait_file);

  ASSERT_EQ(query_plan.size(), 1u);
  EXPECT_THAT(query_plan[0].output_column_names, ::testing::ElementsAre("l_orderkey"));
}
