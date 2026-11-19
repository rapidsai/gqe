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

// Unit tests for substrait_parser::from_json: a JSON plan imports the same as its binary form,
// and invalid JSON or unknown fields are rejected.

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/utility/tpch.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <google/protobuf/util/json_util.h>
#include <substrait/plan.pb.h>

#include <fstream>
#include <stdexcept>
#include <string>

/**
 * @brief Fixture that loads a binary Substrait plan and parses it in binary and JSON form.
 *
 * The plan is `SELECT l_orderkey FROM lineitem WHERE l_quantity < 24`. All TPC-H tables are
 * registered in the catalog so the parser can resolve read-relation column types.
 */
class SubstraitJsonTest : public gqe::test::BaseFixture {
 protected:
  SubstraitJsonTest()
  {
    auto const& table_definitions = gqe::utility::tpch::table_definitions(opms);
    for (auto const& [name, definition] : table_definitions) {
      catalog.register_table(name,
                             definition.columns,
                             gqe::storage_kind::parquet_file{{"/" + name}},
                             gqe::partitioning_schema_kind::automatic{},
                             definition.unique_keys);
    }
  }

  void SetUp() override
  {
    std::ifstream stream(std::string(TEST_RESOURCE_DIR) +
                           "/substrait_plan/unittest/"
                           "substrait_SELECT_l_orderkey_FROM_lineitem_WHERE_l_quantity_lt_24.bin",
                         std::ios::binary);
    ASSERT_TRUE(plan.ParseFromIstream(&stream));
  }

  /**
   * @brief Serialize `plan` to protobuf JSON.
   *
   * @param use_proto_field_names Emit `snake_case` field names instead of `lowerCamelCase`.
   */
  std::string plan_json(bool use_proto_field_names = false) const
  {
    google::protobuf::util::JsonPrintOptions options;
    options.preserve_proto_field_names = use_proto_field_names;
    std::string json;
    EXPECT_TRUE(google::protobuf::util::MessageToJsonString(plan, &json, options).ok());
    return json;
  }

  /**
   * @brief Expect `json` to import to the same queries as the binary serialization of `plan`.
   */
  void expect_matches_binary(std::string const& json)
  {
    std::string binary;
    ASSERT_TRUE(plan.SerializeToString(&binary));

    gqe::substrait_parser binary_parser(&catalog, &opms);
    auto const from_binary = binary_parser.from_binary(binary.data(), binary.size());
    gqe::substrait_parser json_parser(&catalog, &opms);
    auto const from_json = json_parser.from_json(json);

    ASSERT_EQ(from_binary.size(), 1u);
    ASSERT_EQ(from_json.size(), 1u);
    EXPECT_EQ(from_json[0].relation->to_string(), from_binary[0].relation->to_string());
    EXPECT_EQ(from_json[0].output_column_names, from_binary[0].output_column_names);
  }

  gqe::optimization_parameters const opms = gqe::make_optimization_parameters(true);
  gqe::catalog catalog{get_task_manager_ctx()};
  substrait::Plan plan;
};

TEST_F(SubstraitJsonTest, CamelCaseJsonMatchesBinary) { expect_matches_binary(plan_json()); }

TEST_F(SubstraitJsonTest, SnakeCaseJsonMatchesBinary)
{
  expect_matches_binary(plan_json(/*use_proto_field_names=*/true));
}

TEST_F(SubstraitJsonTest, UnknownFieldIsRejected)
{
  auto json = plan_json();
  ASSERT_EQ(json.front(), '{');
  json.insert(1, R"("notAPlanField": 1,)");

  gqe::substrait_parser parser(&catalog, &opms);
  EXPECT_THAT(
    [&] { parser.from_json(json); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("notAPlanField")));
}

TEST_F(SubstraitJsonTest, MalformedJsonIsRejected)
{
  auto json = plan_json();
  json.pop_back();

  gqe::substrait_parser parser(&catalog, &opms);
  EXPECT_THROW(parser.from_json(json), std::invalid_argument);
}
