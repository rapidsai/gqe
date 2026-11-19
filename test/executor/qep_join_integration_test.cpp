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

#include "qep_join_integration_test_util.hpp"

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/bounded_hash_aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/shapes/write_statistics.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gqe_test {

static std::string join_configuration_name(qep_join_test_configuration const& configuration)
{
  return std::string{configuration.is_perfect_hashing ? "Perfect" : "Ordinary"} +
         (configuration.is_hash_map_cache_enabled ? "Cached" : "Uncached") +
         (configuration.policy == physical::broadcast_policy::left ? "BuildLeft" : "BuildRight");
}

/** @brief Exercises common inner-join behavior across supported implementations and options. */
class QepJoinIntegrationTest : public QepJoinIntegrationTestBase,
                               public ::testing::WithParamInterface<qep_join_test_configuration> {};

INSTANTIATE_TEST_SUITE_P(
  JoinImplementations,
  QepJoinIntegrationTest,
  ::testing::Values(qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = false,
                                                .is_hash_map_cache_enabled = true},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = false,
                                                .is_hash_map_cache_enabled = true},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = false,
                                                .is_hash_map_cache_enabled = false},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = true,
                                                .is_hash_map_cache_enabled = false},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = true,
                                                .is_hash_map_cache_enabled = false}),
  [](auto const& info) { return join_configuration_name(info.param); });

TEST_P(QepJoinIntegrationTest, ExecutesPartitionedMixedProjection)
{
  auto result = execute_join(GetParam(), "mixed_projection");
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                          int64_column_wrapper{200, 100, 400}.release(),
                                          int64_column_wrapper{11, 22, 33}.release(),
                                          int64_column_wrapper{2000, 1000, 4000}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
}

TEST_P(QepJoinIntegrationTest, CountOnlyOutputPreservesMatchCount)
{
  auto result =
    execute_join(GetParam(), "count_only", qep_join_execution_options{.projection_indices = {}});
  EXPECT_EQ(qep::try_row_count(result), 3);
}

TEST_P(QepJoinIntegrationTest, MaskedChildrenCrossNormalizationBoundaries)
{
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{10, 20, 30}.release(),
                            int64_column_wrapper{1, 2, 3}.release(),
                            int64_column_wrapper{{100, 10, 0}, {true, true, false}}.release()));
  std::vector<std::unique_ptr<cudf::table>> right_partitions;
  right_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{10, 20, 30}.release(),
                            int64_column_wrapper{{100, 10, 0}, {true, true, false}}.release(),
                            int64_column_wrapper{1, 2, 3}.release()));

  auto result = execute_join(
    GetParam(),
    "masked",
    qep_join_execution_options{.left_partitions  = std::move(left_partitions),
                               .right_partitions = std::move(right_partitions),
                               .left_filter      = std::make_unique<gqe::less_expression>(
                                 std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(2)),
                               .right_filter = std::make_unique<gqe::less_expression>(
                                 std::make_shared<gqe::column_reference_expression>(0),
                                 std::make_shared<gqe::column_reference_expression>(1))});
  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = make_table_from_columns(int64_column_wrapper{{100}, {true}}.release(),
                                          int64_column_wrapper{10}.release(),
                                          int64_column_wrapper{{100}, {true}}.release(),
                                          int64_column_wrapper{{100}, {true}}.release());
  // Join implementations may retain or omit an all-valid null mask.
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*table, expected->view());
}

TEST_P(QepJoinIntegrationTest, EmptyBuildPublishesTypedResult)
{
  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  std::vector<std::unique_ptr<cudf::table>> empty_build;
  empty_build.push_back(make_table_from_columns(cudf::make_empty_column(int64_type),
                                                cudf::make_empty_column(int64_type),
                                                cudf::make_empty_column(int64_type)));

  qep_join_execution_options options;
  if (GetParam().policy == physical::broadcast_policy::left) {
    options.left_partitions = std::move(empty_build);
  } else {
    options.right_partitions = std::move(empty_build);
  }

  auto result = execute_join(GetParam(), "empty_build", std::move(options));
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 4);
  EXPECT_EQ(table->num_rows(), 0);
  for (auto const& column : *table) {
    EXPECT_EQ(column.type(), int64_type);
  }
}

/** @brief Exercises scenarios with explicit join configurations. */
class QepJoinSpecificIntegrationTest : public QepJoinIntegrationTestBase {
 protected:
  void expect_inner_result(physical::broadcast_policy policy,
                           bool is_hash_map_cache_enabled,
                           std::string_view name,
                           gqe::unique_keys_policy unique_keys = gqe::unique_keys_policy::none)
  {
    auto result = execute_join(
      qep_join_test_configuration{.policy                    = policy,
                                  .join_type                 = gqe::join_type_type::inner,
                                  .is_perfect_hashing        = false,
                                  .is_hash_map_cache_enabled = is_hash_map_cache_enabled,
                                  .unique_keys               = unique_keys},
      name);
    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                            int64_column_wrapper{200, 100, 400}.release(),
                                            int64_column_wrapper{11, 22, 33}.release(),
                                            int64_column_wrapper{2000, 1000, 4000}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
};

TEST_F(QepJoinSpecificIntegrationTest, UniqueKeysSupportEitherBuildSideAndCacheSetting)
{
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (bool cache : {false, true}) {
      for (auto unique_keys : {gqe::unique_keys_policy::left, gqe::unique_keys_policy::right}) {
        SCOPED_TRACE(::testing::Message()
                     << "build_side="
                     << (policy == physical::broadcast_policy::left ? "left" : "right")
                     << ", cache=" << cache << ", unique_side="
                     << (unique_keys == gqe::unique_keys_policy::left ? "left" : "right"));
        expect_inner_result(policy, cache, "unique_keys", unique_keys);
      }
    }
  }
}

TEST_F(QepJoinSpecificIntegrationTest, UniqueProbeSideDoesNotDiscardDuplicateBuildMatches)
{
  for (auto policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    for (bool cache : {false, true}) {
      bool const build_is_left = policy == physical::broadcast_policy::left;
      SCOPED_TRACE(::testing::Message()
                   << "build_side=" << (build_is_left ? "left" : "right") << ", cache=" << cache
                   << ", unique_side=" << (build_is_left ? "right" : "left"));
      std::vector<std::unique_ptr<cudf::table>> left;
      std::vector<std::unique_ptr<cudf::table>> right;
      if (build_is_left) {
        left.push_back(make_table_from_columns(int64_column_wrapper{10, 20}.release(),
                                               int64_column_wrapper{1, 1}.release(),
                                               int64_column_wrapper{0, 0}.release()));
        right.push_back(make_table_from_columns(int64_column_wrapper{100}.release(),
                                                int64_column_wrapper{0}.release(),
                                                int64_column_wrapper{1}.release()));
      } else {
        left.push_back(make_table_from_columns(int64_column_wrapper{100}.release(),
                                               int64_column_wrapper{1}.release(),
                                               int64_column_wrapper{0}.release()));
        right.push_back(make_table_from_columns(int64_column_wrapper{10, 20}.release(),
                                                int64_column_wrapper{0, 0}.release(),
                                                int64_column_wrapper{1, 1}.release()));
      }
      auto result = execute_join(
        qep_join_test_configuration{.policy                    = policy,
                                    .join_type                 = gqe::join_type_type::inner,
                                    .is_perfect_hashing        = false,
                                    .is_hash_map_cache_enabled = cache,
                                    .unique_keys = build_is_left ? gqe::unique_keys_policy::right
                                                                 : gqe::unique_keys_policy::left},
        "unique_probe_side",
        qep_join_execution_options{.left_partitions    = std::move(left),
                                   .right_partitions   = std::move(right),
                                   .projection_indices = {0, 3}});
      auto sorted   = cudf::sort(*qep::to_table_view(result));
      auto expected = build_is_left
                        ? make_table_from_columns(int64_column_wrapper{10, 20}.release(),
                                                  int64_column_wrapper{100, 100}.release())
                        : make_table_from_columns(int64_column_wrapper{100, 100}.release(),
                                                  int64_column_wrapper{10, 20}.release());
      CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
    }
  }
}

TEST_F(QepJoinSpecificIntegrationTest, DisabledUniqueKeysPreserveDuplicateBuildMatches)
{
  get_query_ctx()->parameters.join_use_unique_keys = false;
  for (bool cache : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "cache=" << cache);
    std::vector<std::unique_ptr<cudf::table>> left;
    std::vector<std::unique_ptr<cudf::table>> right;
    left.push_back(make_table_from_columns(int64_column_wrapper{10, 20}.release(),
                                           int64_column_wrapper{1, 1}.release(),
                                           int64_column_wrapper{0, 0}.release()));
    right.push_back(make_table_from_columns(int64_column_wrapper{100}.release(),
                                            int64_column_wrapper{0}.release(),
                                            int64_column_wrapper{1}.release()));
    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                               .join_type = gqe::join_type_type::inner,
                                               .is_perfect_hashing        = false,
                                               .is_hash_map_cache_enabled = cache,
                                               .unique_keys = gqe::unique_keys_policy::left},
                   "disabled_unique_keys",
                   qep_join_execution_options{.left_partitions    = std::move(left),
                                              .right_partitions   = std::move(right),
                                              .projection_indices = {0, 3}});
    auto sorted   = cudf::sort(*qep::to_table_view(result));
    auto expected = make_table_from_columns(int64_column_wrapper{10, 20}.release(),
                                            int64_column_wrapper{100, 100}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, CompositeKeysUseEveryEquality)
{
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{110, 120, 210, 220}.release(),
                            int64_column_wrapper{1, 1, 2, 2}.release(),
                            int64_column_wrapper{10, 20, 10, 20}.release()));
  std::vector<std::unique_ptr<cudf::table>> right_partitions;
  right_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{1010, 1020, 2010, 2020}.release(),
                            int64_column_wrapper{10, 20, 10, 20}.release(),
                            int64_column_wrapper{1, 1, 2, 2}.release()));

  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(5)),
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(2),
                                            std::make_shared<gqe::column_reference_expression>(4)));
  auto result =
    execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                             .join_type = gqe::join_type_type::inner,
                                             .is_perfect_hashing        = false,
                                             .is_hash_map_cache_enabled = true},
                 "composite_keys",
                 qep_join_execution_options{.left_partitions    = std::move(left_partitions),
                                            .right_partitions   = std::move(right_partitions),
                                            .projection_indices = {0, 3},
                                            .join_condition     = std::move(join_condition)});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(int64_column_wrapper{110, 120, 210, 220}.release(),
                                          int64_column_wrapper{1010, 1020, 2010, 2020}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
}

TEST_F(QepJoinSpecificIntegrationTest, LeftJoinPreservesUnmatchedRows)
{
  auto result =
    execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                             .join_type = gqe::join_type_type::left,
                                             .is_perfect_hashing        = false,
                                             .is_hash_map_cache_enabled = true},
                 "left_join");
  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(
    int64_column_wrapper{{0, 11, 22, 33}, {false, true, true, true}}.release(),
    int64_column_wrapper{300, 200, 100, 400}.release(),
    int64_column_wrapper{{0, 11, 22, 33}, {false, true, true, true}}.release(),
    int64_column_wrapper{3000, 2000, 1000, 4000}.release());
  auto sorted_expected = cudf::sort(expected->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), sorted_expected->view());
}

TEST_F(QepJoinSpecificIntegrationTest, UniqueStreamingSideOverridesBroadcastSide)
{
  for (auto const policy : {physical::broadcast_policy::left, physical::broadcast_policy::right}) {
    bool const materialized_is_right = policy == physical::broadcast_policy::right;
    SCOPED_TRACE(materialized_is_right ? "broadcast_right" : "broadcast_left");
    std::vector<std::unique_ptr<cudf::table>> left_partitions;
    left_partitions.push_back(make_table_from_columns(int64_column_wrapper{100}.release(),
                                                      int64_column_wrapper{1}.release(),
                                                      int64_column_wrapper{1000}.release()));
    left_partitions.push_back(
      materialized_is_right ? make_table_from_columns(int64_column_wrapper{200}.release(),
                                                      int64_column_wrapper{2}.release(),
                                                      int64_column_wrapper{2000}.release())
                            : make_table_from_columns(int64_column_wrapper{200, 201}.release(),
                                                      int64_column_wrapper{2, 2}.release(),
                                                      int64_column_wrapper{2000, 2010}.release()));
    std::vector<std::unique_ptr<cudf::table>> right_partitions;
    right_partitions.push_back(make_table_from_columns(int64_column_wrapper{10}.release(),
                                                       int64_column_wrapper{11}.release(),
                                                       int64_column_wrapper{1}.release()));
    right_partitions.push_back(materialized_is_right
                                 ? make_table_from_columns(int64_column_wrapper{20, 21}.release(),
                                                           int64_column_wrapper{22, 23}.release(),
                                                           int64_column_wrapper{2, 2}.release())
                                 : make_table_from_columns(int64_column_wrapper{20}.release(),
                                                           int64_column_wrapper{22}.release(),
                                                           int64_column_wrapper{2}.release()));

    auto result = execute_join(
      qep_join_test_configuration{.policy                    = policy,
                                  .join_type                 = gqe::join_type_type::inner,
                                  .is_perfect_hashing        = true,
                                  .is_hash_map_cache_enabled = false,
                                  .unique_keys               = materialized_is_right
                                                                 ? gqe::unique_keys_policy::left
                                                                 : gqe::unique_keys_policy::right},
      materialized_is_right ? "unique_left_broadcast_right" : "unique_right_broadcast_left",
      qep_join_execution_options{.left_partitions  = std::move(left_partitions),
                                 .right_partitions = std::move(right_partitions)});
    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = make_table_from_columns(
      int64_column_wrapper{11, 22, materialized_is_right ? 23 : 22}.release(),
      int64_column_wrapper{100, 200, materialized_is_right ? 200 : 201}.release(),
      int64_column_wrapper{11, 22, materialized_is_right ? 23 : 22}.release(),
      int64_column_wrapper{1000, 2000, materialized_is_right ? 2000 : 2010}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, JoinWriterReportsExpectedRowCount)
{
  auto result = execute_join(
    qep_join_test_configuration{.policy                    = physical::broadcast_policy::right,
                                .join_type                 = gqe::join_type_type::inner,
                                .is_perfect_hashing        = false,
                                .is_hash_map_cache_enabled = true},
    "writer",
    qep_join_execution_options{.transform_root = [](std::shared_ptr<physical::relation> join,
                                                    std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>) {
      return std::make_shared<physical::write_relation>(
        join, std::vector<std::string>{"a", "b", "c", "d"}, "output");
    }});

  auto const statistics = qep::try_write_statistics(result);
  ASSERT_TRUE(statistics.has_value());
  EXPECT_EQ(statistics->num_rows, 3);
}

/** @brief Exercises shared aggregate/probe input with both join families and empty input. */
class QepDiamondIntegrationTest : public QepJoinIntegrationTestBase,
                                  public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(QepDiamondIntegrationTest, ReplaysSharedRowsAfterMinimumAndPreservesTies)
{
  auto const [perfect_hashing, empty_input]                = GetParam();
  get_query_ctx()->parameters.join_use_perfect_hash        = perfect_hashing;
  get_query_ctx()->parameters.aggregation_use_perfect_hash = perfect_hashing;
  std::vector<std::string> const columns{"part", "cost", "supplier"};
  std::vector<std::unique_ptr<cudf::table>> partitions;
  partitions.push_back(make_table_from_columns(int64_column_wrapper{1, 1, 2}.release(),
                                               int64_column_wrapper{9, 5, 7}.release(),
                                               int64_column_wrapper{10, 11, 20}.release()));
  partitions.push_back(make_table_from_columns(int64_column_wrapper{1, 2, 2}.release(),
                                               int64_column_wrapper{5, 8, 7}.release(),
                                               int64_column_wrapper{12, 21, 22}.release()));
  auto table = write_partitions("diamond", columns, std::move(partitions));
  join_table_provider provider(table, table, nullptr);
  std::shared_ptr<physical::relation> input = make_join_read("left", columns);
  if (empty_input) {
    input = std::make_shared<physical::filter_relation>(
      input,
      std::vector<std::shared_ptr<physical::relation>>{},
      std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                             std::make_shared<gqe::column_reference_expression>(0)),
      std::vector<cudf::size_type>{0, 1, 2});
  }
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
  values.emplace_back(cudf::aggregation::MIN,
                      std::make_unique<gqe::column_reference_expression>(1));
  auto minimum = std::make_shared<physical::concatenate_aggregate_relation>(
    input,
    std::vector<std::shared_ptr<physical::relation>>{},
    std::move(keys),
    std::move(values),
    nullptr,
    true);
  auto condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(3)),
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(4)));
  physical::broadcast_join_relation join{input,
                                         minimum,
                                         {},
                                         gqe::join_type_type::inner,
                                         std::move(condition),
                                         {0, 1, 2},
                                         physical::broadcast_policy::right,
                                         gqe::unique_keys_policy::right,
                                         true};
  auto identified_plan = gqe_test::make_borrowed_identified_plan(&join);
  auto lowered =
    qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  gqe::execute_task_graph_single_gpu(ctx_ref, graph.get());
  ASSERT_EQ(graph->root_tasks.size(), 1);
  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto view = qep::to_table_view(*result);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->num_columns(), 3);
  if (empty_input) {
    EXPECT_EQ(view->num_rows(), 0);
  } else {
    auto sorted   = cudf::sort(*view);
    auto expected = make_table_from_columns(int64_column_wrapper{1, 1, 2, 2}.release(),
                                            int64_column_wrapper{5, 5, 7, 7}.release(),
                                            int64_column_wrapper{11, 12, 20, 22}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
}

INSTANTIATE_TEST_SUITE_P(HashAndPerfectHash,
                         QepDiamondIntegrationTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

// These execution scenarios check semantic results. Gather placement is checked separately
// by the physical-lowering tests.
class QepJoinSuccessorIntegrationTest
  : public QepJoinIntegrationTestBase,
    public ::testing::WithParamInterface<qep_join_test_configuration> {};

static auto const successor_configurations =
  ::testing::Values(qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = false,
                                                .is_hash_map_cache_enabled = true},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = false,
                                                .is_hash_map_cache_enabled = true},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = true,
                                                .is_hash_map_cache_enabled = false},
                    qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                .join_type = gqe::join_type_type::inner,
                                                .is_perfect_hashing        = true,
                                                .is_hash_map_cache_enabled = false});

INSTANTIATE_TEST_SUITE_P(JoinImplementations,
                         QepJoinSuccessorIntegrationTest,
                         successor_configurations,
                         [](auto const& info) { return join_configuration_name(info.param); });

class QepChainedJoinIntegrationTest
  : public QepJoinIntegrationTestBase,
    public ::testing::WithParamInterface<std::tuple<qep_join_test_configuration, bool>> {};

INSTANTIATE_TEST_SUITE_P(JoinImplementations,
                         QepChainedJoinIntegrationTest,
                         ::testing::Combine(successor_configurations, ::testing::Bool()),
                         [](auto const& info) {
                           return "First" + join_configuration_name(std::get<0>(info.param)) +
                                  (std::get<1>(info.param) ? "SecondPerfect" : "SecondOrdinary");
                         });

TEST_P(QepJoinSuccessorIntegrationTest, ProjectReduceChainNormalizesFinalResult)
{
  auto result = execute_join(
    GetParam(),
    "project_reduce",
    qep_join_execution_options{.transform_root = [](std::shared_ptr<physical::relation> join,
                                                    std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>) {
      std::vector<std::unique_ptr<gqe::expression>> project_expressions;
      project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
      project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
      auto projected = std::make_shared<physical::project_relation>(
        join, std::vector<std::shared_ptr<physical::relation>>{}, std::move(project_expressions));
      std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
      values.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(0));
      values.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(1));
      return std::make_shared<physical::concatenate_aggregate_relation>(
        projected,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::vector<std::unique_ptr<gqe::expression>>{},
        std::move(values));
    }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = make_table_from_columns(int64_column_wrapper{66}.release(),
                                          int64_column_wrapper{700}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
}

TEST_P(QepJoinSuccessorIntegrationTest, BoundedAggregateConsumesIndirectJoinOutput)
{
  auto result = execute_join(
    GetParam(),
    "bounded_aggregate",
    qep_join_execution_options{.transform_root = [](std::shared_ptr<physical::relation> join,
                                                    std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>) {
      return std::make_shared<physical::bounded_hash_aggregate_relation>(
        join,
        std::vector<cudf::size_type>{0},
        std::vector<physical::bounded_hash_aggregate_relation::value_type>{
          {cudf::aggregation::SUM, 1, gqe::nullability::nullable}},
        /*group_count_bound=*/16);
    }});
  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                          int64_column_wrapper{200, 100, 400}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
}

TEST_P(QepJoinSuccessorIntegrationTest, FilterProjectReduceChainNormalizesFinalResult)
{
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{100, 200, 400}.release(),
                            int64_column_wrapper{1, 2, 4}.release(),
                            int64_column_wrapper{1000, 20, 4000}.release()));
  auto result = execute_join(
    GetParam(),
    "filter_project_reduce",
    qep_join_execution_options{
      .left_partitions = std::move(left_partitions),
      .transform_root  = [](std::shared_ptr<physical::relation> join,
                           std::shared_ptr<physical::relation>,
                           std::shared_ptr<physical::relation>) {
        auto filtered = std::make_shared<physical::filter_relation>(
          join,
          std::vector<std::shared_ptr<physical::relation>>{},
          std::make_unique<gqe::less_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(3)),
          std::vector<cudf::size_type>{0, 1, 2, 3});
        std::vector<std::unique_ptr<gqe::expression>> project_expressions;
        project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
        project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
        auto projected = std::make_shared<physical::project_relation>(
          filtered,
          std::vector<std::shared_ptr<physical::relation>>{},
          std::move(project_expressions));
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
        values.emplace_back(cudf::aggregation::SUM,
                            std::make_unique<gqe::column_reference_expression>(1));
        values.emplace_back(cudf::aggregation::COUNT_ALL,
                            std::make_unique<gqe::column_reference_expression>(0));
        return std::make_shared<physical::concatenate_aggregate_relation>(
          projected,
          std::vector<std::shared_ptr<physical::relation>>{},
          std::vector<std::unique_ptr<gqe::expression>>{},
          std::move(values));
      }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected =
    make_table_from_columns(int64_column_wrapper{500}.release(), int64_column_wrapper{2}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
}

TEST_P(QepJoinSuccessorIntegrationTest, FilterReduceConsumesMixedDirectIndirectJoinOutput)
{
  // Three of the 20 left rows match, so the left source falls below the threshold and is
  // gathered. Every right row matches, so the right source stays indirect. Both hold whichever
  // side is the build, and also if a read splits an input in two.
  get_query_ctx()->parameters.use_late_materialization       = true;
  get_query_ctx()->parameters.join_materialization_threshold = 0.5;

  std::vector<int64_t> left_values(20);
  std::vector<int64_t> left_keys(20);
  std::vector<int64_t> left_payloads(20);
  for (int64_t i = 0; i < 20; ++i) {
    left_values[i]   = i + 1;
    left_keys[i]     = i + 1;
    left_payloads[i] = 100 * (i + 1);
  }
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(make_table_from_columns(
    int64_column_wrapper(left_values.begin(), left_values.end()).release(),
    int64_column_wrapper(left_keys.begin(), left_keys.end()).release(),
    int64_column_wrapper(left_payloads.begin(), left_payloads.end()).release()));
  std::vector<std::unique_ptr<cudf::table>> right_partitions;
  right_partitions.push_back(make_table_from_columns(int64_column_wrapper{7, 8, 9}.release(),
                                                     int64_column_wrapper{150, 50, 400}.release(),
                                                     int64_column_wrapper{1, 2, 3}.release()));

  auto result = execute_join(
    GetParam(),
    "mixed_direct_indirect",
    qep_join_execution_options{
      .left_partitions    = std::move(left_partitions),
      .right_partitions   = std::move(right_partitions),
      .projection_indices = {0, 2, 4},
      .transform_root     = [](std::shared_ptr<physical::relation> join,
                           std::shared_ptr<physical::relation>,
                           std::shared_ptr<physical::relation>) {
        // left_payload < right_payload keeps join keys 1 and 3.
        auto filtered = std::make_shared<physical::filter_relation>(
          join,
          std::vector<std::shared_ptr<physical::relation>>{},
          std::make_unique<gqe::less_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(2)),
          std::vector<cudf::size_type>{0, 1, 2});
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
        values.emplace_back(cudf::aggregation::SUM,
                            std::make_unique<gqe::column_reference_expression>(0));
        values.emplace_back(cudf::aggregation::SUM,
                            std::make_unique<gqe::column_reference_expression>(2));
        return std::make_shared<physical::concatenate_aggregate_relation>(
          filtered,
          std::vector<std::shared_ptr<physical::relation>>{},
          std::vector<std::unique_ptr<gqe::expression>>{},
          std::move(values));
      }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected =
    make_table_from_columns(int64_column_wrapper{4}.release(), int64_column_wrapper{550}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
}

TEST_P(QepJoinSuccessorIntegrationTest, ProjectGroupedAggregateChainNormalizesFinalResult)
{
  auto result = execute_join(
    GetParam(),
    "project_perfect_aggregate",
    qep_join_execution_options{.transform_root = [](std::shared_ptr<physical::relation> join,
                                                    std::shared_ptr<physical::relation>,
                                                    std::shared_ptr<physical::relation>) {
      std::vector<std::unique_ptr<gqe::expression>> project_expressions;
      project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
      project_expressions.push_back(std::make_unique<gqe::column_reference_expression>(1));
      auto projected = std::make_shared<physical::project_relation>(
        join, std::vector<std::shared_ptr<physical::relation>>{}, std::move(project_expressions));
      std::vector<std::unique_ptr<gqe::expression>> keys;
      keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
      std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>> values;
      values.emplace_back(cudf::aggregation::SUM,
                          std::make_unique<gqe::column_reference_expression>(1));
      return std::make_shared<physical::concatenate_aggregate_relation>(
        projected,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::move(keys),
        std::move(values),
        nullptr,
        /*perfect_hashing=*/true);
    }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted   = cudf::sort(*table);
  auto expected = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                          int64_column_wrapper{200, 100, 400}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
}

TEST_P(QepChainedJoinIntegrationTest, SuccessorJoinConsumesIndirectInput)
{
  auto const& [configuration, second_is_perfect] = GetParam();
  auto result                                    = execute_join(
    configuration,
    "successor_join",
    qep_join_execution_options{
                                         .transform_root = [this, second_is_perfect](std::shared_ptr<physical::relation> join,
                                                  std::shared_ptr<physical::relation> left,
                                                  std::shared_ptr<physical::relation>) {
        // The runtime flag permits perfect hashing for either join. Each physical join
        // still selects its own implementation.
        get_query_ctx()->parameters.join_use_perfect_hash |= second_is_perfect;
        return std::make_shared<physical::broadcast_join_relation>(
          join,
          left,
          std::vector<std::shared_ptr<physical::relation>>{},
          gqe::join_type_type::inner,
          std::make_unique<gqe::equal_expression>(
            std::make_shared<gqe::column_reference_expression>(1),
            std::make_shared<gqe::column_reference_expression>(4)),
          std::vector<cudf::size_type>{0, 1, 4},
          physical::broadcast_policy::right,
          second_is_perfect ? gqe::unique_keys_policy::right : gqe::unique_keys_policy::none,
          second_is_perfect);
      }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto sorted          = cudf::sort(*table);
  auto expected        = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                          int64_column_wrapper{200, 100, 400}.release(),
                                          int64_column_wrapper{200, 100, 400}.release());
  auto sorted_expected = cudf::sort(expected->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), sorted_expected->view());
}

TEST_P(QepJoinSuccessorIntegrationTest, FetchTopKChainNormalizesFinalResult)
{
  // Identical projected rows make the Fetch subset independent of join output order.
  std::vector<std::unique_ptr<cudf::table>> left_partitions;
  left_partitions.push_back(
    make_table_from_columns(int64_column_wrapper{400, 400, 400}.release(),
                            int64_column_wrapper{1, 2, 3}.release(),
                            int64_column_wrapper{4000, 4000, 4000}.release()));
  std::vector<std::unique_ptr<cudf::table>> right_partitions;
  right_partitions.push_back(make_table_from_columns(int64_column_wrapper{30, 30, 30}.release(),
                                                     int64_column_wrapper{33, 33, 33}.release(),
                                                     int64_column_wrapper{1, 2, 3}.release()));
  auto result = execute_join(
    GetParam(),
    "fetch_top_k",
    qep_join_execution_options{
      .left_partitions  = std::move(left_partitions),
      .right_partitions = std::move(right_partitions),
      .transform_root   = [](std::shared_ptr<physical::relation> join,
                           std::shared_ptr<physical::relation>,
                           std::shared_ptr<physical::relation>) {
        auto fetched = std::make_shared<physical::fetch_relation>(join, /*offset=*/0, /*count=*/2);
        return std::make_shared<physical::top_k_relation>(
          fetched, 0, cudf::order::DESCENDING, cudf::null_order::AFTER, 0, 3);
      }});

  auto table = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = make_table_from_columns(int64_column_wrapper{33, 33}.release(),
                                          int64_column_wrapper{400, 400}.release(),
                                          int64_column_wrapper{33, 33}.release(),
                                          int64_column_wrapper{4000, 4000}.release());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*table, expected->view());
}

static std::unique_ptr<gqe::expression> mixed_condition(std::unique_ptr<gqe::expression> residual)
{
  return std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(5)),
    std::shared_ptr<gqe::expression>{std::move(residual)});
}

TEST_F(QepJoinSpecificIntegrationTest, MixedInnerBuildLeftFiltersMaskedMultiChunkCandidates)
{
  for (bool const is_perfect_hashing : {false, true}) {
    SCOPED_TRACE(is_perfect_hashing);
    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::left,
                                               .join_type = gqe::join_type_type::inner,
                                               .is_perfect_hashing        = is_perfect_hashing,
                                               .is_hash_map_cache_enabled = false},
                   "mixed_build_left",
                   qep_join_execution_options{
                     .left_partitions    = make_left_partitions(),
                     .right_partitions   = make_right_partitions(),
                     .projection_indices = {4, 0, 4, 2},
                     .left_filter        = std::make_unique<gqe::less_expression>(
                       std::make_shared<gqe::column_reference_expression>(0),
                       std::make_shared<gqe::column_reference_expression>(2)),
                     .right_filter   = nullptr,
                     .join_condition = mixed_condition(std::make_unique<gqe::greater_expression>(
                       std::make_shared<gqe::column_reference_expression>(2),
                       std::make_shared<gqe::column_reference_expression>(4)))});

    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = make_table_from_columns(int64_column_wrapper{11, 22, 33}.release(),
                                            int64_column_wrapper{200, 100, 400}.release(),
                                            int64_column_wrapper{11, 22, 33}.release(),
                                            int64_column_wrapper{2000, 1000, 4000}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedInnerBuildRightSupportsOneSidedProjection)
{
  for (auto const unique_keys : {gqe::unique_keys_policy::none,
                                 gqe::unique_keys_policy::right,
                                 gqe::unique_keys_policy::left}) {
    SCOPED_TRACE(unique_keys == gqe::unique_keys_policy::none    ? "RegularHash"
                 : unique_keys == gqe::unique_keys_policy::right ? "PerfectHashUniqueRight"
                                                                 : "PerfectHashUniqueLeft");
    // Each input appends col0 + col2. Combined indices 3 and 7 are the sums,
    // and index 6 is the right input's equality key.
    auto join_condition = std::make_unique<gqe::logical_and_expression>(
      std::make_shared<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::column_reference_expression>(6)),
      std::make_shared<gqe::greater_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::column_reference_expression>(7)));
    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                               .join_type = gqe::join_type_type::inner,
                                               .is_perfect_hashing =
                                                 unique_keys != gqe::unique_keys_policy::none,
                                               .is_hash_map_cache_enabled = false,
                                               .unique_keys               = unique_keys},
                   "mixed_build_right",
                   qep_join_execution_options{.left_partitions          = make_left_partitions(),
                                              .right_partitions         = make_right_partitions(),
                                              .has_input_sum_projection = true,
                                              .projection_indices       = {0},
                                              .left_filter              = nullptr,
                                              .right_filter             = nullptr,
                                              .join_condition = std::move(join_condition)});

    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = make_table_from_columns(int64_column_wrapper{100, 200, 400}.release());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedInnerSupportsOneSidedIsNullResidualAndCountOnlyOutput)
{
  for (bool const is_perfect_hashing : {false, true}) {
    SCOPED_TRACE(is_perfect_hashing);
    std::vector<std::unique_ptr<cudf::table>> right_partitions;
    right_partitions.push_back(
      make_table_from_columns(int64_column_wrapper{{10, 30}, {true, false}}.release(),
                              int64_column_wrapper{11, 33}.release(),
                              int64_column_wrapper{2, 4}.release()));
    right_partitions.push_back(
      make_table_from_columns(int64_column_wrapper{{20, 40}, {false, true}}.release(),
                              int64_column_wrapper{22, 44}.release(),
                              int64_column_wrapper{1, 5}.release()));

    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                               .join_type = gqe::join_type_type::inner,
                                               .is_perfect_hashing        = is_perfect_hashing,
                                               .is_hash_map_cache_enabled = false},
                   "mixed_is_null_count_only",
                   qep_join_execution_options{
                     .left_partitions    = make_left_partitions(),
                     .right_partitions   = std::move(right_partitions),
                     .projection_indices = {},
                     .left_filter        = nullptr,
                     .right_filter       = nullptr,
                     .join_condition = mixed_condition(std::make_unique<gqe::is_null_expression>(
                       std::make_shared<gqe::column_reference_expression>(3)))});

    EXPECT_EQ(qep::try_row_count(result), 2);
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedJoinsFilterInheritedSources)
{
  for (auto const [join_type, has_match] : {std::pair{gqe::join_type_type::inner, false},
                                            std::pair{gqe::join_type_type::inner, true},
                                            std::pair{gqe::join_type_type::left, false},
                                            std::pair{gqe::join_type_type::left, true}}) {
    std::vector<std::unique_ptr<cudf::table>> left_partitions;
    left_partitions.push_back(make_table_from_columns(int64_column_wrapper{10}.release(),
                                                      int64_column_wrapper{1}.release(),
                                                      int64_column_wrapper{100}.release()));
    std::vector<std::unique_ptr<cudf::table>> right_partitions;
    right_partitions.push_back(has_match
                                 ? make_table_from_columns(int64_column_wrapper{20}.release(),
                                                           int64_column_wrapper{200}.release(),
                                                           int64_column_wrapper{1}.release())
                                 : make_table_from_columns(int64_column_wrapper{}.release(),
                                                           int64_column_wrapper{}.release(),
                                                           int64_column_wrapper{}.release()));
    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                               .join_type = gqe::join_type_type::left,
                                               .is_perfect_hashing        = false,
                                               .is_hash_map_cache_enabled = true},
                   has_match ? "inherited_nullify_matched" : "inherited_nullify_empty",
                   qep_join_execution_options{
                     .left_partitions    = std::move(left_partitions),
                     .right_partitions   = std::move(right_partitions),
                     .projection_indices = {1, 3},
                     .transform_root     = [join_type](std::shared_ptr<physical::relation> join,
                                                   std::shared_ptr<physical::relation>,
                                                   std::shared_ptr<physical::relation>) {
                       // Read A again as C and retain both inherited sources A and B.
                       auto c = make_join_read("left", {"left_value", "join_key", "left_payload"});
                       auto condition = std::make_unique<gqe::logical_and_expression>(
                         std::make_shared<gqe::equal_expression>(
                           std::make_shared<gqe::column_reference_expression>(0),
                           std::make_shared<gqe::column_reference_expression>(3)),
                         std::make_shared<gqe::is_null_expression>(
                           std::make_shared<gqe::column_reference_expression>(1)));
                       return std::make_shared<physical::broadcast_join_relation>(
                         join,
                         c,
                         std::vector<std::shared_ptr<physical::relation>>{},
                         join_type,
                         std::move(condition),
                         std::vector<cudf::size_type>{0, 1, 3},
                         physical::broadcast_policy::right,
                         gqe::unique_keys_policy::none,
                         false,
                         nullptr,
                         nullptr,
                         true);
                     }});
    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto expected = has_match && join_type == gqe::join_type_type::inner
                      ? make_table_from_columns(int64_column_wrapper{}.release(),
                                                int64_column_wrapper{}.release(),
                                                int64_column_wrapper{}.release())
                      : make_table_from_columns(int64_column_wrapper{1}.release(),
                                                int64_column_wrapper{{20}, {has_match}}.release(),
                                                int64_column_wrapper{{1}, {!has_match}}.release());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*table, expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedLeftJoinPreservesRowsWhenResidualFails)
{
  for (bool const is_perfect_hashing : {false, true}) {
    SCOPED_TRACE(is_perfect_hashing);
    std::vector<std::unique_ptr<cudf::table>> right_partitions;
    right_partitions.push_back(make_table_from_columns(int64_column_wrapper{250, 450}.release(),
                                                       int64_column_wrapper{11, 33}.release(),
                                                       int64_column_wrapper{2, 4}.release()));
    right_partitions.push_back(make_table_from_columns(int64_column_wrapper{50, 550}.release(),
                                                       int64_column_wrapper{22, 44}.release(),
                                                       int64_column_wrapper{1, 5}.release()));
    auto result =
      execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                               .join_type = gqe::join_type_type::left,
                                               .is_perfect_hashing        = is_perfect_hashing,
                                               .is_hash_map_cache_enabled = false},
                   "mixed_left",
                   qep_join_execution_options{
                     .left_partitions    = make_left_partitions(),
                     .right_partitions   = std::move(right_partitions),
                     .projection_indices = {0, 3},
                     .left_filter        = nullptr,
                     .right_filter       = nullptr,
                     .join_condition     = mixed_condition(std::make_unique<gqe::less_expression>(
                       std::make_shared<gqe::column_reference_expression>(0),
                       std::make_shared<gqe::column_reference_expression>(3)))});

    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted   = cudf::sort(*table);
    auto expected = make_table_from_columns(
      int64_column_wrapper{100, 200, 300, 400}.release(),
      int64_column_wrapper{{0, 250, 0, 450}, {false, true, false, true}}.release());
    auto sorted_expected = cudf::sort(expected->view());
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), sorted_expected->view());
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedSemiAntiStreamedModesPreserveRepeatedLeftProjections)
{
  // The left fixture has (value, key) rows (100, 1), (200, 2), (300, 3), and (400, 4).
  // A match requires equal keys and left.value < right.value. Duplicate right matches
  // must not duplicate semi output; key 3 has no match and belongs only in anti output.
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (bool count : {false, true}) {
      std::vector<std::unique_ptr<cudf::table>> right;
      right.push_back(make_table_from_columns(int64_column_wrapper{200, 200}.release(),
                                              int64_column_wrapper{0, 0}.release(),
                                              int64_column_wrapper{1, 1}.release()));
      right.push_back(make_table_from_columns(int64_column_wrapper{300, 500}.release(),
                                              int64_column_wrapper{0, 0}.release(),
                                              int64_column_wrapper{2, 4}.release()));
      auto condition = std::make_unique<gqe::logical_and_expression>(
        std::make_shared<gqe::equal_expression>(
          std::make_shared<gqe::column_reference_expression>(1),
          std::make_shared<gqe::column_reference_expression>(5)),
        std::make_shared<gqe::less_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::column_reference_expression>(3)));
      auto result =
        execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                                 .join_type = mode,
                                                 .is_perfect_hashing        = false,
                                                 .is_hash_map_cache_enabled = false},
                     "mixed_streamed",
                     qep_join_execution_options{
                       .right_partitions   = std::move(right),
                       .projection_indices = count ? std::vector<cudf::size_type>{}
                                                   : std::vector<cudf::size_type>{2, 0, 2},
                       .join_condition     = std::move(condition)});
      if (count) {
        if (mode == gqe::join_type_type::left_semi) {
          EXPECT_EQ(qep::try_row_count(result), 3)
            << "Semi must count matching left rows 100, 200, and 400 once each.";
        } else {
          EXPECT_EQ(qep::try_row_count(result), 1)
            << "Anti must count only the unmatched left row 300.";
        }
        continue;
      }
      auto sorted   = cudf::sort(*qep::to_table_view(result));
      auto expected = mode == gqe::join_type_type::left_semi
                        ? make_table_from_columns(int64_column_wrapper{1000, 2000, 4000}.release(),
                                                  int64_column_wrapper{100, 200, 400}.release(),
                                                  int64_column_wrapper{1000, 2000, 4000}.release())
                        : make_table_from_columns(int64_column_wrapper{3000}.release(),
                                                  int64_column_wrapper{300}.release(),
                                                  int64_column_wrapper{3000}.release());
      CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
    }
  }
}

TEST_F(QepJoinSpecificIntegrationTest, MixedSemiAntiProjectedResidualOperandsAndFilteredLeftInput)
{
  auto condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                            std::make_shared<gqe::column_reference_expression>(6)),
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(3),
      std::make_shared<gqe::column_reference_expression>(7)));
  auto filter =
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                           std::make_shared<gqe::literal_expression<int64_t>>(350));
  auto result =
    execute_join(qep_join_test_configuration{.policy    = physical::broadcast_policy::right,
                                             .join_type = gqe::join_type_type::left_semi,
                                             .is_perfect_hashing        = false,
                                             .is_hash_map_cache_enabled = false},
                 "mixed_project",
                 qep_join_execution_options{.has_input_sum_projection = true,
                                            .projection_indices       = {0},
                                            .left_filter              = std::move(filter),
                                            .join_condition           = std::move(condition)});
  auto sorted = cudf::sort(*qep::to_table_view(result));
  int64_column_wrapper expected{100, 200};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

}  // namespace gqe_test
