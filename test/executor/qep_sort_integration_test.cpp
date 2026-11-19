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

#include "../utility.hpp"

#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/qep/physical_qep_transformation.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

namespace physical = gqe::physical;
namespace qep      = gqe::qep;
namespace storage  = gqe::storage;

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

/** @brief Table provider exposing one named table to physical QEP lowering. */
class sort_table_provider final : public storage::table_provider {
 public:
  sort_table_provider(std::string name, std::shared_ptr<storage::table> table)
    : _name(std::move(name)), _table(std::move(table))
  {
  }

  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override
  {
    if (table_name != _name) {
      throw std::logic_error("sort_table_provider: unknown table " + std::string{table_name});
    }
    return _table;
  }

 private:
  std::string _name;
  std::shared_ptr<storage::table> _table;
};

}  // namespace

/** @brief Executes normalized physical sorts against partitioned Parquet input. */
class QepSortIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepSortIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()}
  {
    get_query_ctx()->parameters.num_partitions = 2;
  }

  void expect_ordered_fetch(int64_t offset,
                            int64_t count,
                            std::vector<int64_t> const& expected_payload,
                            bool has_tail = false,
                            bool is_empty = false)
  {
    auto first = make_table_from_columns(
      int64_column_wrapper{{2, 1, 0, 1}, {true, true, false, true}}.release(),
      int64_column_wrapper{1, 2, 2, 2}.release(),
      int64_column_wrapper{20, 12, 90, 10}.release());
    auto second = make_table_from_columns(
      int64_column_wrapper{{1, 3, 2, 0}, {true, true, true, false}}.release(),
      int64_column_wrapper{{1, 0, 0, 1}, {true, true, false, true}}.release(),
      int64_column_wrapper{11, 30, 91, 92}.release());
    if (is_empty) {
      first  = make_table_from_columns(int64_column_wrapper{}.release(),
                                      int64_column_wrapper{}.release(),
                                      int64_column_wrapper{}.release());
      second = make_table_from_columns(int64_column_wrapper{}.release(),
                                       int64_column_wrapper{}.release(),
                                       int64_column_wrapper{}.release());
    }
    auto const first_path  = temp_env->get_temp_filepath("ordered_fetch_first.parquet");
    auto const second_path = temp_env->get_temp_filepath("ordered_fetch_second.parquet");
    std::vector<std::string> const names{"first", "second", "payload"};
    write_table_to_file(first->view(), names, first_path);
    write_table_to_file(second->view(), names, second_path);
    sort_table_provider provider{
      "input", std::make_shared<storage::parquet_table>(std::vector{first_path, second_path})};
    auto const int64_type = cudf::data_type{cudf::type_id::INT64};
    auto read =
      std::make_shared<physical::read_relation>(std::vector<std::shared_ptr<physical::relation>>{},
                                                names,
                                                "input",
                                                nullptr,
                                                std::vector{int64_type, int64_type, int64_type});
    std::vector<std::unique_ptr<gqe::expression>> keys;
    for (cudf::size_type index : {0, 1, 2}) {
      keys.push_back(std::make_unique<gqe::column_reference_expression>(index));
    }
    std::shared_ptr<physical::relation> tail =
      std::make_shared<physical::concatenate_sort_relation>(
        read,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::move(keys),
        std::vector{cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING},
        std::vector{cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER});
    if (has_tail) {
      std::vector<std::unique_ptr<gqe::expression>> projections;
      for (cudf::size_type index : {2, 0, 1}) {
        projections.push_back(std::make_unique<gqe::column_reference_expression>(index));
      }
      auto project = std::make_shared<physical::project_relation>(
        tail, std::vector<std::shared_ptr<physical::relation>>{}, std::move(projections));
      tail = std::make_shared<physical::filter_relation>(
        project,
        std::vector<std::shared_ptr<physical::relation>>{},
        std::make_unique<gqe::greater_expression>(
          std::make_shared<gqe::column_reference_expression>(0),
          std::make_shared<gqe::literal_expression<int64_t>>(11)),
        std::vector<cudf::size_type>{0, 1, 2});
    }
    physical::fetch_relation fetch{tail, offset, count};
    auto identified_plan = gqe_test::make_borrowed_identified_plan(&fetch);
    auto lowered =
      qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
    auto normalized = qep::pipeline_normalization(
      lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations, 1);
    auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
    ASSERT_NE(graph, nullptr);
    gqe::execute_task_graph_single_gpu(ctx_ref, graph.get());
    ASSERT_EQ(graph->root_tasks.size(), 1);
    auto result = graph->root_tasks.front()->qep_state_result();
    ASSERT_TRUE(result.has_value());
    auto table = qep::to_table_view(*result);
    ASSERT_TRUE(table.has_value());
    ASSERT_EQ(table->num_columns(), 3);
    // Payloads uniquely identify rows, including the final tie-breaker and nullable keys.
    // Compare the received sequence directly, so a reordered window cannot pass.
    int64_column_wrapper expected(expected_payload.begin(), expected_payload.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(has_tail ? 0 : 2), expected);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepSortIntegrationTest, ExecutesMultipleColumnKeysAcrossPartitions)
{
  auto first =
    make_table_from_columns(int64_column_wrapper{{2, 1, 0}, {true, true, false}}.release(),
                            int64_column_wrapper{1, 2, 0}.release(),
                            int64_column_wrapper{20, 10, 90}.release());
  auto second =
    make_table_from_columns(int64_column_wrapper{1, 3, 2}.release(),
                            int64_column_wrapper{{1, 0, 0}, {true, true, false}}.release(),
                            int64_column_wrapper{11, 30, 91}.release());

  auto const first_path  = temp_env->get_temp_filepath("physical_sort_first.parquet");
  auto const second_path = temp_env->get_temp_filepath("physical_sort_second.parquet");
  std::vector<std::string> const column_names{"first", "second", "payload"};
  write_table_to_file(first->view(), column_names, first_path);
  write_table_to_file(second->view(), column_names, second_path);
  sort_table_provider provider{
    "input", std::make_shared<storage::parquet_table>(std::vector{first_path, second_path})};

  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  auto read =
    std::make_shared<physical::read_relation>(std::vector<std::shared_ptr<physical::relation>>{},
                                              column_names,
                                              "input",
                                              nullptr,
                                              std::vector{int64_type, int64_type, int64_type});
  std::vector<std::unique_ptr<gqe::expression>> keys;
  keys.push_back(std::make_unique<gqe::column_reference_expression>(0));
  keys.push_back(std::make_unique<gqe::column_reference_expression>(1));
  physical::concatenate_sort_relation sort(read,
                                           {},
                                           std::move(keys),
                                           {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                           {cudf::null_order::AFTER, cudf::null_order::BEFORE});

  auto identified_plan = gqe_test::make_borrowed_identified_plan(&sort);
  auto lowered =
    qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations, 1);
  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  gqe::execute_task_graph_single_gpu(ctx_ref, graph.get());

  ASSERT_EQ(graph->root_tasks.size(), 1);
  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto table = qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 3);

  int64_column_wrapper expected_first{{1, 1, 2, 2, 3, 0}, {true, true, true, true, true, false}};
  int64_column_wrapper expected_second{{2, 1, 1, 0, 0, 0}, {true, true, true, false, true, true}};
  int64_column_wrapper expected_payload{10, 11, 20, 91, 30, 90};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_second);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), expected_payload);
}

TEST_F(QepSortIntegrationTest, FetchPreservesNullableMultiKeyOrderWithTinyMorsels)
{
  expect_ordered_fetch(1, 6, {12, 11, 20, 91, 30, 90});
}

TEST_F(QepSortIntegrationTest, ProjectionFilterFetchPreservesOrderWithTinyMorsels)
{
  expect_ordered_fetch(1, 4, {20, 91, 30, 90}, true);
}

TEST_F(QepSortIntegrationTest, OrderedFetchWithZeroLimitPreservesEmptySchema)
{
  expect_ordered_fetch(1, 0, {});
}

TEST_F(QepSortIntegrationTest, OrderedFetchBeyondEndPreservesEmptySchema)
{
  expect_ordered_fetch(20, 3, {});
}

TEST_F(QepSortIntegrationTest, OrderedFetchOverEmptyInputPreservesSchema)
{
  expect_ordered_fetch(0, 3, {}, false, true);
}
