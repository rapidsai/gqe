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
#include "utilities.hpp"

#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/top_k.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/top_k.hpp>
#include <gqe/qep/physical_qep_transformation.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/utility/error.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/types.hpp>

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

}  // namespace

class QepTopKIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepTopKIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::context_reference ctx_ref;
};

TEST_F(QepTopKIntegrationTest, ExecutesTopKWithProjectionTailThroughTinyNormalizedIndirectBoundary)
{
  auto input          = make_table_from_columns(int64_column_wrapper{8, 1, 6, 4, 7, 3}.release(),
                                       int64_column_wrapper{9, 0, 10, 2, 8, 5}.release(),
                                       int64_column_wrapper{80, 10, 60, 40, 70, 30}.release());
  auto const filepath = temp_env->get_temp_filepath("normalized_physical_top_k.parquet");
  write_table_to_file(input->view(), {"key", "limit", "payload"}, filepath);
  gqe::test::single_table_provider provider{
    "input", std::make_shared<storage::parquet_table>(std::vector<std::string>{filepath})};

  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  auto read             = std::make_shared<physical::read_relation>(
    std::vector<std::shared_ptr<physical::relation>>{},
    std::vector<std::string>{"key", "limit", "payload"},
    "input",
    nullptr,
    std::vector<cudf::data_type>{int64_type, int64_type, int64_type});
  auto filter = std::make_shared<physical::filter_relation>(
    read,
    std::vector<std::shared_ptr<physical::relation>>{},
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                           std::make_shared<gqe::column_reference_expression>(1)),
    std::vector<cudf::size_type>{0, 1, 2});
  auto top_k = std::make_shared<physical::top_k_relation>(
    filter, 0, cudf::order::DESCENDING, cudf::null_order::AFTER, 1, 2);
  std::vector<std::unique_ptr<gqe::expression>> projections;
  for (cudf::size_type index : {2, 0, 1}) {
    projections.push_back(std::make_unique<gqe::column_reference_expression>(index));
  }
  physical::project_relation project{
    top_k, std::vector<std::shared_ptr<physical::relation>>{}, std::move(projections)};

  auto identified_plan = gqe_test::make_borrowed_identified_plan(&project);
  auto lowered =
    qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations, 1);
  auto const& plan        = normalized.plan();
  auto const top_k_tasks  = gqe::test::tasks_of_type<qep::top_k_task>(plan);
  auto const reader_tasks = gqe::test::tasks_of_type<qep::intermediate_state_read_task>(plan);
  auto const writer_tasks = gqe::test::tasks_of_type<qep::intermediate_state_write_task>(plan);
  ASSERT_EQ(top_k_tasks.size(), 1);
  ASSERT_EQ(reader_tasks.size(), 1);
  EXPECT_TRUE(static_cast<qep::intermediate_state_read_task const*>(reader_tasks.front())
                ->is_single_output());
  ASSERT_EQ(writer_tasks.size(), 1);
  auto const top_k_predecessors = plan.predecessors(top_k_tasks.front());
  ASSERT_EQ(top_k_predecessors.size(), 1);
  EXPECT_NE(dynamic_cast<qep::filter_task const*>(top_k_predecessors.front()), nullptr);
  EXPECT_EQ(plan.successors(top_k_tasks.front()),
            std::vector<qep::task const*>{reader_tasks.front()});
  auto const reader_successors = plan.successors(reader_tasks.front());
  ASSERT_EQ(reader_successors.size(), 1);
  EXPECT_NE(dynamic_cast<qep::project_task const*>(reader_successors.front()), nullptr);
  auto const terminal_predecessors = plan.predecessors(writer_tasks.front());
  ASSERT_EQ(terminal_predecessors.size(), 1);
  EXPECT_NE(dynamic_cast<qep::gather_task const*>(terminal_predecessors.front()), nullptr);

  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  gqe::test::execute_stage_roots(*graph);

  ASSERT_EQ(graph->root_tasks.size(), 1);
  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto table = qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper{70, 60});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), int64_column_wrapper{7, 6});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(2), int64_column_wrapper{8, 10});
}
