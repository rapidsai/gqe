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
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/qep/operator/fetch.hpp>
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

class QepFetchIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepFetchIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  gqe::context_reference ctx_ref;
};

TEST_F(QepFetchIntegrationTest, ExecutesPhysicalFetchOverReadPipeline)
{
  auto input = make_table_from_columns(int64_column_wrapper{10, 20, 30, 40, 50, 60}.release(),
                                       int64_column_wrapper{1, 2, 3, 4, 5, 6}.release());
  auto const filepath = temp_env->get_temp_filepath("physical_fetch.parquet");
  write_table_to_file(input->view(), {"value", "payload"}, filepath);
  gqe::test::single_table_provider provider{
    "input", std::make_shared<storage::parquet_table>(std::vector<std::string>{filepath})};

  auto const int64_type = cudf::data_type{cudf::type_id::INT64};
  auto read =
    std::make_shared<physical::read_relation>(std::vector<std::shared_ptr<physical::relation>>{},
                                              std::vector<std::string>{"value", "payload"},
                                              "input",
                                              nullptr,
                                              std::vector<cudf::data_type>{int64_type, int64_type});
  physical::fetch_relation fetch{read, /*offset=*/1, /*count=*/3};

  auto identified_plan = gqe_test::make_borrowed_identified_plan(&fetch);
  auto lowered =
    qep::physical_qep_transformation(identified_plan, &provider, get_query_ctx()->parameters);
  auto normalized = qep::pipeline_normalization(
    lowered.plan, lowered.output_data_types, lowered.stable_keys, lowered.physical_relations);
  ASSERT_EQ(gqe::test::tasks_of_type<qep::fetch_task>(normalized.plan()).size(), 1);

  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  gqe::test::execute_stage_roots(*graph);

  // The single parquet file reads as one partition, so the fetch window is deterministic. It
  // skips the first row and keeps the next three in file order.
  ASSERT_EQ(graph->root_tasks.size(), 1);
  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto table = qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int64_column_wrapper({20, 30, 40}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), int64_column_wrapper({2, 3, 4}));
}
