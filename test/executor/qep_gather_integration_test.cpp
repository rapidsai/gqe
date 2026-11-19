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

#include <gqe/executor/qep_adapter.hpp>
#include <gqe/executor/qep_task_graph_transformation.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

cudf::table one_column_table(std::unique_ptr<cudf::column> column)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return cudf::table{std::move(columns)};
}

class masked_emitter_task : public gqe::qep::iterate_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const& stream_handle,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    return gqe::qep::state_container_builder()
      .add_state(
        gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true, false}.release()},
        stream_handle)
      .add_state(one_column_table(int64_column_wrapper{10, 20, 30, 40}.release()), stream_handle)
      .build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<masked_emitter_task>();
  }

 private:
  mutable bool _emitted = false;
};

class indirect_emitter_task : public gqe::qep::iterate_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const& stream_handle,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted    = true;
    auto source = gqe::qep::make_shared_state(
      gqe::qep::state_kind::cudf_column{int64_column_wrapper{10, 20, 30}.release()}, stream_handle);
    auto positions =
      gqe::qep::make_position_list(int32_column_wrapper{2, 0}.release(), stream_handle);
    auto relation = gqe::qep::indirect_table_source_state::make({std::move(source)});
    auto gather_source =
      gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(positions));
    return gqe::qep::state_container{
      gqe::qep::indirect_column_state::make(gather_source, 0),
      gqe::qep::make_shared_state(
        gqe::qep::state_kind::cudf_column{int64_column_wrapper{100, 200}.release()},
        stream_handle)};
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<indirect_emitter_task>();
  }

 private:
  mutable bool _emitted = false;
};

class QepGatherIntegrationTest : public gqe::test::BaseFixture {
 protected:
  QepGatherIntegrationTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  static void expect_gathered(gqe::qep::state_container const& result)
  {
    ASSERT_EQ(result.size(), 1);
    ASSERT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*result.front()));
    auto table = gqe::qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    int64_column_wrapper expected{10, 30};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
  }

  gqe::context_reference ctx_ref;
};

TEST_F(QepGatherIntegrationTest, StatelessTransformAdapterPublishesOrdinaryTable)
{
  auto emitter =
    std::make_shared<gqe::iterate_adapter_task>(ctx_ref,
                                                0,
                                                0,
                                                std::vector<std::shared_ptr<gqe::task>>{},
                                                std::make_unique<masked_emitter_task>());
  emitter->execute();
  auto gather = std::make_shared<gqe::stateless_transform_adapter_task>(
    ctx_ref,
    1,
    0,
    std::vector<std::shared_ptr<gqe::task>>{emitter},
    std::make_unique<gqe::qep::gather_task>());
  gather->execute();

  auto result = gather->qep_state_result();
  ASSERT_TRUE(result.has_value());
  expect_gathered(*result);
}

TEST_F(QepGatherIntegrationTest, LoweredPipelineExecutesThroughTerminalWriter)
{
  auto emitter    = std::make_unique<masked_emitter_task>();
  auto gather     = std::make_unique<gqe::qep::gather_task>();
  auto* emitter_p = emitter.get();
  auto* gather_p  = gather.get();
  auto qep        = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(emitter))
               .add_task(std::move(gather))
               .add_successor(emitter_p, gather_p)
               .build();
  gqe::qep::task_output_data_types output_data_types;
  for (auto const* task : qep.tasks()) {
    output_data_types.emplace(task, std::vector{cudf::data_type{cudf::type_id::INT64}});
  }
  auto normalized = gqe::qep::pipeline_normalization(
    qep, output_data_types, gqe_test::make_test_task_stable_keys(qep));
  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);

  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2);
  ASSERT_EQ(graph->root_tasks.size(), 1);
  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  expect_gathered(*result);
}

TEST_F(QepGatherIntegrationTest, GatherAdapterPublishesMixedIndirectStateAsTable)
{
  auto emitter =
    std::make_shared<gqe::iterate_adapter_task>(ctx_ref,
                                                0,
                                                0,
                                                std::vector<std::shared_ptr<gqe::task>>{},
                                                std::make_unique<indirect_emitter_task>());
  emitter->execute();
  auto gather = std::make_shared<gqe::stateless_transform_adapter_task>(
    ctx_ref,
    1,
    0,
    std::vector<std::shared_ptr<gqe::task>>{emitter},
    std::make_unique<gqe::qep::gather_task>());
  gather->execute();

  auto result = gather->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 2);
  int64_column_wrapper expected_indirect{30, 10};
  int64_column_wrapper expected_direct{100, 200};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_indirect);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_direct);
}
