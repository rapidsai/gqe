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

#include <gqe/qep/pipeline_normalization.hpp>

#include "../utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/qep/operator/fetch.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/operator/intermediate_state_read.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/operator/merge_sort.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/operator/top_k.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qep      = gqe::qep;
namespace physical = gqe::physical;

/** @brief Add a stable numeric identity to task-family stubs used in graph assertions. */
class identified_task {
 public:
  explicit identified_task(int id) : _id(id) {}
  virtual ~identified_task() = default;
  [[nodiscard]] int id() const noexcept { return _id; }

 private:
  int _id;
};

/** @brief Iterate-task stub with a stable numeric identity for graph assertions. */
class test_iterate_task final : public qep::iterate_task, public identified_task {
 public:
  explicit test_iterate_task(int id) : identified_task(id) {}

  qep::state_container initialize(qep::state_container_view,
                                  gqe::context_reference,
                                  gqe::query_stream const&,
                                  rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<qep::state_container> next(qep::state_container_view,
                                           gqe::context_reference,
                                           gqe::query_stream const&,
                                           rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<test_iterate_task>(id());
  }
};

/** @brief Stateless-transform stub with a stable numeric identity for graph assertions. */
class test_stateless_transform_task final : public qep::stateless_transform_task,
                                            public identified_task {
 public:
  explicit test_stateless_transform_task(int id) : identified_task(id) {}

  std::optional<qep::state_container> next(qep::state_container_view,
                                           gqe::context_reference,
                                           gqe::query_stream const&,
                                           rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<test_stateless_transform_task>(id());
  }
};

/** @brief Fold-task stub with a stable numeric identity for graph assertions. */
class test_fold_task final : public qep::fold_task, public identified_task {
 public:
  explicit test_fold_task(int id) : identified_task(id) {}

  qep::state_container initialize(gqe::context_reference,
                                  gqe::query_stream const&,
                                  rmm::device_async_resource_ref) const override
  {
    return qep::make_row_count_container(0);
  }

  void next(qep::state_container_view,
            qep::state_container_view,
            gqe::context_reference,
            gqe::query_stream const&,
            rmm::device_async_resource_ref) const override
  {
  }

  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<test_fold_task>(id());
  }
};

/** @brief Stateful-transform stub with a stable numeric identity for graph assertions. */
class test_stateful_transform_task final : public qep::stateful_transform_task,
                                           public identified_task {
 public:
  explicit test_stateful_transform_task(int id,
                                        qep::task_ordering policy = qep::task_ordering::unordered)
    : qep::stateful_transform_task(policy), identified_task(id)
  {
  }

  qep::state_container initialize(qep::state_container_view,
                                  gqe::context_reference,
                                  gqe::query_stream const&,
                                  rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<qep::state_container> next(qep::state_container_view,
                                           qep::state_container_view,
                                           gqe::context_reference,
                                           gqe::query_stream const&,
                                           rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<qep::task> clone() const override
  {
    return std::make_unique<test_stateful_transform_task>(id(), ordering_policy());
  }
};

template <typename T>
std::vector<T const*> tasks_of_type(qep::query_execution_plan const& plan)
{
  std::vector<T const*> result;
  for (auto const* task : plan.tasks()) {
    if (auto const* typed = dynamic_cast<T const*>(task); typed != nullptr) {
      result.push_back(typed);
    }
  }
  return result;
}

qep::task const* task_with_id(qep::query_execution_plan const& plan, int id)
{
  for (auto const* task : plan.tasks()) {
    if (auto const* identified = dynamic_cast<identified_task const*>(task);
        identified != nullptr && identified->id() == id) {
      return task;
    }
  }
  return nullptr;
}

qep::task_output_data_types make_output_data_types(qep::query_execution_plan const& plan)
{
  qep::task_output_data_types output_data_types;
  for (auto const* task : plan.tasks()) {
    output_data_types.emplace(task, std::vector{cudf::data_type{cudf::type_id::INT64}});
  }
  return output_data_types;
}

qep::task_physical_relations associate_with_relation(qep::query_execution_plan const& plan,
                                                     physical::relation const* relation)
{
  qep::task_physical_relations physical_relations;
  for (auto const* task : plan.tasks()) {
    physical_relations.emplace(task, relation);
  }
  return physical_relations;
}

qep::pipeline_normalization_result normalize(qep::query_execution_plan const& plan)
{
  return qep::pipeline_normalization(
    plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan));
}

TEST(QepPipelineNormalizationTest, RejectsNonPositiveMorselMaxRows)
{
  auto source            = std::make_unique<test_iterate_task>(1);
  auto original          = qep::query_execution_plan_builder().add_task(std::move(source)).build();
  auto const stable_keys = gqe_test::make_test_task_stable_keys(original);

  for (auto const bound : {cudf::size_type{-1}, cudf::size_type{0}}) {
    EXPECT_THROW(std::ignore = qep::pipeline_normalization(original, {}, stable_keys, {}, bound),
                 std::invalid_argument);
  }
}

TEST(QepPipelineNormalizationTest, RejectsMissingStableTaskKeys)
{
  auto source   = std::make_unique<test_iterate_task>(1);
  auto original = qep::query_execution_plan_builder().add_task(std::move(source)).build();

  EXPECT_THROW(std::ignore = qep::pipeline_normalization(original, {}, {}), std::logic_error);

  auto unrelated             = std::make_unique<test_iterate_task>(2);
  qep::task_stable_keys keys = {
    {unrelated.get(), {.relation_id = 1, .role = qep::lowering_role::read}}};
  EXPECT_THROW(std::ignore = qep::pipeline_normalization(original, {}, keys), std::logic_error);
}

TEST(QepPipelineNormalizationTest, RejectsDuplicateStableTaskKeys)
{
  auto first             = std::make_unique<test_iterate_task>(1);
  auto second            = std::make_unique<test_iterate_task>(2);
  auto const* first_ptr  = first.get();
  auto const* second_ptr = second.get();
  auto original          = qep::query_execution_plan_builder()
                    .add_task(std::move(first))
                    .add_task(std::move(second))
                    .build();
  qep::stable_task_key const duplicate_key{.relation_id = 1, .role = qep::lowering_role::read};
  qep::task_stable_keys const stable_keys{{first_ptr, duplicate_key}, {second_ptr, duplicate_key}};

  EXPECT_THROW(std::ignore = qep::pipeline_normalization(original, {}, stable_keys),
               std::logic_error);
}

TEST(QepPipelineNormalizationTest, CompletesIterateTerminalWithGatherAndWriter)
{
  auto source            = std::make_unique<test_iterate_task>(1);
  auto* source_ptr       = source.get();
  auto original          = qep::query_execution_plan_builder().add_task(std::move(source)).build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto const gathers     = tasks_of_type<qep::gather_task>(normalized);
  auto const writers     = tasks_of_type<qep::intermediate_state_write_task>(normalized);

  ASSERT_EQ(original.tasks().size(), 1);
  EXPECT_TRUE(original.successors(source_ptr).empty());
  ASSERT_EQ(gathers.size(), 1);
  ASSERT_EQ(writers.size(), 1);
  auto const* cloned_source = task_with_id(normalized, 1);
  ASSERT_NE(cloned_source, nullptr);
  EXPECT_EQ(normalized.predecessors(gathers.front()), std::vector<qep::task const*>{cloned_source});
  EXPECT_EQ(normalized.predecessors(writers.front()),
            std::vector<qep::task const*>{gathers.front()});
}

TEST(QepPipelineNormalizationTest, ReturnsMetadataForTheOwnedNormalizedPlan)
{
  auto source         = std::make_unique<test_iterate_task>(1);
  auto transform      = std::make_unique<test_stateless_transform_task>(2);
  auto* source_ptr    = source.get();
  auto* transform_ptr = transform.get();
  auto original       = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(transform))
                    .add_successor(source_ptr, transform_ptr)
                    .build();

  auto initial         = normalize(original);
  auto result          = std::move(initial);
  auto const& plan     = result.plan();
  auto const& metadata = result.metadata();
  auto const tasks     = plan.tasks();

  ASSERT_EQ(metadata.topological_order.size(), tasks.size());
  EXPECT_EQ(std::unordered_set<qep::task const*>(metadata.topological_order.begin(),
                                                 metadata.topological_order.end()),
            std::unordered_set<qep::task const*>(tasks.begin(), tasks.end()));

  std::unordered_map<qep::task const*, std::size_t> task_position;
  for (std::size_t i = 0; i < metadata.topological_order.size(); ++i) {
    task_position.emplace(metadata.topological_order[i], i);
  }
  for (auto const* predecessor : tasks) {
    EXPECT_TRUE(metadata.pipelines.task_to_pipeline.contains(predecessor));
    for (auto const* successor : plan.successors(predecessor)) {
      EXPECT_LT(task_position.at(predecessor), task_position.at(successor));
    }
  }

  ASSERT_EQ(metadata.pipeline_topological_order.size(), metadata.pipelines.num_pipelines);
  std::unordered_map<qep::pipeline_id, std::size_t> pipeline_position;
  for (std::size_t i = 0; i < metadata.pipeline_topological_order.size(); ++i) {
    pipeline_position.emplace(metadata.pipeline_topological_order[i], i);
  }
  for (auto const pipeline : metadata.pipeline_topological_order) {
    for (auto const successor : metadata.pipelines.successors.at(pipeline)) {
      EXPECT_LT(pipeline_position.at(pipeline), pipeline_position.at(successor));
    }
  }
}

TEST(QepPipelineNormalizationTest, CompletesTransformTerminalUnconditionallyWithGatherAndWriter)
{
  auto source         = std::make_unique<test_iterate_task>(1);
  auto transform      = std::make_unique<test_stateless_transform_task>(2);
  auto* source_ptr    = source.get();
  auto* transform_ptr = transform.get();
  auto original       = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(transform))
                    .add_successor(source_ptr, transform_ptr)
                    .build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  EXPECT_EQ(tasks_of_type<qep::gather_task>(normalized).size(), 1);
  EXPECT_EQ(tasks_of_type<qep::intermediate_state_write_task>(normalized).size(), 1);
}

TEST(QepPipelineNormalizationTest, LeavesFoldTerminalComplete)
{
  auto source      = std::make_unique<test_iterate_task>(1);
  auto* source_ptr = source.get();
  auto fold        = std::make_unique<test_fold_task>(2);
  auto* fold_ptr   = fold.get();
  auto original    = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(fold))
                    .add_successor(source_ptr, fold_ptr)
                    .build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  EXPECT_TRUE(tasks_of_type<qep::gather_task>(normalized).empty());
  EXPECT_TRUE(tasks_of_type<qep::intermediate_state_write_task>(normalized).empty());
}

TEST(QepPipelineNormalizationTest, InsertsReaderAfterFoldForPipelinedSuccessors)
{
  auto source         = std::make_unique<test_iterate_task>(1);
  auto first_fold     = std::make_unique<test_fold_task>(2);
  auto transform      = std::make_unique<test_stateless_transform_task>(3);
  auto second_fold    = std::make_unique<test_fold_task>(4);
  auto* source_ptr    = source.get();
  auto* first_ptr     = first_fold.get();
  auto* transform_ptr = transform.get();
  auto* second_ptr    = second_fold.get();
  auto original       = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(first_fold))
                    .add_task(std::move(transform))
                    .add_task(std::move(second_fold))
                    .add_successor(source_ptr, first_ptr)
                    .add_successor(first_ptr, transform_ptr)
                    .add_successor(first_ptr, second_ptr)
                    .build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto readers           = tasks_of_type<qep::intermediate_state_read_task>(normalized);
  ASSERT_EQ(readers.size(), 1);
  EXPECT_EQ(normalized.predecessors(task_with_id(normalized, 3)),
            std::vector<qep::task const*>{readers.front()});
  EXPECT_EQ(normalized.predecessors(task_with_id(normalized, 4)),
            std::vector<qep::task const*>{readers.front()});
  EXPECT_EQ(normalized.predecessors(readers.front()),
            std::vector<qep::task const*>{task_with_id(normalized, 2)});
}

TEST(QepPipelineNormalizationTest, InsertsReadersBetweenConsecutiveFolds)
{
  auto source      = std::make_unique<test_iterate_task>(1);
  auto first_fold  = std::make_unique<test_fold_task>(2);
  auto second_fold = std::make_unique<test_fold_task>(3);
  auto third_fold  = std::make_unique<test_fold_task>(4);
  auto* source_ptr = source.get();
  auto* first_ptr  = first_fold.get();
  auto* second_ptr = second_fold.get();
  auto* third_ptr  = third_fold.get();
  auto original    = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(first_fold))
                    .add_task(std::move(second_fold))
                    .add_task(std::move(third_fold))
                    .add_successor(source_ptr, first_ptr)
                    .add_successor(first_ptr, second_ptr)
                    .add_successor(second_ptr, third_ptr)
                    .build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto const readers     = tasks_of_type<qep::intermediate_state_read_task>(normalized);
  ASSERT_EQ(readers.size(), 2);

  auto const assert_reader_between = [&](int predecessor_id, int successor_id) {
    auto const predecessors = normalized.predecessors(task_with_id(normalized, successor_id));
    ASSERT_EQ(predecessors.size(), 1);
    auto const* reader =
      dynamic_cast<qep::intermediate_state_read_task const*>(predecessors.front());
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(normalized.predecessors(reader),
              std::vector<qep::task const*>{task_with_id(normalized, predecessor_id)});
  };
  assert_reader_between(2, 3);
  assert_reader_between(3, 4);
}

TEST(QepPipelineNormalizationTest, ReusesReaderAcrossDiamondBranches)
{
  auto source      = std::make_unique<test_iterate_task>(1);
  auto first_fold  = std::make_unique<test_fold_task>(2);
  auto left        = std::make_unique<test_stateless_transform_task>(3);
  auto right       = std::make_unique<test_stateless_transform_task>(4);
  auto sink        = std::make_unique<test_fold_task>(5);
  auto* source_ptr = source.get();
  auto* first_ptr  = first_fold.get();
  auto* left_ptr   = left.get();
  auto* right_ptr  = right.get();
  auto* sink_ptr   = sink.get();
  auto original    = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(first_fold))
                    .add_task(std::move(left))
                    .add_task(std::move(right))
                    .add_task(std::move(sink))
                    .add_successor(source_ptr, first_ptr)
                    .add_successor(first_ptr, left_ptr)
                    .add_successor(first_ptr, right_ptr)
                    .add_successor(left_ptr, sink_ptr)
                    .add_successor(right_ptr, sink_ptr)
                    .build();

  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto readers           = tasks_of_type<qep::intermediate_state_read_task>(normalized);
  ASSERT_EQ(readers.size(), 1);
  auto const* normalized_left  = task_with_id(normalized, 3);
  auto const* normalized_right = task_with_id(normalized, 4);
  auto const* normalized_sink  = task_with_id(normalized, 5);
  EXPECT_EQ(normalized.predecessors(normalized_left),
            std::vector<qep::task const*>{readers.front()});
  EXPECT_EQ(normalized.predecessors(normalized_right),
            std::vector<qep::task const*>{readers.front()});
  auto const reader_successors = normalized.successors(readers.front());
  ASSERT_EQ(reader_successors.size(), 2);
  EXPECT_EQ(
    std::unordered_set<qep::task const*>(reader_successors.begin(), reader_successors.end()),
    (std::unordered_set<qep::task const*>{normalized_left, normalized_right}));
  EXPECT_EQ(normalized.predecessors(normalized_sink),
            (std::vector<qep::task const*>{normalized_left, normalized_right}));
}

TEST(QepPipelineNormalizationTest, InsertsAndReusesWriterBeforeIterates)
{
  auto source      = std::make_unique<test_iterate_task>(1);
  auto left        = std::make_unique<test_iterate_task>(2);
  auto right       = std::make_unique<test_iterate_task>(3);
  auto* source_ptr = source.get();
  auto* left_ptr   = left.get();
  auto* right_ptr  = right.get();
  auto original    = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(left))
                    .add_task(std::move(right))
                    .add_successor(source_ptr, left_ptr)
                    .add_successor(source_ptr, right_ptr)
                    .build();
  auto normalized_result  = normalize(original);
  auto const& normalized  = normalized_result.plan();
  auto left_predecessors  = normalized.predecessors(task_with_id(normalized, 2));
  auto right_predecessors = normalized.predecessors(task_with_id(normalized, 3));
  ASSERT_EQ(left_predecessors.size(), 1);
  ASSERT_EQ(right_predecessors.size(), 1);
  EXPECT_EQ(left_predecessors.front(), right_predecessors.front());
  EXPECT_NE(dynamic_cast<qep::intermediate_state_write_task const*>(left_predecessors.front()),
            nullptr);
  EXPECT_EQ(normalized.successors(left_predecessors.front()).size(), 2);
}

TEST(QepPipelineNormalizationTest, ReusesWriterAcrossDiamondBranches)
{
  auto source      = std::make_unique<test_iterate_task>(1);
  auto left        = std::make_unique<test_iterate_task>(2);
  auto right       = std::make_unique<test_iterate_task>(3);
  auto sink        = std::make_unique<test_fold_task>(4);
  auto* source_ptr = source.get();
  auto* left_ptr   = left.get();
  auto* right_ptr  = right.get();
  auto* sink_ptr   = sink.get();
  auto original    = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(left))
                    .add_task(std::move(right))
                    .add_task(std::move(sink))
                    .add_successor(source_ptr, left_ptr)
                    .add_successor(source_ptr, right_ptr)
                    .add_successor(left_ptr, sink_ptr)
                    .add_successor(right_ptr, sink_ptr)
                    .build();

  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto writers           = tasks_of_type<qep::intermediate_state_write_task>(normalized);
  ASSERT_EQ(writers.size(), 1);
  auto const* normalized_left  = task_with_id(normalized, 2);
  auto const* normalized_right = task_with_id(normalized, 3);
  auto const* normalized_sink  = task_with_id(normalized, 4);
  EXPECT_EQ(normalized.predecessors(normalized_left),
            std::vector<qep::task const*>{writers.front()});
  EXPECT_EQ(normalized.predecessors(normalized_right),
            std::vector<qep::task const*>{writers.front()});
  auto const writer_successors = normalized.successors(writers.front());
  ASSERT_EQ(writer_successors.size(), 2);
  EXPECT_EQ(
    std::unordered_set<qep::task const*>(writer_successors.begin(), writer_successors.end()),
    (std::unordered_set<qep::task const*>{normalized_left, normalized_right}));
  EXPECT_EQ(normalized.predecessors(normalized_sink),
            (std::vector<qep::task const*>{normalized_left, normalized_right}));
}

TEST(QepPipelineNormalizationTest, PreservesStatefulPredecessorRolesAndOrder)
{
  auto build_source      = std::make_unique<test_iterate_task>(1);
  auto build_fold        = std::make_unique<test_fold_task>(2);
  auto probe_source      = std::make_unique<test_iterate_task>(3);
  auto probe             = std::make_unique<test_stateful_transform_task>(4);
  auto* build_source_ptr = build_source.get();
  auto* build_fold_ptr   = build_fold.get();
  auto* probe_source_ptr = probe_source.get();
  auto* probe_ptr        = probe.get();
  auto original          = qep::query_execution_plan_builder()
                    .add_task(std::move(build_source))
                    .add_task(std::move(build_fold))
                    .add_task(std::move(probe_source))
                    .add_task(std::move(probe))
                    .add_successor(build_source_ptr, build_fold_ptr)
                    .add_successor(build_fold_ptr, probe_ptr)
                    .add_successor(probe_source_ptr, probe_ptr)
                    .build();

  physical::read_relation stateful_relation({}, {}, "stateful", nullptr, {});
  auto const physical_relations = associate_with_relation(original, &stateful_relation);
  auto normalized_result =
    qep::pipeline_normalization(original,
                                make_output_data_types(original),
                                gqe_test::make_test_task_stable_keys(original),
                                physical_relations);
  auto const& normalized = normalized_result.plan();
  auto predecessors      = normalized.predecessors(task_with_id(normalized, 4));
  ASSERT_EQ(predecessors.size(), 2);
  EXPECT_EQ(dynamic_cast<identified_task const*>(predecessors[0])->id(), 2);
  EXPECT_EQ(dynamic_cast<identified_task const*>(predecessors[1])->id(), 3);
}

TEST(QepPipelineNormalizationTest, RejectsMissingPhysicalRelationForStatefulBreaker)
{
  auto build_source      = std::make_unique<test_iterate_task>(1);
  auto build_fold        = std::make_unique<test_fold_task>(2);
  auto probe_source      = std::make_unique<test_iterate_task>(3);
  auto stateful          = std::make_unique<test_stateful_transform_task>(4);
  auto* build_source_ptr = build_source.get();
  auto* build_fold_ptr   = build_fold.get();
  auto* probe_source_ptr = probe_source.get();
  auto* stateful_ptr     = stateful.get();
  auto original          = qep::query_execution_plan_builder()
                    .add_task(std::move(build_source))
                    .add_task(std::move(build_fold))
                    .add_task(std::move(probe_source))
                    .add_task(std::move(stateful))
                    .add_successor(build_source_ptr, build_fold_ptr)
                    .add_successor(build_fold_ptr, stateful_ptr)
                    .add_successor(probe_source_ptr, stateful_ptr)
                    .build();

  physical::read_relation stateful_relation({}, {}, "stateful", nullptr, {});
  auto const output_data_types = make_output_data_types(original);
  auto const stable_keys       = gqe_test::make_test_task_stable_keys(original);
  qep::task_physical_relations missing_predecessor{{stateful_ptr, &stateful_relation}};
  EXPECT_THROW(std::ignore = qep::pipeline_normalization(
                 original, output_data_types, stable_keys, missing_predecessor),
               std::logic_error);

  qep::task_physical_relations missing_successor{{build_fold_ptr, &stateful_relation}};
  EXPECT_THROW(std::ignore = qep::pipeline_normalization(
                 original, output_data_types, stable_keys, missing_successor),
               std::logic_error);

  qep::task_physical_relations null_predecessor{{build_fold_ptr, nullptr},
                                                {stateful_ptr, &stateful_relation}};
  EXPECT_THROW(std::ignore = qep::pipeline_normalization(
                 original, output_data_types, stable_keys, null_predecessor),
               std::logic_error);

  qep::task_physical_relations null_successor{{build_fold_ptr, &stateful_relation},
                                              {stateful_ptr, nullptr}};
  EXPECT_THROW(std::ignore = qep::pipeline_normalization(
                 original, output_data_types, stable_keys, null_successor),
               std::logic_error);
}

TEST(QepPipelineNormalizationTest, PreservesInternalAndReadsExternalStatefulBreakers)
{
  auto internal_source       = std::make_unique<test_iterate_task>(1);
  auto internal_state        = std::make_unique<test_fold_task>(2);
  auto external_source       = std::make_unique<test_iterate_task>(3);
  auto external_breaker      = std::make_unique<test_fold_task>(4);
  auto stateful              = std::make_unique<test_stateful_transform_task>(5);
  auto* internal_source_ptr  = internal_source.get();
  auto* internal_state_ptr   = internal_state.get();
  auto* external_source_ptr  = external_source.get();
  auto* external_breaker_ptr = external_breaker.get();
  auto* stateful_ptr         = stateful.get();
  auto original              = qep::query_execution_plan_builder()
                    .add_task(std::move(internal_source))
                    .add_task(std::move(internal_state))
                    .add_task(std::move(external_source))
                    .add_task(std::move(external_breaker))
                    .add_task(std::move(stateful))
                    .add_successor(internal_source_ptr, internal_state_ptr)
                    .add_successor(external_source_ptr, external_breaker_ptr)
                    .add_successor(internal_state_ptr, stateful_ptr)
                    .add_successor(external_breaker_ptr, stateful_ptr)
                    .build();

  physical::read_relation stateful_relation({}, {}, "stateful", nullptr, {});
  physical::read_relation external_relation({}, {}, "external", nullptr, {});
  qep::task_physical_relations physical_relations{{internal_source_ptr, &stateful_relation},
                                                  {internal_state_ptr, &stateful_relation},
                                                  {external_source_ptr, &external_relation},
                                                  {external_breaker_ptr, &external_relation},
                                                  {stateful_ptr, &stateful_relation}};
  auto normalized_result =
    qep::pipeline_normalization(original,
                                make_output_data_types(original),
                                gqe_test::make_test_task_stable_keys(original),
                                physical_relations);
  auto const& normalized = normalized_result.plan();
  auto predecessors      = normalized.predecessors(task_with_id(normalized, 5));
  ASSERT_EQ(predecessors.size(), 2);
  EXPECT_EQ(dynamic_cast<identified_task const*>(predecessors[0])->id(), 2);
  auto const* reader = dynamic_cast<qep::intermediate_state_read_task const*>(predecessors[1]);
  ASSERT_NE(reader, nullptr);
  auto reader_predecessors = normalized.predecessors(reader);
  ASSERT_EQ(reader_predecessors.size(), 1);
  EXPECT_EQ(dynamic_cast<identified_task const*>(reader_predecessors.front())->id(), 4);
}

TEST(QepPipelineNormalizationTest, RejectsStatefulTransformWithoutPipelinedPredecessor)
{
  auto left_source       = std::make_unique<test_iterate_task>(1);
  auto left_fold         = std::make_unique<test_fold_task>(2);
  auto right_source      = std::make_unique<test_iterate_task>(3);
  auto right_fold        = std::make_unique<test_fold_task>(4);
  auto stateful          = std::make_unique<test_stateful_transform_task>(5);
  auto* left_source_ptr  = left_source.get();
  auto* left_fold_ptr    = left_fold.get();
  auto* right_source_ptr = right_source.get();
  auto* right_fold_ptr   = right_fold.get();
  auto* stateful_ptr     = stateful.get();
  auto original          = qep::query_execution_plan_builder()
                    .add_task(std::move(left_source))
                    .add_task(std::move(left_fold))
                    .add_task(std::move(right_source))
                    .add_task(std::move(right_fold))
                    .add_task(std::move(stateful))
                    .add_successor(left_source_ptr, left_fold_ptr)
                    .add_successor(right_source_ptr, right_fold_ptr)
                    .add_successor(left_fold_ptr, stateful_ptr)
                    .add_successor(right_fold_ptr, stateful_ptr)
                    .build();

  physical::read_relation stateful_relation({}, {}, "stateful", nullptr, {});
  auto const physical_relations = associate_with_relation(original, &stateful_relation);
  EXPECT_THROW(
    std::ignore = qep::pipeline_normalization(original,
                                              make_output_data_types(original),
                                              gqe_test::make_test_task_stable_keys(original),
                                              physical_relations),
    std::logic_error);
}

TEST(QepPipelineNormalizationTest, NormalizesMixedPredecessorsAndPreservesFanInOrder)
{
  auto materialized_source      = std::make_unique<test_iterate_task>(1);
  auto materialized             = std::make_unique<test_fold_task>(2);
  auto pipelined_source         = std::make_unique<test_iterate_task>(3);
  auto successor                = std::make_unique<test_stateless_transform_task>(4);
  auto* materialized_source_ptr = materialized_source.get();
  auto* materialized_ptr        = materialized.get();
  auto* pipelined_source_ptr    = pipelined_source.get();
  auto* successor_ptr           = successor.get();
  auto original                 = qep::query_execution_plan_builder()
                    .add_task(std::move(materialized_source))
                    .add_task(std::move(materialized))
                    .add_task(std::move(pipelined_source))
                    .add_task(std::move(successor))
                    .add_successor(materialized_source_ptr, materialized_ptr)
                    .add_successor(materialized_ptr, successor_ptr)
                    .add_successor(pipelined_source_ptr, successor_ptr)
                    .build();
  auto normalized_result = normalize(original);
  auto const& normalized = normalized_result.plan();
  auto predecessors      = normalized.predecessors(task_with_id(normalized, 4));
  ASSERT_EQ(predecessors.size(), 2);
  EXPECT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(predecessors[0]), nullptr);
  EXPECT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(predecessors[1]), nullptr);
  EXPECT_EQ(normalized.predecessors(predecessors[0]),
            std::vector<qep::task const*>{task_with_id(normalized, 2)});
  auto const normalized_pipelined_input = normalized.predecessors(predecessors[1]);
  ASSERT_EQ(normalized_pipelined_input.size(), 1);
  EXPECT_NE(
    dynamic_cast<qep::intermediate_state_write_task const*>(normalized_pipelined_input.front()),
    nullptr);
}

TEST(QepPipelineNormalizationTest, CompletesOnlyPipelinedTerminalsAndIsIdempotent)
{
  auto pipelined_source      = std::make_unique<test_iterate_task>(1);
  auto materialized_source   = std::make_unique<test_iterate_task>(2);
  auto materialized          = std::make_unique<test_fold_task>(3);
  auto* materialized_src_ptr = materialized_source.get();
  auto* materialized_ptr     = materialized.get();
  auto original              = qep::query_execution_plan_builder()
                    .add_task(std::move(pipelined_source))
                    .add_task(std::move(materialized_source))
                    .add_task(std::move(materialized))
                    .add_successor(materialized_src_ptr, materialized_ptr)
                    .build();
  auto once_result  = normalize(original);
  auto const& once  = once_result.plan();
  auto twice_result = normalize(once);
  auto const& twice = twice_result.plan();
  EXPECT_EQ(tasks_of_type<qep::gather_task>(once).size(), 1);
  EXPECT_EQ(tasks_of_type<qep::intermediate_state_write_task>(once).size(), 1);
  EXPECT_EQ(once.tasks().size(), twice.tasks().size());
  EXPECT_EQ(tasks_of_type<qep::gather_task>(once).size(),
            tasks_of_type<qep::gather_task>(twice).size());
  EXPECT_EQ(tasks_of_type<qep::intermediate_state_write_task>(once).size(),
            tasks_of_type<qep::intermediate_state_write_task>(twice).size());
}

TEST(QepPipelineNormalizationTest, RejectsInvalidRootsAndCycles)
{
  auto invalid_root =
    qep::query_execution_plan_builder().add_task(std::make_unique<test_fold_task>(1)).build();
  EXPECT_THROW(std::ignore = normalize(invalid_root), std::logic_error);

  auto source      = std::make_unique<test_iterate_task>(1);
  auto left        = std::make_unique<test_stateless_transform_task>(2);
  auto right       = std::make_unique<test_stateless_transform_task>(3);
  auto* source_ptr = source.get();
  auto* left_ptr   = left.get();
  auto* right_ptr  = right.get();
  auto cyclic      = qep::query_execution_plan_builder()
                  .add_task(std::move(source))
                  .add_task(std::move(left))
                  .add_task(std::move(right))
                  .add_successor(source_ptr, left_ptr)
                  .add_successor(left_ptr, right_ptr)
                  .add_successor(right_ptr, left_ptr)
                  .build();
  EXPECT_THROW(std::ignore = normalize(cyclic), std::logic_error);
}

/** @brief Build a streaming bypass that rejoins the branch containing a fold. */
qep::query_execution_plan make_bypass_plan()
{
  auto bypass_source      = std::make_unique<test_iterate_task>(4);
  auto breaker            = std::make_unique<test_fold_task>(5);
  auto bypass             = std::make_unique<test_stateless_transform_task>(6);
  auto rejoin             = std::make_unique<test_stateful_transform_task>(7);
  auto* bypass_source_ptr = bypass_source.get();
  auto* breaker_ptr       = breaker.get();
  auto* bypass_ptr        = bypass.get();
  auto* rejoin_ptr        = rejoin.get();
  return qep::query_execution_plan_builder()
    .add_task(std::move(bypass_source))
    .add_task(std::move(breaker))
    .add_task(std::move(bypass))
    .add_task(std::move(rejoin))
    .add_successor(bypass_source_ptr, breaker_ptr)
    .add_successor(bypass_source_ptr, bypass_ptr)
    .add_successor(breaker_ptr, rejoin_ptr)
    .add_successor(bypass_ptr, rejoin_ptr)
    .build();
}

TEST(QepPipelineNormalizationTest, MaterializesBypassWithoutDuplicatingProducer)
{
  auto bypass_plan        = make_bypass_plan();
  auto const* breaker_ptr = task_with_id(bypass_plan, 5);
  auto const* bypass_ptr  = task_with_id(bypass_plan, 6);
  auto const* rejoin_ptr  = task_with_id(bypass_plan, 7);
  physical::read_relation rejoin_relation({}, {}, "rejoin", nullptr, {});
  auto const physical_relations = associate_with_relation(bypass_plan, &rejoin_relation);
  auto result                   = qep::pipeline_normalization(bypass_plan,
                                            make_output_data_types(bypass_plan),
                                            gqe_test::make_test_task_stable_keys(bypass_plan),
                                            physical_relations);
  auto const& plan              = result.plan();
  EXPECT_EQ(tasks_of_type<test_iterate_task>(plan).size(), 1);
  auto const* source        = task_with_id(plan, 4);
  auto const* fold          = task_with_id(plan, 5);
  auto const* bypass_output = task_with_id(plan, 6);
  auto const* probe         = task_with_id(plan, 7);
  EXPECT_EQ(plan.predecessors(fold), std::vector<qep::task const*>{source});
  auto const inputs = plan.predecessors(probe);
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0], fold);
  ASSERT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(inputs[1]), nullptr);
  auto const* writer = plan.predecessors(inputs[1]).at(0);
  ASSERT_NE(dynamic_cast<qep::intermediate_state_write_task const*>(writer), nullptr);
  auto const* gather = plan.predecessors(writer).at(0);
  EXPECT_EQ(plan.predecessors(gather), std::vector<qep::task const*>{bypass_output});
  auto const& pipelines = result.metadata().pipelines.task_to_pipeline;
  EXPECT_EQ(pipelines.at(source), pipelines.at(fold));
  EXPECT_EQ(pipelines.at(source), pipelines.at(writer));
  EXPECT_NE(pipelines.at(source), pipelines.at(probe));
  EXPECT_EQ(pipelines.at(inputs[1]), pipelines.at(probe));
  EXPECT_EQ(bypass_plan.predecessors(rejoin_ptr),
            (std::vector<qep::task const*>{breaker_ptr, bypass_ptr}));
}

class QepPipelineNormalizationReaderTest : public gqe::test::BaseFixture {};

TEST_F(QepPipelineNormalizationReaderTest, BypassReaderHonorsMorselMaxRows)
{
  auto bypass_plan = make_bypass_plan();
  physical::read_relation rejoin_relation({}, {}, "rejoin", nullptr, {});
  auto const physical_relations = associate_with_relation(bypass_plan, &rejoin_relation);
  auto result                   = qep::pipeline_normalization(bypass_plan,
                                            make_output_data_types(bypass_plan),
                                            gqe_test::make_test_task_stable_keys(bypass_plan),
                                            physical_relations,
                                            3);
  auto const& plan              = result.plan();
  auto const inputs             = plan.predecessors(task_with_id(plan, 7));
  ASSERT_EQ(inputs.size(), 2);
  auto const* reader = dynamic_cast<qep::intermediate_state_read_task const*>(inputs[1]);
  ASSERT_NE(reader, nullptr);

  // Drain the inserted bypass reader and verify contiguous coverage of the input.
  auto input = qep::state_container_builder()
                 .add_state(qep::state_kind::cudf_column{cudf::make_numeric_column(
                              cudf::data_type{cudf::type_id::INT32}, 7)},
                            gqe::query_stream::default_stream())
                 .build();
  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  auto iterator          = reader->initialize(input, ctx_ref);
  cudf::size_type offset = 0;
  for (auto const expected_rows : {3, 3, 1}) {
    auto chunk = reader->next(iterator, ctx_ref);
    ASSERT_TRUE(chunk.has_value());
    auto table = qep::to_table_view(*chunk);
    ASSERT_TRUE(table.has_value());
    EXPECT_EQ(table->num_rows(), expected_rows);
    EXPECT_EQ(table->column(0).offset(), offset);
    offset += expected_rows;
  }
  EXPECT_FALSE(reader->next(iterator, ctx_ref).has_value());
}

TEST(QepPipelineNormalizationTest, RepairsTransitiveBypassesWithSharedStorageAndSeparateReaders)
{
  auto source                 = std::make_unique<test_iterate_task>(1);
  auto first                  = std::make_unique<test_fold_task>(2);
  auto second                 = std::make_unique<test_fold_task>(3);
  auto probe                  = std::make_unique<test_stateful_transform_task>(4);
  auto third                  = std::make_unique<test_fold_task>(5);
  auto later_probe            = std::make_unique<test_stateful_transform_task>(6);
  auto const* source_ptr      = source.get();
  auto const* first_ptr       = first.get();
  auto const* second_ptr      = second.get();
  auto const* probe_ptr       = probe.get();
  auto const* third_ptr       = third.get();
  auto const* later_probe_ptr = later_probe.get();
  auto original               = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(first))
                    .add_task(std::move(second))
                    .add_task(std::move(probe))
                    .add_task(std::move(third))
                    .add_task(std::move(later_probe))
                    .add_successor(source_ptr, first_ptr)
                    .add_successor(first_ptr, second_ptr)
                    .add_successor(second_ptr, probe_ptr)
                    .add_successor(source_ptr, probe_ptr)
                    .add_successor(probe_ptr, third_ptr)
                    .add_successor(third_ptr, later_probe_ptr)
                    .add_successor(source_ptr, later_probe_ptr)
                    .build();
  physical::read_relation relation({}, {}, "cohort", nullptr, {});
  auto result                   = qep::pipeline_normalization(original,
                                            make_output_data_types(original),
                                            gqe_test::make_test_task_stable_keys(original),
                                            associate_with_relation(original, &relation));
  auto expect_repaired_bypasses = [](qep::pipeline_normalization_result const& normalized) {
    auto const& plan         = normalized.plan();
    auto const first_inputs  = plan.predecessors(task_with_id(plan, 4));
    auto const second_inputs = plan.predecessors(task_with_id(plan, 6));
    ASSERT_EQ(first_inputs.size(), 2);
    ASSERT_EQ(second_inputs.size(), 2);
    EXPECT_EQ(first_inputs[0], task_with_id(plan, 3));
    EXPECT_EQ(second_inputs[0], task_with_id(plan, 5));
    auto const* first_reader  = first_inputs[1];
    auto const* second_reader = second_inputs[1];
    ASSERT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(first_reader), nullptr);
    ASSERT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(second_reader), nullptr);
    EXPECT_NE(first_reader, second_reader);
    EXPECT_EQ(plan.predecessors(first_reader), plan.predecessors(second_reader));
    auto const* writer = plan.predecessors(first_reader).at(0);
    ASSERT_NE(dynamic_cast<qep::intermediate_state_write_task const*>(writer), nullptr);
    auto const* gather = plan.predecessors(writer).at(0);
    ASSERT_NE(dynamic_cast<qep::gather_task const*>(gather), nullptr);
    EXPECT_EQ(plan.predecessors(gather), std::vector<qep::task const*>{task_with_id(plan, 1)});
    EXPECT_EQ(tasks_of_type<test_iterate_task>(plan).size(), 1);
    auto const& pipelines = normalized.metadata().pipelines.task_to_pipeline;
    EXPECT_NE(pipelines.at(first_reader), pipelines.at(second_reader));
    EXPECT_NE(pipelines.at(task_with_id(plan, 2)), pipelines.at(task_with_id(plan, 3)));
  };
  expect_repaired_bypasses(result);

  auto const& plan = result.plan();
  auto again       = qep::pipeline_normalization(plan,
                                           make_output_data_types(plan),
                                           gqe_test::make_test_task_stable_keys(plan),
                                           associate_with_relation(plan, &relation));
  EXPECT_EQ(again.plan().tasks().size(), plan.tasks().size());
  expect_repaired_bypasses(again);
}

/** @brief Build the same bypass plan with different task and successor ordering. */
class QepBypassOrderingTest : public ::testing::Test {
 protected:
  enum task_id { left_source_id = 1, right_source_id, merge_id, fold_id, probe_id };

  static qep::query_execution_plan make_plan(std::array<task_id, 5> const& allocation_order,
                                             bool fold_first)
  {
    qep::query_execution_plan_builder builder;
    std::unordered_map<task_id, qep::task const*> tasks;
    for (auto const id : allocation_order) {
      std::unique_ptr<qep::task> task;
      switch (id) {
        case left_source_id:
        case right_source_id: task = std::make_unique<test_iterate_task>(id); break;
        case merge_id:
        case fold_id: task = std::make_unique<test_fold_task>(id); break;
        case probe_id: task = std::make_unique<test_stateful_transform_task>(id); break;
      }
      tasks.emplace(id, task.get());
      builder.add_task(std::move(task));
    }
    auto const* left_source  = tasks.at(left_source_id);
    auto const* right_source = tasks.at(right_source_id);
    auto const* merge        = tasks.at(merge_id);
    auto const* fold         = tasks.at(fold_id);
    auto const* probe        = tasks.at(probe_id);
    builder.add_successor(left_source, merge).add_successor(left_source, probe);
    if (fold_first) {
      builder.add_successor(right_source, fold).add_successor(right_source, merge);
    } else {
      builder.add_successor(right_source, merge).add_successor(right_source, fold);
    }
    builder.add_successor(fold, probe);
    return builder.build();
  }
};

TEST_F(QepBypassOrderingTest, PreservesFanInWhenSelectingBypassCuts)
{
  // left_source  -------> merge <------- right_source
  //      |                                   |
  //      +-------------> probe <----------- fold
  //
  // Both inputs to merge must stay streaming. Materialize left_source -> probe so that
  // probe can wait for fold's materialized result without separating merge's inputs.
  // Task enumeration is unordered. Vary allocation/insertion and successor order to check
  // that this plan succeeds regardless of which independent task normalization visits first.
  std::array allocation_order{left_source_id, right_source_id, merge_id, fold_id, probe_id};
  do {
    for (bool const fold_first : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "allocation order " << ::testing::PrintToString(allocation_order)
                   << ", fold first " << fold_first);
      auto original = make_plan(allocation_order, fold_first);
      physical::read_relation relation({}, {}, "cohort", nullptr, {});
      auto result              = qep::pipeline_normalization(original,
                                                make_output_data_types(original),
                                                gqe_test::make_test_task_stable_keys(original),
                                                associate_with_relation(original, &relation));
      auto const& plan         = result.plan();
      auto const* left_source  = task_with_id(plan, left_source_id);
      auto const* right_source = task_with_id(plan, right_source_id);
      auto const* merge        = task_with_id(plan, merge_id);
      auto const* fold         = task_with_id(plan, fold_id);
      auto const* probe        = task_with_id(plan, probe_id);

      EXPECT_EQ(plan.predecessors(merge),
                (std::vector<qep::task const*>{left_source, right_source}));
      auto const probe_inputs = plan.predecessors(probe);
      ASSERT_EQ(probe_inputs.size(), 2);
      EXPECT_EQ(probe_inputs[1], fold);

      // The other probe input must read the materialized left_source rows.
      auto const* reader = probe_inputs[0];
      ASSERT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(reader), nullptr);
      auto const* writer = plan.predecessors(reader).at(0);
      ASSERT_NE(dynamic_cast<qep::intermediate_state_write_task const*>(writer), nullptr);
      auto const* gather = plan.predecessors(writer).at(0);
      ASSERT_NE(dynamic_cast<qep::gather_task const*>(gather), nullptr);
      EXPECT_EQ(plan.predecessors(gather), std::vector<qep::task const*>{left_source});

      auto const& pipelines = result.metadata().pipelines.task_to_pipeline;
      EXPECT_EQ(pipelines.at(left_source), pipelines.at(merge));
      EXPECT_EQ(pipelines.at(right_source), pipelines.at(merge));
      EXPECT_NE(pipelines.at(left_source), pipelines.at(probe));
    }
  } while (std::ranges::next_permutation(allocation_order).found);
}

TEST(QepPipelineNormalizationTest, PreservesStreamingDiamond)
{
  auto source            = std::make_unique<test_iterate_task>(1);
  auto left              = std::make_unique<test_stateless_transform_task>(2);
  auto right             = std::make_unique<test_stateless_transform_task>(3);
  auto sink              = std::make_unique<test_fold_task>(4);
  auto const* source_ptr = source.get();
  auto const* left_ptr   = left.get();
  auto const* right_ptr  = right.get();
  auto const* sink_ptr   = sink.get();
  auto original          = qep::query_execution_plan_builder()
                    .add_task(std::move(source))
                    .add_task(std::move(left))
                    .add_task(std::move(right))
                    .add_task(std::move(sink))
                    .add_successor(source_ptr, left_ptr)
                    .add_successor(source_ptr, right_ptr)
                    .add_successor(left_ptr, sink_ptr)
                    .add_successor(right_ptr, sink_ptr)
                    .build();
  auto result = normalize(original);
  EXPECT_EQ(result.plan().tasks().size(), original.tasks().size());
  EXPECT_EQ(result.metadata().pipelines.num_pipelines, 1);
}

/** @brief Covers immediate and downstream fan-in with shared or independent streaming inputs. */
class QepBypassFanInTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(QepBypassFanInTest, RejectsReplayWithoutPartitionAndRowCorrespondence)
{
  auto const [has_downstream_fan_in, has_independent_input] = GetParam();
  auto source = std::make_unique<test_iterate_task>(1);
  auto fold   = std::make_unique<test_fold_task>(2);
  auto probe  = std::make_unique<test_stateful_transform_task>(3);
  std::unique_ptr<qep::task> other;
  if (has_independent_input) {
    other = std::make_unique<test_iterate_task>(4);
  } else {
    other = std::make_unique<test_stateless_transform_task>(4);
  }
  auto const* source_ptr = source.get();
  auto const* fold_ptr   = fold.get();
  auto const* probe_ptr  = probe.get();
  auto const* other_ptr  = other.get();
  qep::query_execution_plan_builder builder;
  builder.add_task(std::move(source))
    .add_task(std::move(fold))
    .add_task(std::move(probe))
    .add_task(std::move(other))
    .add_successor(source_ptr, fold_ptr)
    .add_successor(fold_ptr, probe_ptr)
    .add_successor(source_ptr, probe_ptr);
  if (!has_independent_input) { builder.add_successor(source_ptr, other_ptr); }
  if (has_downstream_fan_in) {
    auto sink            = std::make_unique<test_stateless_transform_task>(5);
    auto const* sink_ptr = sink.get();
    builder.add_task(std::move(sink))
      .add_successor(probe_ptr, sink_ptr)
      .add_successor(other_ptr, sink_ptr);
  } else {
    builder.add_successor(other_ptr, probe_ptr);
  }
  auto original = builder.build();
  physical::read_relation relation({}, {}, "cohort", nullptr, {});
  EXPECT_THROW(
    std::ignore = qep::pipeline_normalization(original,
                                              make_output_data_types(original),
                                              gqe_test::make_test_task_stable_keys(original),
                                              associate_with_relation(original, &relation)),
    std::logic_error);
  EXPECT_EQ(original.tasks().size(), has_downstream_fan_in ? 5 : 4);
}

INSTANTIATE_TEST_SUITE_P(ImmediateAndDownstream,
                         QepBypassFanInTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

/** @brief Verify reader behavior, including the schema of an ordered empty result. */
class QepOrderedReaderTest : public gqe::test::BaseFixture {
 protected:
  void expect_single_output(qep::intermediate_state_read_task const& reader)
  {
    gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
    for (auto const rows : {cudf::size_type{7}, cudf::size_type{0}}) {
      auto input = qep::state_container_builder()
                     .add_state(qep::state_kind::cudf_column{cudf::make_numeric_column(
                                  cudf::data_type{cudf::type_id::INT64}, rows)},
                                gqe::query_stream::default_stream())
                     .build();
      auto iterator = reader.initialize(input, ctx_ref);
      auto chunk    = reader.next(iterator, ctx_ref);
      ASSERT_TRUE(chunk.has_value());
      auto table = qep::to_table_view(*chunk);
      ASSERT_TRUE(table.has_value());
      EXPECT_EQ(table->num_rows(), rows);
      ASSERT_EQ(table->num_columns(), 1);
      EXPECT_EQ(table->column(0).type(), cudf::data_type{cudf::type_id::INT64});
      EXPECT_FALSE(reader.next(iterator, ctx_ref).has_value());
    }
  }
};

/** @brief Build a unary plan in reverse storage order to exercise dependency-order analysis. */
qep::query_execution_plan ordered_chain(bool is_top_k, bool has_tail, bool has_explicit_reader)
{
  std::vector<std::unique_ptr<qep::task>> tasks;
  tasks.push_back(std::make_unique<test_iterate_task>(1));
  auto const types = std::vector{cudf::data_type{cudf::type_id::INT64}};
  if (is_top_k) {
    tasks.push_back(std::make_unique<qep::top_k_task>(
      0, cudf::order::ASCENDING, cudf::null_order::AFTER, 0, 5, types));
  } else {
    tasks.push_back(std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                                           std::vector{cudf::order::ASCENDING},
                                                           std::vector{cudf::null_order::AFTER},
                                                           types));
  }
  if (has_tail) {
    std::vector<std::unique_ptr<gqe::expression>> expressions;
    expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
    tasks.push_back(std::make_unique<qep::project_task>(std::move(expressions)));
    tasks.push_back(std::make_unique<qep::filter_task>(std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::literal_expression<int64_t>>(0))));
    // This boundary forces propagation beyond SORT/TOP-K and through inserted writers.
    tasks.push_back(std::make_unique<qep::intermediate_state_read_task>(1));
  }
  if (has_explicit_reader) {
    tasks.push_back(std::make_unique<qep::intermediate_state_read_task>(1));
  }
  tasks.push_back(std::make_unique<qep::fetch_task>(1, 3));
  qep::query_execution_plan_builder builder;
  std::vector<qep::task const*> chain;
  for (auto const& task : tasks) {
    chain.push_back(task.get());
  }
  for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
    builder.add_task(std::move(*it));
  }
  for (std::size_t i = 1; i < chain.size(); ++i) {
    builder.add_successor(chain[i - 1], chain[i]);
  }
  return builder.build();
}

TEST_F(QepOrderedReaderTest, OrderedReadersOverrideRowBoundAndPreserveEmptySchema)
{
  for (bool const is_top_k : {false, true}) {
    for (bool const has_tail : {false, true}) {
      for (bool const has_explicit_reader : {false, true}) {
        SCOPED_TRACE(::testing::Message() << is_top_k << has_tail << has_explicit_reader);
        auto source = ordered_chain(is_top_k, has_tail, has_explicit_reader);
        physical::read_relation input_relation({}, {}, "input", nullptr, {});
        physical::read_relation fetch_relation({}, {}, "fetch", nullptr, {});
        auto relations = associate_with_relation(source, &input_relation);
        for (auto const* task : source.tasks()) {
          if (dynamic_cast<qep::fetch_task const*>(task) != nullptr) {
            relations.at(task) = &fetch_relation;
          }
        }
        auto result  = qep::pipeline_normalization(source,
                                                  make_output_data_types(source),
                                                  gqe_test::make_test_task_stable_keys(source),
                                                  relations,
                                                  1);
        auto readers = tasks_of_type<qep::intermediate_state_read_task>(result.plan());
        ASSERT_FALSE(readers.empty());
        for (auto const* reader : readers) {
          expect_single_output(*reader);
        }
        for (auto const* task : source.tasks()) {
          EXPECT_EQ(task->clone()->ordering_policy(), task->ordering_policy());
        }
      }
    }
  }
}

TEST_F(QepOrderedReaderTest, OrderedReaderIsSharedByPreservingBranches)
{
  auto source = std::make_unique<test_iterate_task>(1);
  auto sort =
    std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                           std::vector{cudf::order::ASCENDING},
                                           std::vector{cudf::null_order::AFTER},
                                           std::vector{cudf::data_type{cudf::type_id::INT64}});
  auto left        = std::make_unique<qep::fetch_task>(0, 3);
  auto right       = std::make_unique<qep::gather_task>();
  auto* source_ptr = source.get();
  auto* sort_ptr   = sort.get();
  auto* left_ptr   = left.get();
  auto* right_ptr  = right.get();
  auto plan        = qep::query_execution_plan_builder()
                .add_task(std::move(source))
                .add_task(std::move(sort))
                .add_task(std::move(left))
                .add_task(std::move(right))
                .add_successor(source_ptr, sort_ptr)
                .add_successor(sort_ptr, left_ptr)
                .add_successor(sort_ptr, right_ptr)
                .build();
  physical::read_relation input_relation({}, {}, "input", nullptr, {});
  physical::read_relation fetch_relation({}, {}, "fetch", nullptr, {});
  auto relations         = associate_with_relation(plan, &input_relation);
  relations.at(left_ptr) = &fetch_relation;
  auto result            = qep::pipeline_normalization(
    plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan), relations, 1);
  auto readers = tasks_of_type<qep::intermediate_state_read_task>(result.plan());
  ASSERT_EQ(readers.size(), 1);
  EXPECT_EQ(result.plan().successors(readers.front()).size(), 2);
  expect_single_output(*readers.front());
}

TEST_F(QepOrderedReaderTest, BypassRepairRetainsOrderedReaderMode)
{
  auto source = std::make_unique<test_iterate_task>(1);
  auto sort =
    std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                           std::vector{cudf::order::ASCENDING},
                                           std::vector{cudf::null_order::AFTER},
                                           std::vector{cudf::data_type{cudf::type_id::INT64}});
  auto fold        = std::make_unique<test_fold_task>(5);
  auto bypass      = std::make_unique<qep::fetch_task>(0, 7);
  auto rejoin      = std::make_unique<test_stateful_transform_task>(7);
  auto* source_ptr = source.get();
  auto* sort_ptr   = sort.get();
  auto* fold_ptr   = fold.get();
  auto* bypass_ptr = bypass.get();
  auto* rejoin_ptr = rejoin.get();
  auto plan        = qep::query_execution_plan_builder()
                .add_task(std::move(source))
                .add_task(std::move(sort))
                .add_task(std::move(fold))
                .add_task(std::move(bypass))
                .add_task(std::move(rejoin))
                .add_successor(source_ptr, sort_ptr)
                .add_successor(sort_ptr, fold_ptr)
                .add_successor(sort_ptr, bypass_ptr)
                .add_successor(fold_ptr, rejoin_ptr)
                .add_successor(bypass_ptr, rejoin_ptr)
                .build();
  physical::read_relation relation({}, {}, "cohort", nullptr, {});
  physical::read_relation fetch_relation({}, {}, "fetch", nullptr, {});
  auto relations           = associate_with_relation(plan, &relation);
  relations.at(bypass_ptr) = &fetch_relation;
  auto result              = qep::pipeline_normalization(
    plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan), relations, 1);
  auto readers = tasks_of_type<qep::intermediate_state_read_task>(result.plan());
  ASSERT_GE(readers.size(), 2);
  for (auto const* reader : readers) {
    expect_single_output(*reader);
  }
}

TEST(QepPipelineNormalizationTest, RejectsOrderedPreservingFanIn)
{
  auto source = std::make_unique<test_iterate_task>(1);
  auto sort =
    std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                           std::vector{cudf::order::ASCENDING},
                                           std::vector{cudf::null_order::AFTER},
                                           std::vector{cudf::data_type{cudf::type_id::INT64}});
  auto other       = std::make_unique<test_iterate_task>(2);
  auto gather      = std::make_unique<qep::gather_task>();
  auto* source_ptr = source.get();
  auto* sort_ptr   = sort.get();
  auto* other_ptr  = other.get();
  auto* gather_ptr = gather.get();
  auto plan        = qep::query_execution_plan_builder()
                .add_task(std::move(source))
                .add_task(std::move(sort))
                .add_task(std::move(other))
                .add_task(std::move(gather))
                .add_successor(source_ptr, sort_ptr)
                .add_successor(sort_ptr, gather_ptr)
                .add_successor(other_ptr, gather_ptr)
                .build();
  EXPECT_THAT(
    [&] {
      std::ignore = qep::pipeline_normalization(
        plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan), {}, 1);
    },
    ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr(
      "pipeline_normalization: preserving ordered output with multiple relational "
      "inputs is currently unsupported and requires an explicit row-alignment contract")));
}

TEST(QepPipelineNormalizationTest, BoundaryClonesRetainIntrinsicOrderingPolicies)
{
  auto const types = std::vector{cudf::data_type{cudf::type_id::INT64}};
  qep::intermediate_state_write_task writer{types};
  qep::intermediate_state_read_task single{qep::intermediate_state_read_task::single_output};
  qep::intermediate_state_read_task bounded{1};
  qep::gather_task gather;
  test_fold_task unknown{1};
  for (qep::task const* task :
       std::array<qep::task const*, 5>{&writer, &single, &bounded, &gather, &unknown}) {
    EXPECT_EQ(task->clone()->ordering_policy(), task->ordering_policy());
  }
  EXPECT_EQ(single.ordering_policy(), qep::task_ordering::preserve);
  EXPECT_EQ(bounded.ordering_policy(), qep::task_ordering::unordered);
  EXPECT_EQ(unknown.ordering_policy(), qep::task_ordering::unordered);
}

TEST_F(QepOrderedReaderTest, UnknownTaskEndsOrderingGuarantee)
{
  auto source = std::make_unique<test_iterate_task>(1);
  auto sort =
    std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                           std::vector{cudf::order::ASCENDING},
                                           std::vector{cudf::null_order::AFTER},
                                           std::vector{cudf::data_type{cudf::type_id::INT64}});
  auto unknown      = std::make_unique<test_stateless_transform_task>(2);
  auto bounded      = std::make_unique<qep::intermediate_state_read_task>(2);
  auto* source_ptr  = source.get();
  auto* sort_ptr    = sort.get();
  auto* unknown_ptr = unknown.get();
  auto* bounded_ptr = bounded.get();
  auto plan         = qep::query_execution_plan_builder()
                .add_task(std::move(source))
                .add_task(std::move(sort))
                .add_task(std::move(unknown))
                .add_task(std::move(bounded))
                .add_successor(source_ptr, sort_ptr)
                .add_successor(sort_ptr, unknown_ptr)
                .add_successor(unknown_ptr, bounded_ptr)
                .build();
  auto result = qep::pipeline_normalization(
    plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan), {}, 1);
  auto readers = tasks_of_type<qep::intermediate_state_read_task>(result.plan());
  ASSERT_EQ(readers.size(), 2);
  auto bounded_reader =
    std::ranges::find_if(readers, [](auto const* reader) { return !reader->is_single_output(); });
  ASSERT_NE(bounded_reader, readers.end());
  auto input = qep::state_container_builder()
                 .add_state(qep::state_kind::cudf_column{cudf::make_numeric_column(
                              cudf::data_type{cudf::type_id::INT64}, 7)},
                            gqe::query_stream::default_stream())
                 .build();
  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  auto iterator = (*bounded_reader)->initialize(input, ctx_ref);
  for (auto const rows : {2, 2, 2, 1}) {
    auto chunk = (*bounded_reader)->next(iterator, ctx_ref);
    ASSERT_TRUE(chunk.has_value());
    auto table = qep::to_table_view(*chunk);
    ASSERT_TRUE(table.has_value());
    EXPECT_EQ(table->num_rows(), rows);
  }
  EXPECT_FALSE((*bounded_reader)->next(iterator, ctx_ref).has_value());
  for (auto const* reader : readers) {
    if (reader->is_single_output()) { expect_single_output(*reader); }
  }
}

TEST_F(QepOrderedReaderTest, StatefulPrivateInitializationDoesNotCountAsRelationalFanIn)
{
  auto source         = std::make_unique<test_iterate_task>(1);
  auto private_source = std::make_unique<test_iterate_task>(2);
  auto sort =
    std::make_unique<qep::merge_sort_task>(std::vector<cudf::size_type>{0},
                                           std::vector{cudf::order::ASCENDING},
                                           std::vector{cudf::null_order::AFTER},
                                           std::vector{cudf::data_type{cudf::type_id::INT64}});
  auto private_state = std::make_unique<test_fold_task>(3);
  auto preserving = std::make_unique<test_stateful_transform_task>(4, qep::task_ordering::preserve);
  auto reader     = std::make_unique<qep::intermediate_state_read_task>(1);
  auto* source_ptr         = source.get();
  auto* private_source_ptr = private_source.get();
  auto* sort_ptr           = sort.get();
  auto* private_state_ptr  = private_state.get();
  auto* preserving_ptr     = preserving.get();
  auto* reader_ptr         = reader.get();
  auto plan                = qep::query_execution_plan_builder()
                .add_task(std::move(source))
                .add_task(std::move(private_source))
                .add_task(std::move(sort))
                .add_task(std::move(private_state))
                .add_task(std::move(preserving))
                .add_task(std::move(reader))
                .add_successor(source_ptr, sort_ptr)
                .add_successor(private_source_ptr, private_state_ptr)
                .add_successor(private_state_ptr, preserving_ptr)
                .add_successor(sort_ptr, preserving_ptr)
                .add_successor(preserving_ptr, reader_ptr)
                .build();
  physical::read_relation input_relation({}, {}, "ordered_input", nullptr, {});
  physical::read_relation private_relation({}, {}, "private_state", nullptr, {});
  auto relations         = associate_with_relation(plan, &private_relation);
  relations.at(sort_ptr) = &input_relation;
  auto result            = qep::pipeline_normalization(
    plan, make_output_data_types(plan), gqe_test::make_test_task_stable_keys(plan), relations, 1);
  auto const inputs = result.plan().predecessors(task_with_id(result.plan(), 4));
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_NE(dynamic_cast<test_fold_task const*>(inputs[0]), nullptr);
  EXPECT_NE(dynamic_cast<qep::intermediate_state_read_task const*>(inputs[1]), nullptr);
  auto readers = tasks_of_type<qep::intermediate_state_read_task>(result.plan());
  ASSERT_EQ(readers.size(), 2);
  for (auto const* ordered_reader : readers) {
    expect_single_output(*ordered_reader);
  }
}
