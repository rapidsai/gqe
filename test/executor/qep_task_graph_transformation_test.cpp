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

#include <gqe/executor/qep_task_graph_transformation.hpp>

#include "../utility.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/qep/operator/filter.hpp>
#include <gqe/qep/operator/hash_aggregate.hpp>
#include <gqe/qep/operator/intermediate_state_write.hpp>
#include <gqe/qep/operator/mark_join.hpp>
#include <gqe/qep/operator/merge_sort.hpp>
#include <gqe/qep/operator/perfect_hash_aggregate.hpp>
#include <gqe/qep/operator/project.hpp>
#include <gqe/qep/pipeline_normalization.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/qep/traits/splittable_source_trait.hpp>
#include <gqe/qep/traits/static_iterate_split_trait.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;
using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;

/**
 * @brief Minimal iterate task that emits a single one-column chunk of the given int64 values
 *        (default `{1, 2, 3}`; an empty `values` yields a schema-correct zero-row chunk).
 */
class single_chunk_iterate_task : public gqe::qep::iterate_task {
 public:
  explicit single_chunk_iterate_task(std::vector<int64_t> values = {1, 2, 3})
    : _values(std::move(values))
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(_values.begin(), _values.end()).release());
    return gqe::qep::state_container_builder()
      .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
      .build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<single_chunk_iterate_task>(_values);
  }

 private:
  std::vector<int64_t> _values;
  mutable bool _emitted = false;
};

/** @brief Records the source ranges selected by the static executor. */
class range_source_task : public single_chunk_iterate_task,
                          public gqe::qep::splittable_source_trait {
 public:
  range_source_task(std::size_t work_count,
                    std::shared_ptr<std::vector<gqe::qep::source_work_range>> selected_ranges)
    : _work_count(work_count), _selected_ranges(std::move(selected_ranges))
  {
  }

  [[nodiscard]] std::size_t work_unit_count() const noexcept override { return _work_count; }

  [[nodiscard]] std::unique_ptr<gqe::qep::iterate_task> split(
    gqe::qep::source_work_range range) const override
  {
    _selected_ranges->push_back(range);
    std::vector<int64_t> values(range.end - range.begin);
    std::iota(values.begin(), values.end(), static_cast<int64_t>(range.begin));
    return std::make_unique<single_chunk_iterate_task>(std::move(values));
  }

  [[nodiscard]] std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<range_source_task>(_work_count, _selected_ranges);
  }

 private:
  std::size_t _work_count;
  std::shared_ptr<std::vector<gqe::qep::source_work_range>> _selected_ranges;
};

/**
 * @brief Minimal schema-bearing source for the merge-sort fold integration test.
 */
class merge_sort_source_task : public gqe::qep::iterate_task {
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
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{30, 10, 20}.release());
    columns.push_back(int64_column_wrapper{3, 1, 2}.release());
    return gqe::qep::state_container_builder()
      .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
      .build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<merge_sort_source_task>();
  }

 private:
  mutable bool _emitted = false;
};

/**
 * @brief Splittable iterate task whose own `next()` yields nothing, but whose `split()`
 *        returns a `single_chunk_iterate_task`, so each lowered split-adapter emits the same
 *        three-row chunk when executed.
 */
class splittable_single_chunk_iterate_task : public gqe::qep::iterate_task,
                                             public gqe::qep::static_iterate_split_trait {
 public:
  explicit splittable_single_chunk_iterate_task(cudf::size_type max_splits)
    : _max_splits(max_splits)
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<splittable_single_chunk_iterate_task>(_max_splits);
  }

  cudf::size_type max_splits() const noexcept override { return _max_splits; }

  std::unique_ptr<gqe::qep::iterate_task> split(cudf::size_type, cudf::size_type) const override
  {
    return std::make_unique<single_chunk_iterate_task>();
  }

 private:
  cudf::size_type _max_splits;
};

/** @brief Atomic task-private counter shared by concurrent adapter calls. */
struct host_count_accumulator : public gqe::qep::task_private_state {
  std::atomic<cudf::size_type> value{0};
};

[[nodiscard]] gqe::qep::state_container make_host_count_accumulator()
{
  return gqe::qep::state_container_builder()
    .add_state(gqe::qep::state_kind::task_private{std::make_unique<host_count_accumulator>()})
    .build();
}

[[nodiscard]] host_count_accumulator& get_host_count_accumulator(
  gqe::qep::state_container_view accumulator)
{
  return gqe::qep::get_task_private_state<host_count_accumulator>(accumulator, 0);
}

/**
 * @brief Minimal fold task that counts the total number of input rows.
 *
 * `initialize` seeds a task-private counter, `next` adds each input's row count to it, and
 * `finalize` emits the result as a `[row_count]` state.
 */
class row_count_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return make_host_count_accumulator();
  }

  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            gqe::query_stream const&,
            rmm::device_async_resource_ref) const override
  {
    get_host_count_accumulator(accumulator).value.fetch_add(gqe::qep::get_row_count(inputs));
  }

  gqe::qep::state_container finalize(gqe::qep::state_container&& accumulator,
                                     gqe::context_reference,
                                     gqe::query_stream const&,
                                     rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(get_host_count_accumulator(accumulator).value.load());
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<row_count_fold_task>();
  }
};

/**
 * @brief Stateless transform task that keeps only the input rows whose value is `> threshold`.
 */
class greater_than_filter_transform_task : public gqe::qep::stateless_transform_task {
 public:
  explicit greater_than_filter_transform_task(int64_t threshold) : _threshold(threshold) {}

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::context_reference,
                                                gqe::query_stream const& stream_handle,
                                                rmm::device_async_resource_ref mr) const override
  {
    auto const stream = stream_handle.ref();
    auto const tv     = gqe::qep::to_table_view(inputs);
    GQE_EXPECTS(tv.has_value(), "greater_than_filter_transform_task: input is not a table");

    cudf::numeric_scalar<int64_t> const threshold{_threshold, /*is_valid=*/true, stream, mr};
    auto const mask = cudf::binary_operation(tv->column(0),
                                             threshold,
                                             cudf::binary_operator::GREATER,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             stream,
                                             mr);
    auto filtered   = cudf::apply_retention_mask(*tv, mask->view(), stream, mr);
    return gqe::qep::state_container_builder()
      .add_state(std::move(*filtered), stream_handle)
      .build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<greater_than_filter_transform_task>(_threshold);
  }

 private:
  int64_t _threshold;
};

/** @brief Stateless-transform stub that signals an early exit on every `next` call. */
class no_output_transform_task : public gqe::qep::stateless_transform_task {
 public:
  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<no_output_transform_task>();
  }
};

/**
 * @brief Minimal stateful transform that maintains a running total of the input rows it has seen.
 *
 * `initialize` seeds a task-private counter from its empty materialized input. `next` adds the
 * pipelined partition's row count and emits the running total as a `[row_count]` state.
 */
class running_row_count_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return make_host_count_accumulator();
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    auto& count           = get_host_count_accumulator(accumulator).value;
    auto const input_rows = gqe::qep::get_row_count(inputs);
    return gqe::qep::make_row_count_container(count.fetch_add(input_rows) + input_rows);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<running_row_count_stateful_transform_task>();
  }
};

/**
 * @brief Check that the actual downstream filter receives and preserves indirect payloads.
 */
class indirect_input_filter_task : public gqe::qep::filter_task {
 public:
  explicit indirect_input_filter_task(std::shared_ptr<std::atomic<int>> calls)
    : filter_task(std::make_unique<gqe::greater_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::literal_expression<int64_t>>(1))),
      _calls{std::move(calls)}
  {
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::context_reference ctx_ref,
                                                gqe::query_stream const& stream_handle,
                                                rmm::device_async_resource_ref mr) const override
  {
    auto const logical = gqe::qep::logical_table_view::try_from(inputs);
    GQE_EXPECTS(
      logical.has_value() && gqe::qep::try_get_indirect_column(logical->column_state(0)) != nullptr,
      "downstream filter must receive indirect mark-join output");
    auto output         = filter_task::next(inputs, ctx_ref, stream_handle, mr);
    auto const filtered = gqe::qep::logical_table_view::try_from(*output);
    EXPECT_EQ(logical->column_state(0), filtered->column_state(0));
    ++*_calls;
    return output;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<indirect_input_filter_task>(_calls);
  }

 private:
  std::shared_ptr<std::atomic<int>> _calls;
};

class QepTaskGraphTransformTest : public gqe::test::BaseFixture {
 protected:
  // Pin `max_num_partitions` so splittable-lowering tests don't depend on the default.
  static constexpr cudf::size_type max_num_partitions = 4;

  QepTaskGraphTransformTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()}
  {
    get_query_ctx()->parameters.num_partitions = max_num_partitions;
  }

  gqe::context_reference ctx_ref;
};

std::unique_ptr<gqe::task_graph> normalize_and_lower(
  gqe::context_reference ctx_ref,
  gqe::qep::query_execution_plan const& plan,
  gqe::qep::task_output_data_types const& output_data_types,
  gqe::qep::task_physical_relations const& physical_relations = {})
{
  auto normalized = gqe::qep::pipeline_normalization(
    plan, output_data_types, gqe_test::make_test_task_stable_keys(plan), physical_relations);
  return gqe::qep_task_graph_transform(ctx_ref, normalized);
}

gqe::qep::task_output_data_types output_types(gqe::qep::task const* task, std::size_t num_columns)
{
  return {{task, std::vector<cudf::data_type>(num_columns, cudf::data_type{cudf::type_id::INT64})}};
}

void execute_all_stages(gqe::task_graph const& graph)
{
  for (auto const& stage : graph.stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
}

/**
 * @brief A single-task QEP is normalized with a terminal writer and executes its iterator.
 */
TEST_F(QepTaskGraphTransformTest, SingleIterateTaskLowers)
{
  auto t      = std::make_unique<single_chunk_iterate_task>();
  auto* t_ptr = t.get();
  auto qep    = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(t_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->root_tasks.size(), 1u);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  EXPECT_EQ(graph->stage_root_tasks[0].size(), 1u);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
  auto result = graph->root_tasks[0]->result();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->num_rows(), 3);
}

TEST_F(QepTaskGraphTransformTest, RejectsDeferredReaderWhenCalledDirectly)
{
  auto source            = std::make_unique<single_chunk_iterate_task>();
  auto const* source_ptr = source.get();
  auto plan = gqe::qep::query_execution_plan_builder().add_task(std::move(source)).build();
  get_query_ctx()->parameters.source_reader = gqe::source_reader_type::deferred;

  EXPECT_THROW(std::ignore = normalize_and_lower(ctx_ref, plan, output_types(source_ptr, 1)),
               std::invalid_argument);
}

/**
 * @brief A splittable iterate task lowers to one adapter per split, capped by
 *        `optimization_parameters::num_partitions`.
 */
TEST_F(QepTaskGraphTransformTest, SplittableIterateLowersIntoMultipleAdapters)
{
  // Request more splits than the cap so the lowering must clamp to `max_num_partitions`.
  auto t      = std::make_unique<splittable_single_chunk_iterate_task>(max_num_partitions + 1);
  auto* t_ptr = t.get();
  auto qep    = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(t_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  EXPECT_EQ(graph->stage_root_tasks[0].size(), static_cast<std::size_t>(max_num_partitions));
  EXPECT_EQ(graph->root_tasks.size(), 1u);
}

/**
 * @brief When the iterator's `max_splits()` is less than
 *        `optimization_parameters::num_partitions`, the lowering uses the iterator's
 *        value (the cap chooses the smaller of the two).
 */
TEST_F(QepTaskGraphTransformTest, SplittableIterateCappedByMaxSplits)
{
  constexpr cudf::size_type iterator_max_splits = max_num_partitions - 1;

  auto t      = std::make_unique<splittable_single_chunk_iterate_task>(iterator_max_splits);
  auto* t_ptr = t.get();
  auto qep    = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(t_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  EXPECT_EQ(graph->stage_root_tasks[0].size(), static_cast<std::size_t>(iterator_max_splits));
  EXPECT_EQ(graph->root_tasks.size(), 1u);
}

/**
 * @brief A two-task `iterate -> fold` QEP lowers to two stages — the accumulate (with the iterate
 *        adapter cascaded in as its dependency) followed by the finalize — and executing the graph
 *        stage by stage runs the fold over the iterator's output.
 */
TEST_F(QepTaskGraphTransformTest, IterateThenFoldLowersAndExecutes)
{
  auto iterate            = std::make_unique<single_chunk_iterate_task>();
  auto fold               = std::make_unique<row_count_fold_task>();
  auto const* iterate_ptr = iterate.get();
  auto const* fold_ptr    = fold.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(fold))
               .add_successor(iterate_ptr, fold_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  // Stage 0: the single accumulate adapter (the iterate adapter is its cascaded dependency).
  // Stage 1: the single finalize adapter, which is also the graph's only root task.
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  // Drive the graph stage by stage: stage 0's accumulate cascades the iterate via
  // `prepare_dependencies`; stage 1's finalize then consumes the completed accumulator.
  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
}

TEST_F(QepTaskGraphTransformTest, IterateThenMergeSortLowersAndExecutes)
{
  auto iterate = std::make_unique<merge_sort_source_task>();
  auto sort    = std::make_unique<gqe::qep::merge_sort_task>(
    std::vector<cudf::size_type>{1},
    std::vector<cudf::order>{cudf::order::ASCENDING},
    std::vector<cudf::null_order>{cudf::null_order::AFTER},
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                    cudf::data_type{cudf::type_id::INT64}});
  auto const* iterate_ptr = iterate.get();
  auto const* sort_ptr    = sort.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(sort))
               .add_successor(iterate_ptr, sort_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2);
  ASSERT_EQ(graph->stage_root_tasks[0].size(), 1);
  ASSERT_EQ(graph->stage_root_tasks[1].size(), 1);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto result = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(result.has_value());
  auto table = gqe::qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_payload{10, 20, 30};
  int64_column_wrapper expected_key{1, 2, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_payload);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_key);
}

/**
 * @brief A fold's materialized output feeding two consumers in different stages is a single root.
 *
 * The lowering registers a pipeline breaker's output as a stage root once, as the breaker is
 * lowered. This test guards that a fan-out output stays a *single* root even when its consumers are
 * spread across different stages — the case a naive "register once per consuming stage" scheme
 * would get wrong by double-registering.
 *
 *     agg_src ─► shared_agg ─┬──────────────────────► join_lo ─► mid_agg ─┐
 *                            │                                            ▼
 *                            └────────────────────────────────────► star_join ◄─ hi_src
 *
 * `shared_agg`'s result is the build (materialized) input of both `join_lo` (shallow) and
 * `star_join` (deep — it also waits on `mid_agg`, a second aggregate built from `join_lo`'s
 * output). Staging therefore places `shared_agg`'s finalize at stage 1, but its two consumers at
 * stages 2 and 4, so the finalize must be exactly one root.
 */
TEST_F(QepTaskGraphTransformTest, FoldFinalizeFansOutToConsumersAtDifferentStages)
{
  auto agg_src    = std::make_unique<single_chunk_iterate_task>();
  auto shared_agg = std::make_unique<row_count_fold_task>();
  auto lo_src     = std::make_unique<single_chunk_iterate_task>();
  auto join_lo    = std::make_unique<running_row_count_stateful_transform_task>();
  auto mid_agg    = std::make_unique<row_count_fold_task>();
  auto hi_src     = std::make_unique<single_chunk_iterate_task>();
  auto star_join  = std::make_unique<running_row_count_stateful_transform_task>();

  auto* agg_src_p    = agg_src.get();
  auto* shared_agg_p = shared_agg.get();
  auto* lo_src_p     = lo_src.get();
  auto* join_lo_p    = join_lo.get();
  auto* mid_agg_p    = mid_agg.get();
  auto* hi_src_p     = hi_src.get();
  auto* star_join_p  = star_join.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(agg_src))
    .add_task(std::move(shared_agg))
    .add_task(std::move(lo_src))
    .add_task(std::move(join_lo))
    .add_task(std::move(mid_agg))
    .add_task(std::move(hi_src))
    .add_task(std::move(star_join));

  builder.add_successor(agg_src_p, shared_agg_p);
  builder.add_successor(shared_agg_p, join_lo_p);  // shallow consumer: join_lo's build side
  builder.add_successor(lo_src_p, join_lo_p);      // join_lo's probe (pipelined)
  builder.add_successor(join_lo_p, mid_agg_p);     // a second aggregate deepens star_join
  builder.add_successor(shared_agg_p,
                        star_join_p);             // deep consumer: star_join's build side (fan-out)
  builder.add_successor(mid_agg_p, star_join_p);  // star_join's other build side
  builder.add_successor(hi_src_p, star_join_p);   // star_join's probe (pipelined)

  auto const qep = builder.build();
  // Treat every task as part of the same physical cohort so the fold inputs remain materialized.
  gqe::physical::read_relation stateful_relation({}, {}, "stateful", nullptr, {});
  gqe::qep::task_physical_relations physical_relations;
  for (auto const* task : qep.tasks()) {
    physical_relations.emplace(task, &stateful_relation);
  }
  auto const graph =
    normalize_and_lower(ctx_ref, qep, output_types(star_join_p, 0), physical_relations);

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{6});

  std::vector<std::size_t> root_counts;
  std::unordered_set<int32_t> distinct_root_ids;
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    root_counts.push_back(stages[stage].size());
    for (auto const& weak_root : stages[stage]) {
      auto const root = weak_root.lock();
      ASSERT_TRUE(root);
      // A root is laid out in its own stage's slot...
      EXPECT_EQ(root->stage_id(), static_cast<int32_t>(stage));
      // ...belongs to exactly one pipeline (the invariant a double-registered fan-out producer
      // would violate)...
      EXPECT_EQ(root->pipeline_ids().size(), std::size_t{1});
      // ...and is registered at most once across all stages.
      EXPECT_TRUE(distinct_root_ids.insert(root->task_id()).second);
    }
  }

  // One root per stage: shared_agg accumulate/finalize (stages 0/1), mid_agg accumulate/finalize
  // (stages 2/3), star_join (stage 4), and its inserted terminal writer (stage 5).
  EXPECT_EQ(root_counts, (std::vector<std::size_t>{1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(distinct_root_ids.size(), std::size_t{6});

  // Driving the six stages in order runs the whole graph; star_join emits the running count of its
  // probe (hi_src's three rows).
  for (auto const& stage : stages) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
}

/**
 * @brief A fold's N partitioned partial aggregates fan in to a single merged finalize.
 *
 * Shape: `src ─► fold ─► sink`, where `src` is a splittable source that lowers to N parallel
 * partition adapters. The fold therefore lowers to N `accumulate` adapters (the fan-in) merged
 * by one `finalize`:
 *
 *     stage 0: N accumulates   (parallel partial aggregation — N stage roots)
 *     stage 1: finalize        (merged once after the stage-0 barrier)
 *     stage 2: sink            (consumes the finalize)
 *
 * This exercises the N→1 fan-in side: the accumulates are independent stage roots co-scheduled
 * in one stage, and `finalize` — depending on all of them across the barrier — runs once in the
 * next stage.
 */
TEST_F(QepTaskGraphTransformTest, FoldFansInParallelAccumulatesIntoOneFinalize)
{
  constexpr cudf::size_type requested_splits = max_num_partitions;
  // The transform clamps the split count by `max_num_partitions`; mirror that to know N.
  auto const cap        = static_cast<cudf::size_type>(get_query_ctx()->parameters.num_partitions);
  auto const partitions = std::max<cudf::size_type>(1, std::min(requested_splits, cap));

  auto src  = std::make_unique<splittable_single_chunk_iterate_task>(requested_splits);
  auto fold = std::make_unique<row_count_fold_task>();
  auto sink = std::make_unique<single_chunk_iterate_task>();

  auto* src_p  = src.get();
  auto* fold_p = fold.get();
  auto* sink_p = sink.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(src)).add_task(std::move(fold)).add_task(std::move(sink));
  builder.add_successor(src_p, fold_p);
  builder.add_successor(fold_p, sink_p);

  auto const qep   = builder.build();
  auto const graph = normalize_and_lower(ctx_ref, qep, output_types(sink_p, 1));

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{4});

  std::vector<std::size_t> root_counts;
  std::unordered_set<int32_t> distinct_root_ids;
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    root_counts.push_back(stages[stage].size());
    for (auto const& weak_root : stages[stage]) {
      auto const root = weak_root.lock();
      ASSERT_TRUE(root);
      EXPECT_EQ(root->stage_id(), static_cast<int32_t>(stage));
      EXPECT_EQ(root->pipeline_ids().size(), std::size_t{1});
      EXPECT_TRUE(distinct_root_ids.insert(root->task_id()).second);
    }
  }

  EXPECT_EQ(root_counts, (std::vector<std::size_t>{static_cast<std::size_t>(partitions), 1, 1, 1}));
  EXPECT_EQ(distinct_root_ids.size(), static_cast<std::size_t>(partitions) + 3);

  // Drive every stage and read the fan-in result from the finalize: each of the N partitions emits
  // a three-row chunk, so the finalize sums them to 3 * N.
  auto finalize = stages[1].front().lock();
  ASSERT_NE(finalize, nullptr);
  for (auto const& stage : stages) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
  auto state = finalize->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3 * partitions));
}

/**
 * @brief A diamond — one source fanning out to two branches that reconverge — lowers correctly.
 *
 * `src` feeds two pipelined transforms, and a `fold` reconverges them. The whole diamond is a
 * single pipeline (connected by pipelined edges), so the partition must treat it as one pipeline
 * rather than choke on the undirected cycle, and the fold must wire *both* branches as dependencies
 * while they share the one `src` adapter at the apex.
 *
 *     src ─┬─► tL ─┐
 *          │       ├─► fold
 *          └─► tR ─┘
 */
TEST_F(QepTaskGraphTransformTest, DiamondFanOutReconvergesAtFold)
{
  auto src  = std::make_unique<single_chunk_iterate_task>();
  auto tL   = std::make_unique<greater_than_filter_transform_task>(/*threshold=*/0);
  auto tR   = std::make_unique<greater_than_filter_transform_task>(/*threshold=*/0);
  auto fold = std::make_unique<row_count_fold_task>();

  auto* src_p  = src.get();
  auto* tL_p   = tL.get();
  auto* tR_p   = tR.get();
  auto* fold_p = fold.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(src))
    .add_task(std::move(tL))
    .add_task(std::move(tR))
    .add_task(std::move(fold));
  builder.add_successor(src_p, tL_p);  // fan out
  builder.add_successor(src_p, tR_p);
  builder.add_successor(tL_p, fold_p);  // reconverge
  builder.add_successor(tR_p, fold_p);

  auto const qep   = builder.build();
  auto const graph = normalize_and_lower(ctx_ref, qep, {});

  // One pipeline broken once by the fold: accumulate at stage 0, finalize at stage 1.
  ASSERT_EQ(graph->stage_root_tasks.size(), std::size_t{2});
  ASSERT_EQ(graph->root_tasks.size(), std::size_t{1});

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  // The fold reconverges both branches: it counts the rows of the two filtered views of `src`
  // (both keep all three rows), horizontally concatenated into one input.
  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
}

/**
 * @brief A materializing diamond places two folds at different DAG depths in the same stage.
 *
 * `src` fans out to `fold1` directly and to `fold2` through a pipelined `filter` — so `fold2` sits
 * one pipelined task deeper than `fold1`. But pipelined edges do not advance the stage, so `src`,
 * `filter`, and both fold accumulate sides form a single stage-0 pipeline; only the two `fold ─►
 * sink` breaker edges cross stages. Both folds therefore accumulate at stage 0 and finalize at
 * stage 1, and `sink` (reconverging both materialized results) sits at stage 2.
 *
 *     src ─┬──────────────► fold1 ─┐
 *          │                       ├─► sink
 *          └─► filter ─► fold2 ────┘
 */
TEST_F(QepTaskGraphTransformTest, MaterializingDiamondCoSchedulesFolds)
{
  auto src    = std::make_unique<single_chunk_iterate_task>();
  auto fold1  = std::make_unique<row_count_fold_task>();
  auto filter = std::make_unique<greater_than_filter_transform_task>(/*threshold=*/0);
  auto fold2  = std::make_unique<row_count_fold_task>();
  auto sink   = std::make_unique<single_chunk_iterate_task>();

  auto* src_p    = src.get();
  auto* fold1_p  = fold1.get();
  auto* filter_p = filter.get();
  auto* fold2_p  = fold2.get();
  auto* sink_p   = sink.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(src))
    .add_task(std::move(fold1))
    .add_task(std::move(filter))
    .add_task(std::move(fold2))
    .add_task(std::move(sink));

  builder.add_successor(src_p, fold1_p);   // shallow branch
  builder.add_successor(src_p, filter_p);  // deeper branch (one pipelined task further)
  builder.add_successor(filter_p, fold2_p);
  builder.add_successor(fold1_p, sink_p);  // sink reconverges both materialized aggregates
  builder.add_successor(fold2_p, sink_p);

  auto const qep   = builder.build();
  auto const graph = normalize_and_lower(ctx_ref, qep, output_types(sink_p, 1));

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{4});

  std::vector<std::size_t> root_counts;
  std::unordered_set<int32_t> distinct_root_ids;
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    root_counts.push_back(stages[stage].size());
    for (auto const& weak_root : stages[stage]) {
      auto const root = weak_root.lock();
      ASSERT_TRUE(root);
      EXPECT_EQ(root->stage_id(), static_cast<int32_t>(stage));
      EXPECT_EQ(root->pipeline_ids().size(), std::size_t{1});
      EXPECT_TRUE(distinct_root_ids.insert(root->task_id()).second);
    }
  }

  // Both folds share stage 0 (accumulates) and stage 1 (finalizes); sink runs at stage 2 and its
  // inserted terminal writer finalizes at stage 3.
  EXPECT_EQ(root_counts, (std::vector<std::size_t>{2, 2, 1, 1}));
  EXPECT_EQ(distinct_root_ids.size(), std::size_t{6});

  // Drive every stage and read each fold's count from its finalize: `src` and `src` filtered by
  // `> 0` both keep all three rows, so both stage-1 finalizes hold a count of 3.
  std::vector<std::shared_ptr<gqe::task>> finalizes;
  for (auto const& weak_finalize : stages[1]) {
    auto finalize = weak_finalize.lock();
    ASSERT_NE(finalize, nullptr);
    finalizes.push_back(std::move(finalize));
  }
  for (auto const& stage : stages) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }
  for (auto const& finalize : finalizes) {
    auto state = finalize->qep_state_result();
    ASSERT_TRUE(state.has_value());
    EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
  }
}

/**
 * @brief Like `IterateThenFoldLowersAndExecutes`, but the iterate emits a schema-correct zero-row
 *        chunk: the fold still lowers and executes, producing a row count of 0.
 */
TEST_F(QepTaskGraphTransformTest, IterateThenFoldEmptyInputLowersAndExecutes)
{
  auto iterate            = std::make_unique<single_chunk_iterate_task>(std::vector<int64_t>{});
  auto fold               = std::make_unique<row_count_fold_task>();
  auto const* iterate_ptr = iterate.get();
  auto const* fold_ptr    = fold.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(fold))
               .add_successor(iterate_ptr, fold_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(0));
}

/**
 * @brief An `iterate -> stateless_transform` pipeline executes through its normalized writer.
 */
TEST_F(QepTaskGraphTransformTest, IterateThenTransformLowersAndExecutes)
{
  auto iterate            = std::make_unique<single_chunk_iterate_task>();
  auto transform          = std::make_unique<greater_than_filter_transform_task>(/*threshold=*/1);
  auto const* iterate_ptr = iterate.get();
  auto const* xform_ptr   = transform.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(transform))
               .add_successor(iterate_ptr, xform_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(xform_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  execute_all_stages(*graph);

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 2);  // {1, 2, 3} filtered to {2, 3}.
}

TEST_F(QepTaskGraphTransformTest, IterateThenQepFilterLowersAndExecutes)
{
  auto iterate   = std::make_unique<single_chunk_iterate_task>();
  auto condition = std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::literal_expression<int64_t>>(1));
  auto filter                 = std::make_unique<gqe::qep::filter_task>(std::move(condition));
  auto const* iterate_ptr     = iterate.get();
  auto const* filter_task_ptr = filter.get();
  auto qep                    = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(filter))
               .add_successor(iterate_ptr, filter_task_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(filter_task_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  execute_all_stages(*graph);
  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto table = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected{2, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
}

TEST_F(QepTaskGraphTransformTest, IterateThenQepProjectLowersAndExecutes)
{
  auto iterate = std::make_unique<single_chunk_iterate_task>();
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(std::make_unique<gqe::column_reference_expression>(0));
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(5)));
  auto project                 = std::make_unique<gqe::qep::project_task>(std::move(expressions));
  auto const* iterate_ptr      = iterate.get();
  auto const* project_task_ptr = project.get();
  auto qep                     = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(project))
               .add_successor(iterate_ptr, project_task_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(project_task_ptr, 2));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  execute_all_stages(*graph);
  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto table = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 2);
  int64_column_wrapper expected_reference{1, 2, 3};
  int64_column_wrapper expected_computed{6, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_reference);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_computed);
}

TEST_F(QepTaskGraphTransformTest, IterateFilterProjectMaterializesTerminalMask)
{
  auto iterate   = std::make_unique<single_chunk_iterate_task>();
  auto condition = std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::literal_expression<int64_t>>(1));
  auto filter = std::make_unique<gqe::qep::filter_task>(std::move(condition));
  std::vector<std::unique_ptr<gqe::expression>> expressions;
  expressions.push_back(
    std::make_unique<gqe::add_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                          std::make_shared<gqe::literal_expression<int64_t>>(10)));
  auto project                 = std::make_unique<gqe::qep::project_task>(std::move(expressions));
  auto const* iterate_ptr      = iterate.get();
  auto const* filter_task_ptr  = filter.get();
  auto const* project_task_ptr = project.get();
  auto qep                     = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(filter))
               .add_task(std::move(project))
               .add_successor(iterate_ptr, filter_task_ptr)
               .add_successor(filter_task_ptr, project_task_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(project_task_ptr, 1));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  execute_all_stages(*graph);
  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto table = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_values{{12, 13}, {true, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_values);
}

/**
 * @brief A three-task `iterate -> stateless_transform -> fold` QEP lowers to two stages. The
 *        transform stays in the iterate/accumulate stage, and the fold breaks the pipeline into a
 *        finalize stage. Executing the graph folds over the transform's filtered output.
 */
TEST_F(QepTaskGraphTransformTest, IterateTransformFoldLowersAndExecutes)
{
  auto iterate            = std::make_unique<single_chunk_iterate_task>();
  auto transform          = std::make_unique<greater_than_filter_transform_task>(/*threshold=*/1);
  auto fold               = std::make_unique<row_count_fold_task>();
  auto const* iterate_ptr = iterate.get();
  auto const* xform_ptr   = transform.get();
  auto const* fold_ptr    = fold.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(transform))
               .add_task(std::move(fold))
               .add_successor(iterate_ptr, xform_ptr)
               .add_successor(xform_ptr, fold_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  // Stage 0: the accumulate adapter (iterate + transform adapters cascaded as its dependencies).
  // Stage 1: the finalize adapter, which is also the graph's only root task.
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(2));  // {2, 3} -> count 2.
}

TEST_F(QepTaskGraphTransformTest, PerfectHashApplyConcatApplyLowersAndExecutes)
{
  constexpr cudf::size_type partitions = 2;
  auto iterate = std::make_unique<splittable_single_chunk_iterate_task>(partitions);
  auto partial = std::make_unique<gqe::qep::perfect_hash_partial_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{
      {cudf::aggregation::SUM, 0}, {cudf::aggregation::COUNT_ALL, 0}},
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64}});
  auto combine = std::make_unique<gqe::qep::perfect_hash_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1},
                                                                     {cudf::aggregation::SUM, 2}},
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64}});
  auto const* iterate_ptr = iterate.get();
  auto const* partial_ptr = partial.get();
  auto const* combine_ptr = combine.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(partial))
               .add_task(std::move(combine))
               .add_successor(iterate_ptr, partial_ptr)
               .add_successor(partial_ptr, combine_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->stage_root_tasks[0].size(), 2u);
  ASSERT_EQ(graph->stage_root_tasks[1].size(), 1u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto result = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(result.has_value());
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{1, 2, 3}.release());
  expected_columns.push_back(int64_column_wrapper{2, 4, 6}.release());
  expected_columns.push_back(int64_column_wrapper{2, 2, 2}.release());
  auto expected        = cudf::table{std::move(expected_columns)};
  auto sorted_result   = cudf::sort(*result);
  auto sorted_expected = cudf::sort(expected.view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_result->view(), sorted_expected->view());
}

TEST_F(QepTaskGraphTransformTest, HashApplyConcatApplyLowersAndExecutes)
{
  constexpr cudf::size_type partitions = 2;
  auto iterate = std::make_unique<splittable_single_chunk_iterate_task>(partitions);
  auto partial = std::make_unique<gqe::qep::hash_partial_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{
      {cudf::aggregation::SUM, 0}, {cudf::aggregation::COUNT_ALL, 0}});
  auto combine = std::make_unique<gqe::qep::hash_aggregate_task>(
    std::vector<cudf::size_type>{0},
    std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>>{{cudf::aggregation::SUM, 1},
                                                                     {cudf::aggregation::SUM, 2}},
    std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64},
                                 cudf::data_type{cudf::type_id::INT64}});
  auto const* iterate_ptr = iterate.get();
  auto const* partial_ptr = partial.get();
  auto const* combine_ptr = combine.get();
  auto qep                = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(partial))
               .add_task(std::move(combine))
               .add_successor(iterate_ptr, partial_ptr)
               .add_successor(partial_ptr, combine_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, {});
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->stage_root_tasks[0].size(), 2u);
  ASSERT_EQ(graph->stage_root_tasks[1].size(), 1u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);
  execute_all_stages(*graph);

  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto result = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(result.has_value());
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{1, 2, 3}.release());
  expected_columns.push_back(int64_column_wrapper{2, 4, 6}.release());
  expected_columns.push_back(int64_column_wrapper{2, 2, 2}.release());
  auto expected        = cudf::table{std::move(expected_columns)};
  auto sorted_result   = cudf::sort(*result);
  auto sorted_expected = cudf::sort(expected.view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_result->view(), sorted_expected->view());
}

/**
 * @brief A stateful transform reachable only through a pipelined iterate lowers and executes
 *        end-to-end over the iterator's output.
 *
 * The iterate is a data pipeline, not a breaker, so per the QEP contract it is a pipelined input
 * feeding `next`. With no pipeline-breaker predecessor, `initialize` receives an empty materialized
 * input. Normalization appends a count-only writer.
 */
TEST_F(QepTaskGraphTransformTest, IterateThenStatefulTransformLowersAndExecutes)
{
  auto iterate              = std::make_unique<single_chunk_iterate_task>();
  auto transform            = std::make_unique<running_row_count_stateful_transform_task>();
  auto const* iterate_ptr   = iterate.get();
  auto const* transform_ptr = transform.get();
  auto qep                  = gqe::qep::query_execution_plan_builder()
               .add_task(std::move(iterate))
               .add_task(std::move(transform))
               .add_successor(iterate_ptr, transform_ptr)
               .build();

  auto graph = normalize_and_lower(ctx_ref, qep, output_types(transform_ptr, 0));
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->stage_root_tasks.size(), 2u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  for (auto const& stage : graph->stage_root_tasks) {
    for (auto const& weak_root : stage) {
      auto root = weak_root.lock();
      ASSERT_NE(root, nullptr);
      root->execute();
    }
  }

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(3));
}

TEST_F(QepTaskGraphTransformTest, NormalizedFoldToWriterExecutes)
{
  auto iterate =
    std::make_unique<single_chunk_iterate_task>(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7});
  auto fold = std::make_unique<row_count_fold_task>();
  auto writer =
    std::make_unique<gqe::qep::intermediate_state_write_task>(std::vector<cudf::data_type>{});
  auto const* iterate_ptr = iterate.get();
  auto const* fold_ptr    = fold.get();
  auto const* writer_ptr  = writer.get();
  auto raw_qep            = gqe::qep::query_execution_plan_builder()
                   .add_task(std::move(iterate))
                   .add_task(std::move(fold))
                   .add_task(std::move(writer))
                   .add_successor(iterate_ptr, fold_ptr)
                   .add_successor(fold_ptr, writer_ptr)
                   .build();
  gqe::qep::task_output_data_types output_data_types{
    {iterate_ptr, {cudf::data_type{cudf::type_id::INT64}}}, {fold_ptr, {}}, {writer_ptr, {}}};
  auto normalized = gqe::qep::pipeline_normalization(
    raw_qep, output_data_types, gqe_test::make_test_task_stable_keys(raw_qep));

  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  execute_all_stages(*graph);

  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  EXPECT_THAT(gqe::qep::try_row_count(*state), testing::Optional(7));
}

TEST_F(QepTaskGraphTransformTest, NormalizedMixedFoldInputsUseMatchingAdapterCounts)
{
  auto materialized_source = std::make_unique<single_chunk_iterate_task>();
  auto materialized        = std::make_unique<row_count_fold_task>();
  auto pipelined_source =
    std::make_unique<splittable_single_chunk_iterate_task>(max_num_partitions);
  auto successor                = std::make_unique<row_count_fold_task>();
  auto* materialized_source_ptr = materialized_source.get();
  auto* materialized_ptr        = materialized.get();
  auto* pipelined_source_ptr    = pipelined_source.get();
  auto* successor_ptr           = successor.get();
  auto raw_qep                  = gqe::qep::query_execution_plan_builder()
                   .add_task(std::move(materialized_source))
                   .add_task(std::move(materialized))
                   .add_task(std::move(pipelined_source))
                   .add_task(std::move(successor))
                   .add_successor(materialized_source_ptr, materialized_ptr)
                   .add_successor(materialized_ptr, successor_ptr)
                   .add_successor(pipelined_source_ptr, successor_ptr)
                   .build();

  gqe::qep::task_output_data_types output_data_types{
    {materialized_source_ptr, {cudf::data_type{cudf::type_id::INT64}}},
    {materialized_ptr, {}},
    {pipelined_source_ptr, {cudf::data_type{cudf::type_id::INT64}}},
    {successor_ptr, {}}};
  auto normalized = gqe::qep::pipeline_normalization(
    raw_qep, output_data_types, gqe_test::make_test_task_stable_keys(raw_qep));
  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
}

TEST_F(QepTaskGraphTransformTest, NormalizedAbsentOutputEmitsSchemaCorrectEmptyTable)
{
  auto iterate              = std::make_unique<single_chunk_iterate_task>();
  auto transform            = std::make_unique<no_output_transform_task>();
  auto const* iterate_ptr   = iterate.get();
  auto const* transform_ptr = transform.get();
  auto raw_qep              = gqe::qep::query_execution_plan_builder()
                   .add_task(std::move(iterate))
                   .add_task(std::move(transform))
                   .add_successor(iterate_ptr, transform_ptr)
                   .build();
  gqe::qep::task_output_data_types output_data_types{
    {iterate_ptr, {cudf::data_type{cudf::type_id::INT64}}},
    {transform_ptr, {cudf::data_type{cudf::type_id::INT64}}}};
  auto normalized = gqe::qep::pipeline_normalization(
    raw_qep, output_data_types, gqe_test::make_test_task_stable_keys(raw_qep));

  auto graph = gqe::qep_task_graph_transform(ctx_ref, normalized);
  ASSERT_NE(graph, nullptr);
  execute_all_stages(*graph);

  auto state = graph->root_tasks.front()->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto result = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->column(0).type().id(), cudf::type_id::INT64);
}

TEST_F(QepTaskGraphTransformTest, MarkJoinPartitionsReachFilterWithoutGather)
{
  namespace qep                        = gqe::qep;
  constexpr cudf::size_type partitions = 3;
  for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto left     = std::make_unique<splittable_single_chunk_iterate_task>(partitions);
    auto right    = std::make_unique<single_chunk_iterate_task>(std::vector<int64_t>{1, 3});
    auto build    = std::make_unique<qep::mark_join_build_task>(std::vector<cudf::size_type>{0},
                                                             cudf::null_equality::UNEQUAL);
    auto probe    = std::make_unique<qep::mark_join_probe_task>(std::vector<cudf::size_type>{0});
    auto complete = std::make_unique<qep::mark_join_complete_task>();
    auto scan     = std::make_unique<qep::mark_join_scan_task>(
      type,
      std::vector<cudf::size_type>{0},
      std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT64}});
    auto calls       = std::make_shared<std::atomic<int>>(0);
    auto filter      = std::make_unique<indirect_input_filter_task>(calls);
    auto* left_p     = left.get();
    auto* right_p    = right.get();
    auto* build_p    = build.get();
    auto* probe_p    = probe.get();
    auto* complete_p = complete.get();
    auto* scan_p     = scan.get();
    auto* filter_p   = filter.get();
    auto plan        = qep::query_execution_plan_builder()
                  .add_task(std::move(left))
                  .add_task(std::move(right))
                  .add_task(std::move(build))
                  .add_task(std::move(probe))
                  .add_task(std::move(complete))
                  .add_task(std::move(scan))
                  .add_task(std::move(filter))
                  .add_successor(left_p, build_p)
                  .add_successor(build_p, probe_p)
                  .add_successor(right_p, probe_p)
                  .add_successor(probe_p, complete_p)
                  .add_successor(build_p, scan_p)
                  .add_successor(complete_p, scan_p)
                  .add_successor(scan_p, filter_p)
                  .build();
    gqe::physical::read_relation relation({}, {}, "mark", nullptr, {});
    qep::task_physical_relations relations;
    for (auto const* task : plan.tasks()) {
      relations.emplace(task, &relation);
    }
    auto graph = normalize_and_lower(ctx_ref, plan, output_types(filter_p, 1), relations);
    execute_all_stages(*graph);
    EXPECT_EQ(calls->load(), partitions);
    auto result = graph->root_tasks.front()->qep_state_result();
    ASSERT_TRUE(result.has_value());
    auto table = qep::to_table_view(*result);
    ASSERT_TRUE(table.has_value());
    std::vector<int64_t> values(partitions, type == gqe::join_type_type::left_semi ? 3 : 2);
    int64_column_wrapper expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
  }
}

TEST(SourceWorkRangeTest, BalancesOffsetRangesAndEmptyAssignments)
{
  using gqe::qep::balanced_source_work_range;
  using gqe::qep::source_work_range;
  EXPECT_EQ(balanced_source_work_range({5, 10}, 0, 3), (source_work_range{5, 7}));
  EXPECT_EQ(balanced_source_work_range({5, 10}, 1, 3), (source_work_range{7, 9}));
  EXPECT_EQ(balanced_source_work_range({5, 10}, 2, 3), (source_work_range{9, 10}));
  EXPECT_EQ(balanced_source_work_range({5, 10}, 6, 7), (source_work_range{10, 10}));
  EXPECT_EQ(balanced_source_work_range({7, 7}, 1, 3), (source_work_range{7, 7}));
  auto const max = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(balanced_source_work_range({max - 5, max}, 2, 3), (source_work_range{max - 1, max}));
  EXPECT_THROW(std::ignore = balanced_source_work_range({2, 1}, 0, 1), std::invalid_argument);
  EXPECT_THROW(std::ignore = balanced_source_work_range({0, 1}, 0, 0), std::invalid_argument);
  EXPECT_THROW(std::ignore = balanced_source_work_range({0, 1}, 2, 2), std::invalid_argument);
}

TEST_F(QepTaskGraphTransformTest, SourceRangesCoverInputIncludingEmptySources)
{
  for (std::size_t work_count : {0, 2, 5}) {
    auto selected_ranges   = std::make_shared<std::vector<gqe::qep::source_work_range>>();
    auto source            = std::make_unique<range_source_task>(work_count, selected_ranges);
    auto const* source_ptr = source.get();
    auto qep   = gqe::qep::query_execution_plan_builder().add_task(std::move(source)).build();
    auto graph = normalize_and_lower(ctx_ref, qep, output_types(source_ptr, 1));
    ASSERT_NE(graph, nullptr);
    auto const expected_count = static_cast<std::size_t>(max_num_partitions);
    ASSERT_EQ(selected_ranges->size(), expected_count);
    std::size_t next = 0;
    for (auto range : *selected_ranges) {
      EXPECT_EQ(range.begin, next);
      EXPECT_LE(range.begin, range.end);
      next = range.end;
    }
    EXPECT_EQ(next, work_count);
    if (work_count == 5) {
      EXPECT_EQ(*selected_ranges,
                (std::vector<gqe::qep::source_work_range>{{0, 2}, {2, 3}, {3, 4}, {4, 5}}));
    }
    execute_all_stages(*graph);
    auto result = graph->root_tasks.front()->result();
    ASSERT_TRUE(result);
    std::vector<int64_t> values(work_count);
    std::iota(values.begin(), values.end(), int64_t{0});
    int64_column_wrapper expected(values.begin(), values.end());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT((cudf::table_view{{expected}}), *result);
  }
}

TEST_F(QepTaskGraphTransformTest, ZippedSourcesUseSameSplitCountWithDifferentWorkCounts)
{
  for (auto const partitions : {0, max_num_partitions}) {
    get_query_ctx()->parameters.num_partitions = partitions;
    for (std::size_t first_work_count : {0, 2}) {
      auto first_ranges      = std::make_shared<std::vector<gqe::qep::source_work_range>>();
      auto second_ranges     = std::make_shared<std::vector<gqe::qep::source_work_range>>();
      auto first             = std::make_unique<range_source_task>(first_work_count, first_ranges);
      auto second            = std::make_unique<range_source_task>(5, second_ranges);
      auto transform         = std::make_unique<greater_than_filter_transform_task>(0);
      auto const* first_ptr  = first.get();
      auto const* second_ptr = second.get();
      auto const* transform_ptr = transform.get();
      auto qep                  = gqe::qep::query_execution_plan_builder()
                   .add_task(std::move(first))
                   .add_task(std::move(second))
                   .add_task(std::move(transform))
                   .add_successor(first_ptr, transform_ptr)
                   .add_successor(second_ptr, transform_ptr)
                   .build();

      auto graph = normalize_and_lower(ctx_ref, qep, output_types(transform_ptr, 2));
      ASSERT_NE(graph, nullptr);
      auto const expected_count = static_cast<std::size_t>(std::max(1, partitions));
      EXPECT_EQ(first_ranges->size(), expected_count);
      EXPECT_EQ(second_ranges->size(), expected_count);
    }
  }
}
