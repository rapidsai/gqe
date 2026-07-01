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

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/qep/traits/splittable_iterate_trait.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

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
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    if (_emitted) { return std::nullopt; }
    _emitted = true;
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper(_values.begin(), _values.end()).release());
    return gqe::qep::state_container_builder().add_state(cudf::table{std::move(columns)}).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<single_chunk_iterate_task>(_values);
  }

 private:
  std::vector<int64_t> _values;
  mutable bool _emitted = false;
};

/**
 * @brief Splittable iterate task whose own `next()` yields nothing, but whose `split()`
 *        returns a `single_chunk_iterate_task` (so each lowered split-adapter emits one
 *        chunk if executed). Tests using this stub only inspect the lowering shape
 *        (number of adapters), so the runtime content isn't exercised by the stub itself.
 */
class splittable_single_chunk_iterate_task : public gqe::qep::iterate_task,
                                             public gqe::qep::splittable_iterate_trait {
 public:
  explicit splittable_single_chunk_iterate_task(cudf::size_type max_splits)
    : _max_splits(max_splits)
  {
  }

  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
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

/**
 * @brief Minimal fold task that counts the total number of input rows.
 *
 * `initialize` seeds a `[row_count{0}]` accumulator, `next` adds each input's row count to it, and
 * `finalize` (the base default) returns the accumulator unchanged.
 */
class row_count_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  void next(gqe::qep::state_container_view inputs,
            gqe::qep::state_container_view accumulator,
            gqe::context_reference,
            rmm::cuda_stream_view,
            rmm::device_async_resource_ref) const override
  {
    // The accumulator is a single `row_count` slot; bump it by this input's row count. The inner
    // shared_ptr is const but its pointee is mutable, per the `state_container_view` contract.
    auto& count = std::get<gqe::qep::state_kind::row_count>(*accumulator.front());
    count.value += gqe::qep::get_row_count(inputs);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<row_count_fold_task>();
  }
};

/**
 * @brief optional_transform task that keeps only the input rows whose value is `> threshold`.
 */
class greater_than_filter_transform_task : public gqe::qep::optional_transform_task {
 public:
  explicit greater_than_filter_transform_task(int64_t threshold) : _threshold(threshold) {}

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr) const override
  {
    auto const tv = gqe::qep::to_table_view(inputs);
    GQE_EXPECTS(tv.has_value(), "greater_than_filter_transform_task: input is not a table");

    cudf::numeric_scalar<int64_t> const threshold{_threshold, /*is_valid=*/true, stream, mr};
    auto const mask = cudf::binary_operation(tv->column(0),
                                             threshold,
                                             cudf::binary_operator::GREATER,
                                             cudf::data_type{cudf::type_id::BOOL8},
                                             stream,
                                             mr);
    auto filtered   = cudf::apply_boolean_mask(*tv, mask->view(), stream, mr);
    return gqe::qep::state_container_builder().add_state(std::move(*filtered)).build();
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<greater_than_filter_transform_task>(_threshold);
  }

 private:
  int64_t _threshold;
};

/**
 * @brief Minimal stateful transform that maintains a running total of the input rows it has seen.
 *
 * `initialize` seeds a `[row_count{0}]` accumulator from its (empty) materialized input; `next`
 * adds the pipelined partition's row count to the shared accumulator and emits the running total.
 */
class running_row_count_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       rmm::cuda_stream_view,
                                       rmm::device_async_resource_ref) const override
  {
    return gqe::qep::make_row_count_container(0);
  }

  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view inputs,
                                                gqe::qep::state_container_view accumulator,
                                                gqe::context_reference,
                                                rmm::cuda_stream_view,
                                                rmm::device_async_resource_ref) const override
  {
    auto& count = std::get<gqe::qep::state_kind::row_count>(*accumulator.front());
    count.value += gqe::qep::get_row_count(inputs);
    return gqe::qep::make_row_count_container(count.value);
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<running_row_count_stateful_transform_task>();
  }
};

class QepTaskGraphTransformTest : public gqe::test::BaseFixture {
 protected:
  // Pin `max_num_partitions` so splittable-lowering tests don't depend on the default.
  static constexpr cudf::size_type max_num_partitions = 4;

  QepTaskGraphTransformTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()}
  {
    get_query_ctx()->parameters.max_num_partitions = max_num_partitions;
  }

  gqe::context_reference ctx_ref;
};

/**
 * @brief A single-task QEP with one non-splittable iterate task lowers to a task graph
 *        with one root task that, when executed, emits the iterator's chunk.
 */
TEST_F(QepTaskGraphTransformTest, SingleIterateTaskLowers)
{
  auto t   = std::make_unique<single_chunk_iterate_task>();
  auto qep = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(graph->root_tasks.size(), 1u);
  ASSERT_EQ(graph->stage_root_tasks.size(), 1u);
  EXPECT_EQ(graph->stage_root_tasks[0].size(), 1u);

  graph->root_tasks[0]->execute();
  auto result = graph->root_tasks[0]->result();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->num_columns(), 1);
  EXPECT_EQ(result->num_rows(), 3);
}

/**
 * @brief A splittable iterate task lowers to one adapter per split, capped by
 *        `optimization_parameters::max_num_partitions`.
 */
TEST_F(QepTaskGraphTransformTest, SplittableIterateLowersIntoMultipleAdapters)
{
  // Request more splits than the cap so the lowering must clamp to `max_num_partitions`.
  auto t   = std::make_unique<splittable_single_chunk_iterate_task>(max_num_partitions + 1);
  auto qep = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->root_tasks.size(), static_cast<std::size_t>(max_num_partitions));
}

/**
 * @brief When the iterator's `max_splits()` is less than
 *        `optimization_parameters::max_num_partitions`, the lowering uses the iterator's
 *        value (the cap chooses the smaller of the two).
 */
TEST_F(QepTaskGraphTransformTest, SplittableIterateCappedByMaxSplits)
{
  constexpr cudf::size_type iterator_max_splits = max_num_partitions - 1;

  auto t   = std::make_unique<splittable_single_chunk_iterate_task>(iterator_max_splits);
  auto qep = gqe::qep::query_execution_plan_builder().add_task(std::move(t)).build();

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->root_tasks.size(), static_cast<std::size_t>(iterator_max_splits));
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

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
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

  auto const qep   = builder.build();
  auto const graph = gqe::qep_task_graph_transform(ctx_ref, qep);

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{5});

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
  // (stages 2/3), and the star_join sink (stage 4).
  EXPECT_EQ(root_counts, (std::vector<std::size_t>{1, 1, 1, 1, 1}));
  EXPECT_EQ(distinct_root_ids.size(), std::size_t{5});

  // Driving the five stages in order runs the whole graph; star_join emits the running count of its
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
  auto const cap = static_cast<cudf::size_type>(get_query_ctx()->parameters.max_num_partitions);
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
  auto const graph = gqe::qep_task_graph_transform(ctx_ref, qep);

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{3});

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

  EXPECT_EQ(root_counts, (std::vector<std::size_t>{static_cast<std::size_t>(partitions), 1, 1}));
  EXPECT_EQ(distinct_root_ids.size(), static_cast<std::size_t>(partitions) + 2);

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
  auto const graph = gqe::qep_task_graph_transform(ctx_ref, qep);

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
  auto const graph = gqe::qep_task_graph_transform(ctx_ref, qep);

  auto const& stages = graph->stage_root_tasks;
  ASSERT_EQ(stages.size(), std::size_t{3});

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

  // Both folds share stage 0 (accumulates) and stage 1 (finalizes); sink sinks at stage 2.
  EXPECT_EQ(root_counts, (std::vector<std::size_t>{2, 2, 1}));
  EXPECT_EQ(distinct_root_ids.size(), std::size_t{5});

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

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
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
 * @brief A two-task `iterate -> optional_transform` QEP lowers to a single stage with one root —
 *        the transform adapter, with the iterate adapter cascaded in as its dependency — and
 *        executing it filters the iterator's output.
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

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
  ASSERT_NE(graph, nullptr);
  // A pure pipeline stage: the transform stays in the iterate's stage, so there is a single stage
  // whose only root is the transform adapter (the iterate adapter is its cascaded dependency).
  ASSERT_EQ(graph->stage_root_tasks.size(), 1u);
  ASSERT_EQ(graph->root_tasks.size(), 1u);

  graph->root_tasks[0]->execute();

  auto state = graph->root_tasks[0]->qep_state_result();
  ASSERT_TRUE(state.has_value());
  auto tv = gqe::qep::to_table_view(*state);
  ASSERT_TRUE(tv.has_value());
  ASSERT_EQ(tv->num_columns(), 1);
  EXPECT_EQ(tv->num_rows(), 2);  // {1, 2, 3} filtered to {2, 3}.
}

/**
 * @brief A three-task `iterate -> optional_transform -> fold` QEP lowers to two stages — the
 *        transform stays in the iterate/accumulate stage, the fold breaks the pipeline into a
 *        finalize stage — and executing the graph folds over the transform's filtered output.
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

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
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

/**
 * @brief A stateful transform reachable only through a pipelined iterate lowers and executes
 *        end-to-end over the iterator's output.
 *
 * The two-task `iterate -> stateful_transform` QEP lowers to a single stage. The iterate is a data
 * pipeline, not a breaker, so per the QEP contract it is a pipelined input feeding `next`. With no
 * pipeline-breaker predecessor, `initialize` receives an empty materialized input.
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

  auto graph = gqe::qep_task_graph_transform(ctx_ref, qep);
  ASSERT_NE(graph, nullptr);
  // Single stage: the stateful_transform adapter, with the iterate adapter cascaded in as its
  // pipelined dependency. It is also the graph's only root task.
  ASSERT_EQ(graph->stage_root_tasks.size(), 1u);
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
