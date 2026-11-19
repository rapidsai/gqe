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

#include <gqe/qep/algorithms.hpp>

#include "../utility.hpp"

#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/query_context.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

/**
 * @brief Stub `qep::stateless_transform_task` for topological-order tests.
 *
 * The QEP test only inspects task identity and graph structure; the stub overrides the
 * pure-virtual `next` and `clone` to satisfy the abstract interface but never executes.
 */
class stub_stateless_transform_task : public gqe::qep::stateless_transform_task {
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
    return std::make_unique<stub_stateless_transform_task>();
  }
};

/**
 * @brief Stub `qep::fold_task` for pipeline-partition tests.
 *
 * Acts as a pipeline breaker (since `task::type() == fold`). The pure-virtual `initialize`,
 * `next`, and `clone` overrides exist only to satisfy the abstract interface — the task is
 * never executed.
 */
class stub_fold_task : public gqe::qep::fold_task {
 public:
  gqe::qep::state_container initialize(gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }
  void next(gqe::qep::state_container_view,
            gqe::qep::state_container_view,
            gqe::context_reference,
            gqe::query_stream const&,
            rmm::device_async_resource_ref) const override
  {
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<stub_fold_task>();
  }
};

/**
 * @brief Stub `qep::stateful_transform_task` for pipeline-partition tests.
 *
 * Streams downstream of a fold (consuming the fold's payload via its init input). The
 * pure-virtual overrides exist only to satisfy the abstract interface — the task is never
 * executed.
 */
class stub_stateful_transform_task : public gqe::qep::stateful_transform_task {
 public:
  gqe::qep::state_container initialize(gqe::qep::state_container_view,
                                       gqe::context_reference,
                                       gqe::query_stream const&,
                                       rmm::device_async_resource_ref) const override
  {
    return {};
  }
  std::optional<gqe::qep::state_container> next(gqe::qep::state_container_view,
                                                gqe::qep::state_container_view,
                                                gqe::context_reference,
                                                gqe::query_stream const&,
                                                rmm::device_async_resource_ref) const override
  {
    return std::nullopt;
  }

  std::unique_ptr<gqe::qep::task> clone() const override
  {
    return std::make_unique<stub_stateful_transform_task>();
  }
};

/**
 * @brief Index of `t` in `order`, or `order.size()` if not present.
 */
std::size_t index_of(std::vector<gqe::qep::task const*> const& order, gqe::qep::task const* t)
{
  auto const it = std::find(order.begin(), order.end(), t);
  return static_cast<std::size_t>(it - order.begin());
}

/**
 * @brief Topologically sort the QEP's tasks via `qep::sort_topologically`.
 *
 * Returns `std::nullopt` if the QEP is cyclic.
 */
std::optional<std::vector<gqe::qep::task const*>> topo(gqe::qep::query_execution_plan const& qep)
{
  auto const tasks = qep.tasks();
  return gqe::qep::sort_topologically<gqe::qep::task const*>(
    std::span<gqe::qep::task const* const>(tasks),
    [&](gqe::qep::task const* t) { return qep.predecessors(t); },
    [&](gqe::qep::task const* t) { return qep.successors(t); });
}

/** @brief One task, no edges: the sole node is the entire ordering. */
TEST(SortTopologicallyTest, SingleTask)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));

  auto const qep    = builder.build();
  auto const result = topo(qep);

  EXPECT_THAT(result, ::testing::Optional(::testing::ElementsAre(a_ptr)));
}

/** @brief Linear chain `A → B → C`: each node must precede its successor. */
TEST(SortTopologicallyTest, LinearChain)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(b_ptr, c_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{3});
  EXPECT_LT(index_of(order, a_ptr), index_of(order, b_ptr));
  EXPECT_LT(index_of(order, b_ptr), index_of(order, c_ptr));
}

/** @brief Fork `A → {B, C}`: `A` precedes both `B` and `C`. */
TEST(SortTopologicallyTest, Fork)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(a_ptr, c_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{3});
  auto const pos_a = index_of(order, a_ptr);
  EXPECT_LT(pos_a, index_of(order, b_ptr));
  EXPECT_LT(pos_a, index_of(order, c_ptr));
}

/** @brief Diamond `A → {B, C} → D`: every node respects the join-style ordering. */
TEST(SortTopologicallyTest, Diamond)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_task(std::move(d));
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(a_ptr, c_ptr);
  builder.add_successor(b_ptr, d_ptr);
  builder.add_successor(c_ptr, d_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{4});
  auto const pos_a = index_of(order, a_ptr);
  auto const pos_b = index_of(order, b_ptr);
  auto const pos_c = index_of(order, c_ptr);
  auto const pos_d = index_of(order, d_ptr);
  EXPECT_LT(pos_a, pos_b);
  EXPECT_LT(pos_a, pos_c);
  EXPECT_LT(pos_b, pos_d);
  EXPECT_LT(pos_c, pos_d);
}

/** @brief Bipartite `{A, B} → {C, D}`: every edge crosses the partition. */
TEST(SortTopologicallyTest, Bipartite)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_task(std::move(d));
  builder.add_successor(a_ptr, c_ptr);
  builder.add_successor(a_ptr, d_ptr);
  builder.add_successor(b_ptr, c_ptr);
  builder.add_successor(b_ptr, d_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{4});
  auto const pos_a = index_of(order, a_ptr);
  auto const pos_b = index_of(order, b_ptr);
  auto const pos_c = index_of(order, c_ptr);
  auto const pos_d = index_of(order, d_ptr);
  EXPECT_LT(pos_a, pos_c);
  EXPECT_LT(pos_a, pos_d);
  EXPECT_LT(pos_b, pos_c);
  EXPECT_LT(pos_b, pos_d);
}

/** @brief Two independent chains `A → B` and `C → D` both appear in the output. */
TEST(SortTopologicallyTest, DisconnectedComponents)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_task(std::move(d));
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(c_ptr, d_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{4});
  EXPECT_LT(index_of(order, a_ptr), index_of(order, b_ptr));
  EXPECT_LT(index_of(order, c_ptr), index_of(order, d_ptr));
}

/**
 * @brief The LIFO ready stack drains each successor chain before backtracking to a
 *        sibling, so sibling chains don't interleave. Shape: `A → {B, C}`, `B → D`. The
 *        deep chain `B → D` must appear as a contiguous block in the output.
 */
TEST(SortTopologicallyTest, LifoOrdering)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_task(std::move(c));
  builder.add_task(std::move(d));
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(a_ptr, c_ptr);
  builder.add_successor(b_ptr, d_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  ASSERT_TRUE(result.has_value());
  auto const& order = *result;

  ASSERT_EQ(order.size(), std::size_t{4});
  // The B → D chain is contiguous (no sibling interleaves between them).
  auto const pos_b = index_of(order, b_ptr);
  auto const pos_d = index_of(order, d_ptr);
  EXPECT_EQ(pos_d, pos_b + 1);
}

/**
 * @brief Shape `R → A → B → A`: `R` is the lone root, while `A` and `B` form a cycle. A cyclic
 *        graph has no topological order, so `sort_topologically` returns `std::nullopt` rather
 *        than a partial ordering.
 */
TEST(SortTopologicallyTest, CycleReturnsNullopt)
{
  auto r      = std::make_unique<stub_stateless_transform_task>();
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto* r_ptr = r.get();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(r));
  builder.add_task(std::move(a));
  builder.add_task(std::move(b));
  builder.add_successor(r_ptr, a_ptr);
  builder.add_successor(a_ptr, b_ptr);
  builder.add_successor(b_ptr, a_ptr);

  auto const qep    = builder.build();
  auto const result = topo(qep);
  EXPECT_EQ(result, std::nullopt);
}

// --------------------------------------------------------------------------------------
// partition_into_pipelines
// --------------------------------------------------------------------------------------

/** @brief One task lands in one pipeline with no neighbours. */
TEST(PartitionIntoPipelinesTest, SingleTask)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a));
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{1});
  auto const pid = p.task_to_pipeline.at(a_ptr);
  EXPECT_THAT(p.pipeline_to_tasks.at(pid), ::testing::ElementsAre(a_ptr));
  EXPECT_THAT(p.predecessors.at(pid), ::testing::IsEmpty());
  EXPECT_THAT(p.successors.at(pid), ::testing::IsEmpty());
}

/** @brief `A → B → C` with no fold: streaming-connected, single pipeline. */
TEST(PartitionIntoPipelinesTest, StreamingChainIsOnePipeline)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(b))
    .add_task(std::move(c))
    .add_successor(a_ptr, b_ptr)
    .add_successor(b_ptr, c_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{1});
  auto const pid = p.task_to_pipeline.at(a_ptr);
  EXPECT_EQ(p.task_to_pipeline.at(b_ptr), pid);
  EXPECT_EQ(p.task_to_pipeline.at(c_ptr), pid);
  EXPECT_THAT(p.pipeline_to_tasks.at(pid), ::testing::UnorderedElementsAre(a_ptr, b_ptr, c_ptr));
  EXPECT_THAT(p.predecessors.at(pid), ::testing::IsEmpty());
  EXPECT_THAT(p.successors.at(pid), ::testing::IsEmpty());
}

/**
 * @brief `R → F → C` with `F` a fold: pipeline `{R, F}` then pipeline `{C}`, joined by one
 *        cross-pipeline edge.
 */
TEST(PartitionIntoPipelinesTest, FoldBreaksTheChain)
{
  auto r      = std::make_unique<stub_stateless_transform_task>();
  auto f      = std::make_unique<stub_fold_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* r_ptr = r.get();
  auto* f_ptr = f.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(r))
    .add_task(std::move(f))
    .add_task(std::move(c))
    .add_successor(r_ptr, f_ptr)
    .add_successor(f_ptr, c_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{2});

  auto const rf_pid = p.task_to_pipeline.at(r_ptr);
  auto const c_pid  = p.task_to_pipeline.at(c_ptr);
  EXPECT_NE(rf_pid, c_pid);
  EXPECT_EQ(p.task_to_pipeline.at(f_ptr), rf_pid);

  EXPECT_THAT(p.successors.at(rf_pid), ::testing::ElementsAre(c_pid));
  EXPECT_THAT(p.successors.at(c_pid), ::testing::IsEmpty());

  EXPECT_THAT(p.predecessors.at(rf_pid), ::testing::IsEmpty());
  EXPECT_THAT(p.predecessors.at(c_pid), ::testing::ElementsAre(rf_pid));
}

/** @brief `R → F1 → F2 → T`: two folds in a row split into three pipelines. */
TEST(PartitionIntoPipelinesTest, BackToBackFolds)
{
  auto r       = std::make_unique<stub_stateless_transform_task>();
  auto f1      = std::make_unique<stub_fold_task>();
  auto f2      = std::make_unique<stub_fold_task>();
  auto t       = std::make_unique<stub_stateless_transform_task>();
  auto* r_ptr  = r.get();
  auto* f1_ptr = f1.get();
  auto* f2_ptr = f2.get();
  auto* t_ptr  = t.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(r))
    .add_task(std::move(f1))
    .add_task(std::move(f2))
    .add_task(std::move(t))
    .add_successor(r_ptr, f1_ptr)
    .add_successor(f1_ptr, f2_ptr)
    .add_successor(f2_ptr, t_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{3});

  EXPECT_EQ(p.task_to_pipeline.at(r_ptr), p.task_to_pipeline.at(f1_ptr));
  EXPECT_NE(p.task_to_pipeline.at(f1_ptr), p.task_to_pipeline.at(f2_ptr));
  EXPECT_NE(p.task_to_pipeline.at(f2_ptr), p.task_to_pipeline.at(t_ptr));
}

/**
 * @brief Equivalent QEPs receive the same canonical pipeline IDs regardless of task insertion
 *        order.
 *
 * The stable-key ordering deliberately differs from the pipeline topology. The upstream pipeline
 * has the largest representative key, the middle pipeline has the smallest, and the downstream
 * pipeline has the middle key. Checking the exact IDs verifies the canonical-key rule in addition
 * to comparing the two partitions.
 */
TEST(PartitionIntoPipelinesTest, CanonicalPipelineIdsIgnoreTaskInsertionOrder)
{
  struct test_plan {
    gqe::qep::query_execution_plan qep;
    gqe::qep::task const* upstream_source;
    gqe::qep::task const* upstream_breaker;
    gqe::qep::task const* middle_breaker;
    gqe::qep::task const* downstream_transform;
    gqe::qep::task_stable_keys stable_keys;
  };

  auto make_plan = [](bool reverse_insertion_order) {
    auto upstream_source     = std::make_unique<stub_stateless_transform_task>();
    auto upstream_breaker    = std::make_unique<stub_fold_task>();
    auto middle_breaker      = std::make_unique<stub_fold_task>();
    auto downstream          = std::make_unique<stub_stateless_transform_task>();
    auto* upstream_source_p  = upstream_source.get();
    auto* upstream_breaker_p = upstream_breaker.get();
    auto* middle_breaker_p   = middle_breaker.get();
    auto* downstream_p       = downstream.get();

    gqe::qep::query_execution_plan_builder builder;
    if (reverse_insertion_order) {
      builder.add_task(std::move(downstream))
        .add_task(std::move(middle_breaker))
        .add_task(std::move(upstream_breaker))
        .add_task(std::move(upstream_source));
    } else {
      builder.add_task(std::move(upstream_source))
        .add_task(std::move(upstream_breaker))
        .add_task(std::move(middle_breaker))
        .add_task(std::move(downstream));
    }
    builder.add_successor(upstream_source_p, upstream_breaker_p)
      .add_successor(upstream_breaker_p, middle_breaker_p)
      .add_successor(middle_breaker_p, downstream_p);

    gqe::qep::task_stable_keys stable_keys{
      {upstream_source_p, {.relation_id = 90, .role = gqe::qep::lowering_role::read}},
      {upstream_breaker_p, {.relation_id = 91, .role = gqe::qep::lowering_role::aggregate_reduce}},
      {middle_breaker_p, {.relation_id = 10, .role = gqe::qep::lowering_role::aggregate_final}},
      {downstream_p, {.relation_id = 50, .role = gqe::qep::lowering_role::filter}}};

    return test_plan{.qep                  = builder.build(),
                     .upstream_source      = upstream_source_p,
                     .upstream_breaker     = upstream_breaker_p,
                     .middle_breaker       = middle_breaker_p,
                     .downstream_transform = downstream_p,
                     .stable_keys          = std::move(stable_keys)};
  };

  auto const forward = make_plan(false);
  auto const reverse = make_plan(true);

  auto expect_chain = [](test_plan const& plan) {
    EXPECT_THAT(plan.qep.successors(plan.upstream_source),
                ::testing::ElementsAre(plan.upstream_breaker));
    EXPECT_THAT(plan.qep.successors(plan.upstream_breaker),
                ::testing::ElementsAre(plan.middle_breaker));
    EXPECT_THAT(plan.qep.successors(plan.middle_breaker),
                ::testing::ElementsAre(plan.downstream_transform));
    EXPECT_THAT(plan.qep.successors(plan.downstream_transform), ::testing::IsEmpty());
  };
  expect_chain(forward);
  expect_chain(reverse);

  EXPECT_EQ(forward.stable_keys.at(forward.upstream_source),
            reverse.stable_keys.at(reverse.upstream_source));
  EXPECT_EQ(forward.stable_keys.at(forward.upstream_breaker),
            reverse.stable_keys.at(reverse.upstream_breaker));
  EXPECT_EQ(forward.stable_keys.at(forward.middle_breaker),
            reverse.stable_keys.at(reverse.middle_breaker));
  EXPECT_EQ(forward.stable_keys.at(forward.downstream_transform),
            reverse.stable_keys.at(reverse.downstream_transform));

  auto const forward_partition =
    gqe::qep::partition_into_pipelines(forward.qep, forward.stable_keys);
  auto const reverse_partition =
    gqe::qep::partition_into_pipelines(reverse.qep, reverse.stable_keys);

  auto expect_canonical_ids = [](test_plan const& plan,
                                 gqe::qep::pipeline_partition const& partition) {
    ASSERT_EQ(partition.num_pipelines, std::size_t{3});
    EXPECT_EQ(partition.task_to_pipeline.at(plan.middle_breaker), gqe::qep::pipeline_id{0});
    EXPECT_EQ(partition.task_to_pipeline.at(plan.downstream_transform), gqe::qep::pipeline_id{1});
    EXPECT_EQ(partition.task_to_pipeline.at(plan.upstream_source), gqe::qep::pipeline_id{2});
    EXPECT_EQ(partition.task_to_pipeline.at(plan.upstream_breaker), gqe::qep::pipeline_id{2});
  };
  expect_canonical_ids(forward, forward_partition);
  expect_canonical_ids(reverse, reverse_partition);

  EXPECT_EQ(forward_partition.task_to_pipeline.at(forward.upstream_source),
            reverse_partition.task_to_pipeline.at(reverse.upstream_source));
  EXPECT_EQ(forward_partition.task_to_pipeline.at(forward.upstream_breaker),
            reverse_partition.task_to_pipeline.at(reverse.upstream_breaker));
  EXPECT_EQ(forward_partition.task_to_pipeline.at(forward.middle_breaker),
            reverse_partition.task_to_pipeline.at(reverse.middle_breaker));
  EXPECT_EQ(forward_partition.task_to_pipeline.at(forward.downstream_transform),
            reverse_partition.task_to_pipeline.at(reverse.downstream_transform));
}

/**
 * @brief Two streaming sources feed a fold: both belong to the fold's pipeline.
 *        `L → F ← R`, `F → C`.
 */
TEST(PartitionIntoPipelinesTest, TwoInputsIntoFold)
{
  auto l      = std::make_unique<stub_stateless_transform_task>();
  auto r      = std::make_unique<stub_stateless_transform_task>();
  auto f      = std::make_unique<stub_fold_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* l_ptr = l.get();
  auto* r_ptr = r.get();
  auto* f_ptr = f.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(l))
    .add_task(std::move(r))
    .add_task(std::move(f))
    .add_task(std::move(c))
    .add_successor(l_ptr, f_ptr)
    .add_successor(r_ptr, f_ptr)
    .add_successor(f_ptr, c_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{2});

  auto const fold_pid = p.task_to_pipeline.at(f_ptr);
  EXPECT_EQ(p.task_to_pipeline.at(l_ptr), fold_pid);
  EXPECT_EQ(p.task_to_pipeline.at(r_ptr), fold_pid);
  EXPECT_NE(p.task_to_pipeline.at(c_ptr), fold_pid);
}

/**
 * @brief Perfect-hash-join shape. Build-side `B → BF` (fold) feeds the stateful_transform via
 *        `BF → S`; probe-side `P → S` (stateful_transform) streams alongside. Expected:
 *        `{B, BF}` feeds `{P, S}` via a single cross-pipeline edge.
 */
TEST(PartitionIntoPipelinesTest, HashJoinPipeline)
{
  auto b       = std::make_unique<stub_stateless_transform_task>();
  auto bf      = std::make_unique<stub_fold_task>();
  auto p_in    = std::make_unique<stub_stateless_transform_task>();
  auto s       = std::make_unique<stub_stateful_transform_task>();
  auto* b_ptr  = b.get();
  auto* bf_ptr = bf.get();
  auto* p_ptr  = p_in.get();
  auto* s_ptr  = s.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(b))
    .add_task(std::move(bf))
    .add_task(std::move(p_in))
    .add_task(std::move(s))
    .add_successor(b_ptr, bf_ptr)
    .add_successor(p_ptr, s_ptr)
    .add_successor(bf_ptr, s_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
  ASSERT_EQ(p.num_pipelines, std::size_t{2});

  auto const build_pid = p.task_to_pipeline.at(bf_ptr);
  auto const probe_pid = p.task_to_pipeline.at(s_ptr);
  EXPECT_NE(build_pid, probe_pid);
  EXPECT_EQ(p.task_to_pipeline.at(b_ptr), build_pid);
  EXPECT_EQ(p.task_to_pipeline.at(p_ptr), probe_pid);

  EXPECT_THAT(p.successors.at(build_pid), ::testing::ElementsAre(probe_pid));
}

/**
 * @brief A streaming-only bypass that reunites a pipeline breaker with its downstream
 *        consumer is an ill-formed shape. `A → F → D` (cross-pipeline) competes with
 *        `A → C → D` (streaming-only), so `D` would need to be in both `F`'s and `C`'s
 *        pipeline simultaneously — no pipelining scheme honors this. `partition_into_pipelines`
 *        rejects it.
 */
TEST(PartitionIntoPipelinesTest, FoldWithBypassingStreamingPathThrows)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto f      = std::make_unique<stub_fold_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* f_ptr = f.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(f))
    .add_task(std::move(c))
    .add_task(std::move(d))
    .add_successor(a_ptr, f_ptr)
    .add_successor(a_ptr, c_ptr)
    .add_successor(c_ptr, d_ptr)
    .add_successor(f_ptr, d_ptr);
  auto const qep = builder.build();

  EXPECT_THROW(
    {
      std::ignore =
        gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
    },
    std::logic_error);
}

/**
 * @brief Bypass that reaches a *transitive* descendant of a fold. `A → F1 → S → F2 → T` is
 *        the breaker chain; a streaming bypass `A → B → T` reunites `T` with `F1`'s
 *        pipeline. The direct successors `S` and `T` of `F1` and `F2` look fine in
 *        isolation, but the resulting pipeline DAG has a cycle. `partition_into_pipelines`
 *        must reject this too.
 */
TEST(PartitionIntoPipelinesTest, ChainedFoldBypassThrows)
{
  auto a       = std::make_unique<stub_stateless_transform_task>();
  auto f1      = std::make_unique<stub_fold_task>();
  auto s       = std::make_unique<stub_stateless_transform_task>();
  auto f2      = std::make_unique<stub_fold_task>();
  auto t       = std::make_unique<stub_stateless_transform_task>();
  auto b       = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr  = a.get();
  auto* f1_ptr = f1.get();
  auto* s_ptr  = s.get();
  auto* f2_ptr = f2.get();
  auto* t_ptr  = t.get();
  auto* b_ptr  = b.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(f1))
    .add_task(std::move(s))
    .add_task(std::move(f2))
    .add_task(std::move(t))
    .add_task(std::move(b))
    .add_successor(a_ptr, f1_ptr)
    .add_successor(f1_ptr, s_ptr)
    .add_successor(s_ptr, f2_ptr)
    .add_successor(f2_ptr, t_ptr)
    .add_successor(a_ptr, b_ptr)
    .add_successor(b_ptr, t_ptr);
  auto const qep = builder.build();

  EXPECT_THROW(
    {
      std::ignore =
        gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));
    },
    std::logic_error);
}

// --------------------------------------------------------------------------------------
// terminals
// --------------------------------------------------------------------------------------

/** @brief Diamond `A → {B, C} → D`: only `D` is a QEP-wide terminal. */
TEST(QepTerminalsTest, WholeQepTerminals)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto d      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();
  auto* d_ptr = d.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(b))
    .add_task(std::move(c))
    .add_task(std::move(d))
    .add_successor(a_ptr, b_ptr)
    .add_successor(a_ptr, c_ptr)
    .add_successor(b_ptr, d_ptr)
    .add_successor(c_ptr, d_ptr);
  auto const qep = builder.build();

  auto const all  = qep.tasks();
  auto const term = gqe::qep::terminals(qep, std::span<gqe::qep::task const* const>(all));
  EXPECT_THAT(term, ::testing::ElementsAre(d_ptr));
}

/**
 * @brief Per-pipeline terminals on `R → F → C` (with `F` a fold). Pipeline `{R, F}` has
 *        `F` as its sole terminal (`F → C` leaves the pipeline); pipeline `{C}` has `C`
 *        as its sole terminal (no outgoing edges).
 */
TEST(QepTerminalsTest, PipelineTerminals)
{
  auto r      = std::make_unique<stub_stateless_transform_task>();
  auto f      = std::make_unique<stub_fold_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* r_ptr = r.get();
  auto* f_ptr = f.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(r))
    .add_task(std::move(f))
    .add_task(std::move(c))
    .add_successor(r_ptr, f_ptr)
    .add_successor(f_ptr, c_ptr);
  auto const qep = builder.build();

  auto const p = gqe::qep::partition_into_pipelines(qep, gqe_test::make_test_task_stable_keys(qep));

  auto const& rf_members = p.pipeline_to_tasks.at(p.task_to_pipeline.at(f_ptr));
  auto const rf_term = gqe::qep::terminals(qep, std::span<gqe::qep::task const* const>(rf_members));
  EXPECT_THAT(rf_term, ::testing::ElementsAre(f_ptr));

  auto const& c_members = p.pipeline_to_tasks.at(p.task_to_pipeline.at(c_ptr));
  auto const c_term = gqe::qep::terminals(qep, std::span<gqe::qep::task const* const>(c_members));
  EXPECT_THAT(c_term, ::testing::ElementsAre(c_ptr));
}

/**
 * @brief A node with at least one successor outside the subgraph is a terminal even if it
 *        also has successors inside. Shape: `A → B`, `A → C`. Subgraph `{A, B}` excludes
 *        `C`, so `A` is a terminal (it has an outside successor) and `B` is a terminal
 *        (no successors at all).
 */
TEST(QepTerminalsTest, MixedSuccessorsIsTerminal)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(b))
    .add_task(std::move(c))
    .add_successor(a_ptr, b_ptr)
    .add_successor(a_ptr, c_ptr);
  auto const qep = builder.build();

  std::vector<gqe::qep::task const*> subgraph{a_ptr, b_ptr};
  auto const term = gqe::qep::terminals(qep, std::span<gqe::qep::task const* const>(subgraph));

  EXPECT_THAT(term, ::testing::UnorderedElementsAre(a_ptr, b_ptr));
}

/**
 * @brief Results follow the order of `subgraph_nodes`, not `qep.tasks()`. Shape:
 *        `A → C`, `B → C`; subgraph `{B, A}` (deliberately reversed from insertion order).
 *        Both `A` and `B` are terminals (their only successor `C` is outside the subgraph),
 *        and the result must list `B` before `A`.
 */
TEST(QepTerminalsTest, PreservesInputOrder)
{
  auto a      = std::make_unique<stub_stateless_transform_task>();
  auto b      = std::make_unique<stub_stateless_transform_task>();
  auto c      = std::make_unique<stub_stateless_transform_task>();
  auto* a_ptr = a.get();
  auto* b_ptr = b.get();
  auto* c_ptr = c.get();

  gqe::qep::query_execution_plan_builder builder;
  builder.add_task(std::move(a))
    .add_task(std::move(b))
    .add_task(std::move(c))
    .add_successor(a_ptr, c_ptr)
    .add_successor(b_ptr, c_ptr);
  auto const qep = builder.build();

  std::vector<gqe::qep::task const*> subgraph{b_ptr, a_ptr};
  auto const term = gqe::qep::terminals(qep, std::span<gqe::qep::task const* const>(subgraph));

  EXPECT_THAT(term, ::testing::ElementsAre(b_ptr, a_ptr));
}
