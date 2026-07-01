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

#include <gqe/executor/qep_adapter.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/executor/task_graph.hpp>
#include <gqe/qep/algorithms.hpp>
#include <gqe/qep/query_execution_plan.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/qep/traits/splittable_iterate_trait.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {

namespace {

/**
 * @brief Mutable state carried across per-task lowering steps.
 *
 * The state collected as we walk the QEP is:
 *
 *   - output tasks per QEP task (for wiring dependencies),
 *   - the running task id,
 *   - each pipeline's depth-layered stage,
 *   - and the stage roots collected while lowering.
 *
 * See `qep_task_graph_transform` for the lowering algorithm and stage-root rules.
 */
struct transform_state {
  context_reference ctx_ref;  ///< Execution context shared by every adapter task.
  qep::pipeline_partition const* partition =
    nullptr;  ///< QEP pipeline partition (borrowed; owned by the driver).
  std::unordered_map<qep::pipeline_id, int32_t>
    pipeline_stage;  ///< Depth-layered stage of each pipeline.
  std::unordered_map<qep::task const*, std::vector<std::shared_ptr<task>>>
    outputs;                    ///< Output adapter tasks per QEP task.
  int32_t current_task_id = 0;  ///< Running, globally-unique adapter task id.
  std::vector<std::shared_ptr<task>>
    stage_roots;  ///< Stage roots, in registration order; grouped by `build_stage_root_tasks`.

  /** @brief The depth-layered stage of the pipeline that owns QEP task `qt`. */
  [[nodiscard]] int32_t stage_of(qep::task const* qt) const
  {
    return pipeline_stage.at(partition->task_to_pipeline.at(qt));
  }

  /** @brief Mark task `t` as a root of its own stage (`t->stage_id()`). */
  void register_root(std::shared_ptr<task> t) { stage_roots.push_back(std::move(t)); }

  /**
   * @brief Lay the registered roots out as the dense, stage-ordered `task_graph::stage_root_tasks`.
   *
   * Entry `s` lists stage `s`'s roots, each assigned a unique-within-stage pipeline id via
   * `assign_pipeline`. The executor walks the outer vector in order, so the index must equal the
   * stage id. Every stage has at least one root, so the layout is dense (no entry is empty).
   */
  [[nodiscard]] std::vector<std::vector<std::weak_ptr<task>>> build_stage_root_tasks() const
  {
    int32_t max_stage = -1;
    for (auto const& root : stage_roots) {
      max_stage = std::max(max_stage, root->stage_id());
    }

    std::vector<std::vector<std::weak_ptr<task>>> stage_root_tasks(
      static_cast<std::size_t>(max_stage + 1));
    for (auto const& root : stage_roots) {
      auto& stage_tasks = stage_root_tasks[static_cast<std::size_t>(root->stage_id())];
      root->assign_pipeline(static_cast<int32_t>(stage_tasks.size()));
      stage_tasks.push_back(root);
    }
    return stage_root_tasks;
  }
};

/**
 * @brief Decide the split count for a splittable iterate task.
 *
 * Clamps `splittable_iterate_trait::max_splits()` by
 * `query_context::parameters::max_num_partitions`.
 */
cudf::size_type decide_iterate_split_count(qep::splittable_iterate_trait const& splittable,
                                           query_context const& query_ctx)
{
  auto const max_supported = splittable.max_splits();
  auto const cap           = static_cast<cudf::size_type>(query_ctx.parameters.max_num_partitions);
  return std::max<cudf::size_type>(1, std::min(max_supported, cap));
}

/**
 * @brief Visitor that lowers one QEP task into its adapter(s) and wires dependencies.
 *
 * Driven once per QEP task in topological order so that every predecessor's adapters are
 * already registered in `_state->outputs` when this visitor lowers a task.
 */
class transform_visitor : public qep::qep_visitor {
 public:
  transform_visitor(qep::query_execution_plan const* qep,
                    transform_state* state,
                    query_context const* query_ctx)
    : _qep(qep), _state(state), _query_ctx(query_ctx)
  {
  }

  void visit(qep::iterate_task const& qep_task) override
  {
    auto const preds = _qep->predecessors(&qep_task);

    // Per-partition adapters share the predecessor adapters as their dependencies. The
    // adapter concatenates each predecessor's qep_state container into a single `initialize`
    // input. The pipeline driver registers each predecessor as an exit of its own pipeline,
    // so no breaker is needed here.
    std::vector<std::shared_ptr<task>> shared_deps;
    for (auto const* pred : preds) {
      auto const& pred_adapters = _state->outputs.at(pred);
      GQE_EXPECTS(pred_adapters.size() == 1,
                  "qep_task_graph_transform: iterate predecessor must produce a single "
                  "adapter");
      shared_deps.push_back(pred_adapters.front());
    }

    auto const stage       = _state->stage_of(&qep_task);
    auto const* splittable = dynamic_cast<qep::splittable_iterate_trait const*>(&qep_task);
    std::vector<std::shared_ptr<task>> adapters;
    if (splittable != nullptr) {
      auto const N = decide_iterate_split_count(*splittable, *_query_ctx);
      adapters.reserve(N);
      for (cudf::size_type i = 0; i < N; ++i) {
        adapters.push_back(std::make_shared<iterate_adapter_task>(
          _state->ctx_ref, _state->current_task_id++, stage, shared_deps, splittable->split(i, N)));
      }
    } else {
      adapters.push_back(std::make_shared<iterate_adapter_task>(
        _state->ctx_ref,
        _state->current_task_id++,
        stage,
        std::move(shared_deps),
        qep::clone_qep_task_as<qep::iterate_task>(qep_task)));
    }
    GQE_LOG_TRACE(
      "qep_task_graph_transform: lowered iterate_task into {} adapter(s) (splittable={})",
      adapters.size(),
      splittable != nullptr);
    _state->outputs[&qep_task] = std::move(adapters);
  }

  void visit(qep::optional_transform_task const& qep_task) override
  {
    auto const preds = _qep->predecessors(&qep_task);
    GQE_EXPECTS(!preds.empty(),
                "qep_task_graph_transform: optional_transform_task must have at least one "
                "predecessor");

    // Each predecessor must satisfy two preconditions:
    //
    //  1. It is pipelined, not a pipeline breaker. `optional_transform_task` has no `initialize`,
    //     so it has no materialized inputs; a breaker's result would have nowhere to go.
    //  2. It has the same number of partitions as the others. The predecessors' outputs are zipped
    //     chunk-wise (chunk i of each goes to one adapter, whose `next` horizontally concatenates
    //     them), so the partition counts must match.
    auto const num_partitions = _state->outputs.at(preds.front()).size();
    for (auto const* pred : preds) {
      GQE_EXPECTS(
        !pred->is_pipeline_breaker(),
        "qep_task_graph_transform: optional_transform_task predecessors must be pipelined "
        "(non-pipeline-breaker) inputs");
      GQE_EXPECTS(_state->outputs.at(pred).size() == num_partitions,
                  "qep_task_graph_transform: optional_transform_task predecessors must share the "
                  "same number of partitions");
    }

    // Each adapter is a pure pipeline stage (not a pipeline breaker), so it stays in its
    // predecessors' stage.
    auto const stage = _state->stage_of(&qep_task);
    std::vector<std::shared_ptr<task>> adapters;
    adapters.reserve(num_partitions);
    for (std::size_t chunk = 0; chunk < num_partitions; ++chunk) {
      // deps = [predecessor_0[chunk], predecessor_1[chunk], ...]
      std::vector<std::shared_ptr<task>> deps;
      deps.reserve(preds.size());
      for (auto const* pred : preds) {
        deps.push_back(_state->outputs.at(pred)[chunk]);
      }
      adapters.push_back(std::make_shared<optional_transform_adapter_task>(
        _state->ctx_ref,
        _state->current_task_id++,
        stage,
        std::move(deps),
        qep::clone_qep_task_as<qep::optional_transform_task>(qep_task)));
    }
    GQE_LOG_TRACE("qep_task_graph_transform: lowered optional_transform_task into {} adapter(s)",
                  adapters.size());
    _state->outputs[&qep_task] = std::move(adapters);
  }

  void visit(qep::fold_task const& qep_task) override
  {
    auto const preds = _qep->predecessors(&qep_task);
    GQE_EXPECTS(!preds.empty(),
                "qep_task_graph_transform: fold_task must have at least one predecessor");

    // Each predecessor must satisfy two preconditions:
    //
    //  1. It is pipelined, not a pipeline breaker. `fold_task::initialize` takes no inputs, so the
    //     fold has no materialized inputs; a breaker's result would have nowhere to go.
    //  2. It has the same number of partitions as the others. The predecessors' outputs are zipped
    //     chunk-wise (chunk i of each goes to one accumulate adapter, whose `next` horizontally
    //     concatenates them), so the partition counts must match.
    auto const num_partitions = _state->outputs.at(preds.front()).size();
    for (auto const* pred : preds) {
      GQE_EXPECTS(!pred->is_pipeline_breaker(),
                  "qep_task_graph_transform: fold_task predecessors must be pipelined "
                  "(non-pipeline-breaker) inputs");
      GQE_EXPECTS(_state->outputs.at(pred).size() == num_partitions,
                  "qep_task_graph_transform: fold_task predecessors must share the same number of "
                  "partitions");
    }

    // The fold is a pipeline breaker: its N accumulate adapters are stage roots in the fold's own
    // stage; the single finalize is a root one stage deeper, after the fan-in barrier.
    auto const accumulate_stage = _state->stage_of(&qep_task);
    auto const finalize_stage   = accumulate_stage + 1;

    // One accumulate adapter per chunk, all sharing one accumulator. They stay in the predecessors'
    // stage: pipeline_id propagation through the dep edges keeps each accumulate adapter and its
    // predecessor chunks in one pipeline.
    auto const shared_accumulator = fold_accumulate_adapter_task::make_shared_accumulator();
    std::vector<std::shared_ptr<task>> accumulate_adapters;
    accumulate_adapters.reserve(num_partitions);
    for (std::size_t chunk = 0; chunk < num_partitions; ++chunk) {
      // deps = [predecessor_0[chunk], predecessor_1[chunk], ...]
      std::vector<std::shared_ptr<task>> deps;
      deps.reserve(preds.size());
      for (auto const* pred : preds) {
        deps.push_back(_state->outputs.at(pred)[chunk]);
      }
      auto accumulate = std::make_shared<fold_accumulate_adapter_task>(
        _state->ctx_ref,
        _state->current_task_id++,
        accumulate_stage,
        std::move(deps),
        shared_accumulator,
        qep::clone_qep_task_as<qep::fold_task>(qep_task));
      _state->register_root(accumulate);
      accumulate_adapters.push_back(std::move(accumulate));
    }

    auto finalize_adapter = std::make_shared<fold_finalize_adapter_task>(
      _state->ctx_ref,
      _state->current_task_id++,
      finalize_stage,
      accumulate_adapters,
      qep::clone_qep_task_as<qep::fold_task>(qep_task));
    _state->register_root(finalize_adapter);
    GQE_LOG_TRACE(
      "qep_task_graph_transform: lowered fold_task into {} accumulate adapter(s) + 1 "
      "finalize adapter",
      accumulate_adapters.size());
    _state->outputs[&qep_task] = {finalize_adapter};
  }

  void visit(qep::stateful_transform_task const& qep_task) override
  {
    auto preds = _qep->predecessors(&qep_task);

    // Per the QEP task contract, partition predecessors by pipeline-breaker-ness: materialized
    // inputs (breakers) are the arguments to `initialize`; pipelined inputs (data pipelines) are
    // the arguments to `next`. `stable_partition` groups them as `[materialized..., pipelined...]`
    // while preserving each group's positional order.
    auto const pipelined_begin =
      std::stable_partition(preds.begin(), preds.end(), [](qep::task const* pred) {
        return pred->is_pipeline_breaker();
      });
    auto const num_materialized =
      static_cast<std::size_t>(std::distance(preds.begin(), pipelined_begin));

    // Each materialized predecessor lowers to a single adapter (e.g. a fold finalize); they are
    // shared by every per-partition adapter.
    std::vector<std::shared_ptr<task>> materialized_adapters;
    materialized_adapters.reserve(num_materialized);
    for (auto it = preds.begin(); it != pipelined_begin; ++it) {
      auto const& pred_adapters = _state->outputs.at(*it);
      GQE_EXPECTS(pred_adapters.size() == 1,
                  "qep_task_graph_transform: stateful_transform materialized predecessor must "
                  "produce a single adapter");
      materialized_adapters.push_back(pred_adapters.front());
    }

    // The pipelined predecessors (the streaming side) fan out into partition adapters. Their
    // outputs are zipped chunk-wise: chunk i of each predecessor goes to one stateful_transform
    // adapter, whose `next` horizontally concatenates them (the contract's pipelined-argument
    // partition). One adapter is created per chunk, all sharing one accumulator; the pipelined
    // predecessors must therefore share the same number of partitions.
    GQE_EXPECTS(
      pipelined_begin != preds.end(),
      "qep_task_graph_transform: stateful_transform_task must have at least one pipelined "
      "predecessor");
    auto const num_partitions = _state->outputs.at(*pipelined_begin).size();
    for (auto it = pipelined_begin; it != preds.end(); ++it) {
      GQE_EXPECTS(_state->outputs.at(*it).size() == num_partitions,
                  "qep_task_graph_transform: stateful_transform pipelined predecessors must share "
                  "the same number of partitions");
    }

    auto const stage              = _state->stage_of(&qep_task);
    auto const shared_accumulator = stateful_transform_adapter_task::make_shared_accumulator();
    std::vector<std::shared_ptr<task>> adapters;
    adapters.reserve(num_partitions);
    for (std::size_t chunk = 0; chunk < num_partitions; ++chunk) {
      // deps = [materialized..., pipelined_0[chunk], pipelined_1[chunk], ...]
      std::vector<std::shared_ptr<task>> deps = materialized_adapters;
      for (auto it = pipelined_begin; it != preds.end(); ++it) {
        deps.push_back(_state->outputs.at(*it)[chunk]);
      }
      adapters.push_back(std::make_shared<stateful_transform_adapter_task>(
        _state->ctx_ref,
        _state->current_task_id++,
        stage,
        std::move(deps),
        num_materialized,
        shared_accumulator,
        qep::clone_qep_task_as<qep::stateful_transform_task>(qep_task)));
    }
    GQE_LOG_TRACE(
      "qep_task_graph_transform: lowered stateful_transform_task into {} adapter(s) "
      "({} materialized input(s))",
      adapters.size(),
      num_materialized);
    _state->outputs[&qep_task] = std::move(adapters);
  }

 private:
  qep::query_execution_plan const* _qep;
  transform_state* _state;
  query_context const* _query_ctx;
};

/**
 * @brief ASAP upsweep: compute the earliest stage each pipeline can occupy.
 *
 * Walks the pipelines in topological order so every predecessor is final before it is read. A
 * pipeline's stage is the longest weighted path from a source; each cross-pipeline (breaker) edge
 * advances the stage by two (one boundary for the fan-in, one for the fan-out).
 *
 * @param[in] partition The QEP's pipeline partition, providing each pipeline's predecessors.
 * @param[in] pipeline_topo The pipelines in topological order.
 *
 * @return The earliest stage of each pipeline.
 */
[[nodiscard]] std::unordered_map<qep::pipeline_id, int32_t> stage_upsweep(
  qep::pipeline_partition const& partition, std::span<qep::pipeline_id const> pipeline_topo)
{
  std::unordered_map<qep::pipeline_id, int32_t> asap;
  asap.reserve(partition.num_pipelines);
  for (auto pid : pipeline_topo) {
    int32_t stage = 0;
    for (auto pred : partition.predecessors.at(pid)) {
      stage = std::max(stage, asap.at(pred) + 2);
    }
    asap[pid] = stage;
  }
  return asap;
}

/**
 * @brief ALAP downsweep: push each pipeline to the latest stage its successors allow.
 *
 * Walks the topological order in reverse so every successor is final before it is read. A terminal
 * pipeline (no successors) is anchored at its ASAP stage; otherwise its stage is two below the
 * earliest of its successors.
 *
 * @param[in] partition The QEP's pipeline partition, providing each pipeline's successors.
 * @param[in] pipeline_topo The pipelines in topological order.
 * @param[in] asap The earliest stage of each pipeline, from `stage_upsweep`.
 *
 * @return The final depth-layered stage of each pipeline.
 */
[[nodiscard]] std::unordered_map<qep::pipeline_id, int32_t> stage_downsweep(
  qep::pipeline_partition const& partition,
  std::span<qep::pipeline_id const> pipeline_topo,
  std::unordered_map<qep::pipeline_id, int32_t> const& asap)
{
  std::unordered_map<qep::pipeline_id, int32_t> pipeline_stage;
  pipeline_stage.reserve(partition.num_pipelines);
  for (auto pid : std::views::reverse(pipeline_topo)) {
    auto const& succs = partition.successors.at(pid);
    if (succs.empty()) {
      pipeline_stage[pid] = asap.at(pid);
    } else {
      int32_t consumer_stage = pipeline_stage.at(succs.front());
      for (auto succ : succs) {
        consumer_stage = std::min(consumer_stage, pipeline_stage.at(succ));
      }
      pipeline_stage[pid] = consumer_stage - 2;
    }
  }
  return pipeline_stage;
}

/**
 * @brief Collect the task graph's top-level root tasks and register the terminal sink roots.
 *
 * The QEP's terminal tasks supply the task graph's top-level root tasks; their output tasks anchor
 * the entire graph via shared_ptr deps. A non-breaker terminal is itself a sink stage root; a
 * breaker's output was already registered as a root when the breaker was lowered.
 *
 * @param[in] qep The query execution plan being lowered.
 * @param[in] all_tasks All tasks of the QEP, from which the terminals are computed.
 * @param[in,out] state The lowering state; each terminal sink root is registered into it.
 *
 * @return The task graph's top-level root tasks.
 */
[[nodiscard]] std::vector<std::shared_ptr<task>> collect_root_tasks(
  qep::query_execution_plan const& qep,
  std::span<qep::task const* const> all_tasks,
  transform_state& state)
{
  std::vector<std::shared_ptr<task>> root_tasks;
  for (auto* qep_task : qep::terminals(qep, all_tasks)) {
    for (auto const& t : state.outputs.at(qep_task)) {
      root_tasks.push_back(t);
      // A non-breaker terminal is itself a sink root; a breaker's output was already registered.
      if (!qep_task->is_pipeline_breaker()) { state.register_root(t); }
    }
  }
  return root_tasks;
}

}  // namespace

std::unique_ptr<task_graph> qep_task_graph_transform(context_reference ctx_ref,
                                                     qep::query_execution_plan const& qep)
{
  // Phase 1: partition the QEP into pipelines (connected components over pipelined edges).
  auto const partition = qep::partition_into_pipelines(qep);

  // Phase 2: topologically sort the QEP tasks.
  auto const all_tasks   = qep.tasks();
  auto const global_topo = qep::sort_topologically<qep::task const*>(
    std::span<qep::task const* const>(all_tasks),
    [&](qep::task const* t) { return qep.predecessors(t); },
    [&](qep::task const* t) { return qep.successors(t); });
  GQE_EXPECTS(global_topo.has_value(), "qep_task_graph_transform: QEP contains a cycle");

  // Phase 3: topologically sort the pipelines.
  auto pipeline_id_view = std::views::iota(std::size_t(0), partition.num_pipelines) |
                          std::views::transform([](std::size_t i) { return qep::pipeline_id(i); });
  std::vector<qep::pipeline_id> all_pipelines(pipeline_id_view.begin(), pipeline_id_view.end());
  auto const pipeline_topo = qep::sort_topologically<qep::pipeline_id>(
    std::span<qep::pipeline_id const>(all_pipelines),
    [&](qep::pipeline_id p) -> auto const& { return partition.predecessors.at(p); },
    [&](qep::pipeline_id p) -> auto const& { return partition.successors.at(p); });
  GQE_EXPECTS(pipeline_topo.has_value(),
              "qep_task_graph_transform: pipeline graph contains a cycle");

  // Phase 4: assign each pipeline to a stage via depth-layered staging — a forward ASAP upsweep
  // followed by a reverse ALAP downsweep.
  auto const asap     = stage_upsweep(partition, *pipeline_topo);
  auto pipeline_stage = stage_downsweep(partition, *pipeline_topo, asap);

  transform_state state{
    .ctx_ref = ctx_ref, .partition = &partition, .pipeline_stage = std::move(pipeline_stage)};
  transform_visitor visitor{&qep, &state, ctx_ref._query_context};

  // Phase 5: lower each QEP task in topological order, so every predecessor's adapters are already
  // in `state.outputs` when its consumer is lowered. A breaker's visitor also registers its fan-in
  // and output tasks as roots.
  for (auto* current : *global_topo) {
    current->accept(visitor);
  }

  // Phase 6: register the terminal sink roots and collect the task graph's top-level root tasks.
  auto root_tasks = collect_root_tasks(qep, std::span<qep::task const* const>(all_tasks), state);
  GQE_EXPECTS(!root_tasks.empty(), "qep_task_graph_transform: QEP has no terminal tasks");

  return std::make_unique<task_graph>(
    task_graph{std::move(root_tasks), state.build_stage_root_tasks(), nullptr});
}

}  // namespace gqe
