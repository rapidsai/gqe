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

#pragma once

#include <gqe/executor/task.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/utility/error.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {

namespace detail {
/**
 * @brief The accumulator one fold's N accumulate adapters cooperatively build.
 */
struct shared_fold_accumulator;
/**
 * @brief The accumulator one stateful transform's N streaming-side adapters share.
 */
struct shared_stateful_transform_accumulator;
}  // namespace detail

/**
 * @brief Trait interface for executor tasks whose unprepared work a successor can cancel.
 *
 * # Purpose
 *
 * A stateful transform that observes the QEP early-exit signal skips its unprepared upstream
 * subtree. The subtree is cancelled rather than dropped. The skipping adapter probes each
 * unprepared same-stage dependency for this interface via `dynamic_cast` and releases it. A
 * cancelled task:
 *
 *  - emits the empty early-exit marker as its result,
 *  - releases its own unprepared same-stage dependencies, and
 *  - runs its exactly-once shared-state bookkeeping. In particular, a stateful transform's
 *    initialized accumulator still receives its `finalize` call, the stream-ordered teardown
 *    point that a destructor cannot provide.
 *
 * Tasks that are never cancellation targets do not implement the interface. For example, the
 * fold adapters are stage roots that always execute.
 *
 * # Concurrency
 *
 * Cancellation needs no status transition. It rests on three rules:
 *
 *  - Every same-stage successor performs exactly one action on a task: prepare it for execution,
 *    or `release` it.
 *  - Releases are counted against the successor count that `register_successor` establishes, and
 *    only the last release cancels. A task with several successors (a diamond shape) is
 *    therefore cancelled only when no successor executes it.
 *  - Graph construction completes before execution starts, so every registration happens before
 *    the first release. A successor registered later could misread the early-exit marker as its
 *    input, or cancel the task a second time.
 */
class cancellable_task_trait {
 public:
  virtual ~cancellable_task_trait() = default;

  /**
   * @brief Register one same-stage successor.
   *
   * Called once per same-stage graph edge when an adapter is constructed, before execution
   * starts (see `# Concurrency`).
   */
  void register_successor() noexcept
  {
    _unreleased_successors.fetch_add(1, std::memory_order_relaxed);
  }

  /**
   * @brief Relinquish this task's output on behalf of one successor.
   *
   * Cancels the task via `cancel` when this is the last registered successor to release it.
   *
   * @throws std::logic_error If called more often than successors were registered, which
   *         indicates a graph that was modified after execution started.
   */
  void release()
  {
    auto const remaining = _unreleased_successors.fetch_sub(1, std::memory_order_acq_rel);
    GQE_EXPECTS(remaining >= 1,
                "cancellable_task_trait::release: more releases than registered successors");
    if (remaining != 1) { return; }
    cancel();
  }

  /**
   * @brief Cancel this task, because no successor needs its output.
   *
   * Invoked by the last `release`, or directly by an adapter that observes the early-exit signal
   * before starting and cancels itself. A no-op when the task already carries a result.
   *
   * Like `task::execute`, this is an executor entry point that runs on the invoking worker
   * thread. Implementations use that worker's cuDF default stream and pin the memory resource
   * once, as the execute paths do.
   */
  virtual void cancel() = 0;

 private:
  std::atomic<int32_t> _unreleased_successors{0};  ///< Same-stage successors that have not
                                                   ///< released their need for this task.
};

/**
 * @brief Static task-graph adapter for `qep::stateless_transform_task`.
 *
 * Owns its own `qep::stateless_transform_task` instance (cloned by the builder) and runs its
 * `next(...)` once, emitting the result. `next` is N-ary over its pipelined inputs, so the adapter
 * horizontally concatenates every predecessor's `qep_state_result()` — in dependency order — into
 * the single `inputs` view it passes to `next`.
 *
 * # Lowering
 *
 * The wrapped task is a pure pipeline stage, so it lowers to one adapter per upstream partition,
 * with no pipeline breaker — the adapters stay in their predecessors' pipeline. The predecessors'
 * outputs are zipped chunk-wise: each adapter consumes chunk `i` of every predecessor, so the
 * predecessors must share the same number of partitions.
 *
 * # Early exit
 *
 * The static task graph runs every adapter exactly once, so the QEP early-exit protocol (see
 * `qep::task`) maps onto the adapters' emitted results. An engaged QEP result is never empty
 * because it carries at least the schema, so an empty emitted container unambiguously marks the
 * early exit:
 *
 *  - The wrapped `next` returns `std::nullopt` -> the adapter emits an empty container.
 *  - A predecessor emitted an empty container (it exited) -> the adapter skips the wrapped `next`
 *    and emits an empty container.
 */
class stateless_transform_adapter_task : public task, public cancellable_task_trait {
 public:
  stateless_transform_adapter_task(context_reference ctx_ref,
                                   int32_t task_id,
                                   int32_t stage_id,
                                   std::vector<std::shared_ptr<task>> dependencies,
                                   std::unique_ptr<qep::stateless_transform_task> qep_task);

  void execute() override;

  /** @copydoc cancellable_task_trait::cancel */
  void cancel() override;

 private:
  std::unique_ptr<qep::stateless_transform_task> _qep_task;
};

/**
 * @brief Static task-graph adapter that folds one chunk of input into a shared accumulator.
 *
 * One instance per chunk. The N instances share an opaque `task_private` state via the existing
 * `qep::shared_state` machinery — no additional shared_ptr layer. Each adapter holds its own
 * `qep::fold_task` clone; the clones are stateless and behave identically. `execute()`:
 *
 *   1. Lazy-initializes the shared accumulator via `std::call_once`.
 *   2. Horizontally concatenates its predecessor chunks into one `next` input and calls
 *      `_qep_task->next(input, accumulator, ...)` (thread-safe per the QEP contract). An empty
 *      concatenated input means every predecessor early-exited. The wrapped `next` is then
 *      skipped because there is nothing to fold.
 *   3. Emits a shallow copy of the shared accumulator as its result (every task graph task emits
 *      a real `state_container`).
 *
 * Downstream consumers depend on the paired `fold_finalize_adapter_task`, never directly on an
 * accumulate adapter.
 */
class fold_accumulate_adapter_task : public task {
 public:
  fold_accumulate_adapter_task(context_reference ctx_ref,
                               int32_t task_id,
                               int32_t stage_id,
                               std::vector<std::shared_ptr<task>> dependencies,
                               qep::shared_state shared_accumulator,
                               std::unique_ptr<qep::fold_task> qep_task);

  void execute() override;

  /**
   * @brief Factory for the shared accumulator handed to the N accumulate adapters of one fold.
   *
   * # Purpose
   *
   * The N accumulate adapters of a single fold share one accumulator. The transform calls this
   * once per fold and passes the result to every `fold_accumulate_adapter_task` constructor; the
   * `shared_state`'s inner shared_ptr provides the shared ownership.
   *
   * @return An opaque accumulator slot ready to hand to the fold's accumulate adapters.
   */
  [[nodiscard]] static qep::shared_state make_shared_accumulator();

 private:
  qep::shared_state _shared_accumulator;
  detail::shared_fold_accumulator* _shared_fold;  ///< Typed view into `_shared_accumulator`'s
                                                  ///< owned state, validated at construction.
                                                  ///< Dangles after `_shared_accumulator.reset()`.
  std::unique_ptr<qep::fold_task> _qep_task;
};

/**
 * @brief Static task-graph adapter that finalizes a fold after all partitions complete.
 *
 * Single instance per fold; its dependencies are the N `fold_accumulate_adapter_task`s. The static
 * task graph guarantees this task runs only after they all complete, so the shared accumulator
 * (reachable via any predecessor's result) is fully populated. `execute()` calls
 * `_qep_task->finalize(std::move(accumulator), ...)` and emits the result.
 */
class fold_finalize_adapter_task : public task {
 public:
  fold_finalize_adapter_task(context_reference ctx_ref,
                             int32_t task_id,
                             int32_t stage_id,
                             std::vector<std::shared_ptr<task>> dependencies,
                             std::unique_ptr<qep::fold_task> qep_task);

  void execute() override;

 private:
  std::unique_ptr<qep::fold_task> _qep_task;
};

/**
 * @brief Static task-graph adapter for `qep::stateful_transform_task`.
 *
 * One instance per pipelined-input partition. The N instances share an opaque `task_private` state
 * via the existing `qep::shared_state` machinery — no additional shared_ptr layer. Each adapter
 * holds its own `qep::stateful_transform_task` clone; the clones are stateless and behave
 * identically.
 *
 * Per the QEP task contract, a task's dependencies split by predecessor kind: materialized inputs
 * (from pipeline breakers) are the arguments to `initialize`, while pipelined inputs (from data
 * pipelines) are the arguments to `next`. The transform performs this split and supplies the
 * dependencies grouped as `[materialized..., pipelined...]`, with `num_materialized_inputs` marking
 * the boundary. `execute()`:
 *
 *   1. Lazy-initialises the shared accumulator from the materialized inputs via `std::call_once`
 *      (an empty container when there are none).
 *   2. Calls `_qep_task->next(pipelined_inputs, accumulator, ...)` (thread-safe per the QEP
 *      contract), updating the shared accumulator.
 *   3. Emits the wrapped task's result, or an empty `state_container` when `next` returns
 *      `std::nullopt`.
 *
 * # Early exit
 *
 * The N sibling adapters of one stateful transform share an `exhausted` flag next to the shared
 * accumulator. It implements the QEP early-exit protocol (see `qep::task`) in the static task
 * graph:
 *
 *  - The wrapped `next` returns `std::nullopt` -> the adapter raises `exhausted` and emits an
 *    empty container.
 *  - `exhausted` is raised before the adapter starts -> the adapter cancels itself. Execution is
 *    pull-driven, so its unprepared upstream subtree never runs.
 *  - `exhausted` is raised while the adapter is in flight -> the wrapped `next` is skipped, or
 *    finishes normally when already past the check, which the contract permits.
 *  - Every pipelined predecessor emitted an empty container (they exited) -> the adapter skips the
 *    wrapped `next` and emits an empty container.
 *
 * The shared accumulator is initialized at most once and, when initialized, finalized exactly
 * once:
 *
 *  - Cancelled adapters participate in the pending-adapters count, so the last arrival finalizes
 *    whether it executed or was cancelled (see `cancellable_task_trait`).
 *  - An accumulator that no adapter initialized holds no state and is not finalized.
 */
class stateful_transform_adapter_task : public task, public cancellable_task_trait {
 public:
  stateful_transform_adapter_task(context_reference ctx_ref,
                                  int32_t task_id,
                                  int32_t stage_id,
                                  std::vector<std::shared_ptr<task>> dependencies,
                                  std::size_t num_materialized_inputs,
                                  qep::shared_state shared_accumulator,
                                  std::unique_ptr<qep::stateful_transform_task> qep_task);

  void execute() override;

  /**
   * @brief Factory for the shared accumulator handed to the N adapters of one stateful transform.
   *
   * # Purpose
   *
   * The N streaming-side adapters of a single stateful transform share one accumulator, built once
   * from the build-side predecessor. The transform calls this once per stateful transform and
   * passes the result to every `stateful_transform_adapter_task` constructor; the `shared_state`'s
   * inner shared_ptr provides the shared ownership.
   *
   * @return An opaque accumulator slot ready to hand to the stateful transform's adapters.
   */
  [[nodiscard]] static qep::shared_state make_shared_accumulator();

  /**
   * @copydoc cancellable_task_trait::cancel
   *
   * Mirrors the pending-adapters tail of `execute`, so a cancelled adapter still decrements the
   * shared count and the last arrival finalizes an initialized accumulator.
   */
  void cancel() override;

 private:
  /**
   * @brief Publish the result and retire this adapter's shared-accumulator hold.
   *
   * The tail shared by `execute` and `cancel`. It drops the upstream dependencies, emits the
   * result (the empty early-exit marker when disengaged), lets the last-arriving adapter finalize
   * an initialized accumulator exactly once, and releases this adapter's references.
   *
   * @param[in] result The wrapped task's output, or `std::nullopt` on an early exit.
   * @param[in] stream_handle the query stream handle
   * @param[in] mr Memory resource.
   */
  void shutdown(std::optional<qep::state_container>&& result,
                gqe::query_stream const& stream_handle,
                rmm::device_async_resource_ref mr);

  std::size_t _num_materialized_inputs;
  qep::shared_state _shared_accumulator;
  detail::shared_stateful_transform_accumulator*
    _shared_transform;  ///< Typed view into `_shared_accumulator`'s owned state, validated at
                        ///< construction. Dangles after `_shared_accumulator.reset()`.
  std::unique_ptr<qep::stateful_transform_task> _qep_task;
};

/**
 * @brief Static task-graph adapter for `qep::iterate_task`.
 *
 * Owns its own `qep::iterate_task` instance (cloned by the builder) and drives the
 * `initialize` → `next*` → `finalize` lifecycle once, emitting the iterator's chunks as a
 * single `qep_state` result. Multi-chunk outputs are concatenated; an empty stream is
 * rejected because the schema would be lost.
 *
 * Chunks may contain owned or borrowed states. A borrowed chunk that views storage owned by an
 * input `shared_state` must retain that source state for as long as the view is used. When an
 * input state is already borrowed, its external owner must still outlive every use of the emitted
 * view.
 *
 * Dependencies are optional: a haystack predecessor's `qep_state_result()` is consumed as
 * the `initialize` input when present, or an empty container is passed otherwise.
 */
class iterate_adapter_task : public task, public cancellable_task_trait {
 public:
  iterate_adapter_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::vector<std::shared_ptr<task>> dependencies,
                       std::unique_ptr<qep::iterate_task> qep_task);

  void execute() override;

  /** @copydoc cancellable_task_trait::cancel */
  void cancel() override;

 private:
  std::unique_ptr<qep::iterate_task> _qep_task;
};

}  // namespace gqe
