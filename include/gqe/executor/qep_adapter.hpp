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

#include <cstddef>
#include <cstdint>
#include <memory>
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
 * @brief Static task-graph adapter for `qep::optional_transform_task`.
 *
 * Owns its own `qep::optional_transform_task` instance (cloned by the builder) and runs its
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
 * # Empty output
 *
 * `qep::optional_transform_task::next` returns an optional because a partition may produce no
 * output at all — e.g. every row was filtered out and no schema is carried forward. A `nullopt` is
 * a legitimate result, not an error; the adapter forwards it faithfully as an empty
 * `qep::state_container`.
 *
 * A transform that instead wants a *schema-correct* empty result returns an empty, zero-row table
 * (which carries the schema), and that flows through the normal value path.
 */
class optional_transform_adapter_task : public task {
 public:
  optional_transform_adapter_task(context_reference ctx_ref,
                                  int32_t task_id,
                                  int32_t stage_id,
                                  std::vector<std::shared_ptr<task>> dependencies,
                                  std::unique_ptr<qep::optional_transform_task> qep_task);

  void execute() override;

 private:
  std::unique_ptr<qep::optional_transform_task> _qep_task;
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
 *      `_qep_task->next(input, accumulator, ...)` (thread-safe per the QEP contract).
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
 */
class stateful_transform_adapter_task : public task {
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

 private:
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
 * Dependencies are optional: a haystack predecessor's `qep_state_result()` is consumed as
 * the `initialize` input when present, or an empty container is passed otherwise.
 */
class iterate_adapter_task : public task {
 public:
  iterate_adapter_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::vector<std::shared_ptr<task>> dependencies,
                       std::unique_ptr<qep::iterate_task> qep_task);

  void execute() override;

 private:
  std::unique_ptr<qep::iterate_task> _qep_task;
};

}  // namespace gqe
