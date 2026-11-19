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

#include <gqe/context_reference.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>

namespace gqe {
namespace qep {

class qep_visitor;

/**
 * @brief Intrinsic effect of a task on its relational result's row sequence.
 *
 * This policy describes task semantics. Normalization derives the result's ordering guarantee
 * from the dependency graph and keeps preserving paths single-chunk.
 */
enum class task_ordering {
  unordered,  ///< No row-sequence guarantee is supplied.
  establish,  ///< The finalized result establishes a row sequence, including an empty result.
  preserve,   ///< A single relational input's relative row order is retained.
};

/**
 * @brief Query Execution Plan (QEP) task.
 *
 * This is a base class for all QEP tasks. A task represents execution logic. It doesn't own any
 * state or manage dependencies.
 *
 * The task's `next` function can be called multiple times to process multiple inputs.
 *
 * # Design
 *
 * ## Task Semantics, Signatures, and Names
 *
 * Tasks are pure functors. The intention is that tasks are composable into a QEP, and the executor
 * is responsible for managing state and dependencies. This allows the executor to have full control
 * over state ownership, replication, and lifetime without duplicate task instances in the
 * dependency graph.
 *
 * Tasks follow functional programming design patterns. Specifying a handful of semantics
 * consolidates the API surface area of tasks, and makes it easier to reason about a task's
 * behavior. The patterns also enable externalization of state.
 *
 * The naming and signatures are inspired by [Rust
 * iterators](https://doc.rust-lang.org/std/iter/trait.Iterator.html). You may find similar
 * signatures in Haskell, Scala, and other languages.
 *
 * ## Task Execution
 *
 * An executor implementation only needs to support calling these abstract task APIs. The executor
 * doesn't need to know the concrete task types.
 *
 * `shared_state` objects are passed by reference or moved when possible to avoid unnecessary state
 * reference counting.
 *
 * ## Task Arguments and Results
 *
 * Tasks take input arguments at each of their three functions: `initialize`, `next`, and
 * `finalize`. Of these, `initialize` and `next` may depend on other tasks, whereas `next` and
 * `finalize` may take the task's own state.
 *
 * A task must treat all state reachable through its `inputs` argument as read-only. An executor
 * may share a predecessor's output with multiple successor tasks. A task that needs mutable state
 * must create or retain it in its returned iterator, accumulator, or output container rather than
 * mutate dependency inputs.
 *
 * However, the QEP does not differentiate between dependencies of `initialize` and `next`. This is
 * because it would require introducing either (a) two different dependency types or (b) exposing
 * the two functions as "task connectors". Neither design feels clean, because tasks have different
 * signatures and it would add an extra argument to `query_execution_plan::add_successor`.
 *
 * Instead, the solution presents itself with an insight: Semantically, `initialize` can only take
 * materialized arguments and `next` can only take pipelined arguments. Thus, if the dependency is
 * on a pipeline breaker, the result of the predecessor's `finalize` must be the argument to
 * `initalize`. In constrast, if the dependency is on a data pipeline, then the result of the
 * predecessor's `next` must be the argument to `next`. This generalizes to N-ary tasks by
 * partitioning the materialized and pipelined arguments while maintaining their positional order
 * (i.e., `[P0, M1, P2, M3]` → `[[M1, M3], [P0, P2]]`).
 *
 * In summary, tasks differ between materialized and pipelined dependencies:
 *
 *  - `initialize` can consume materialized arguments.
 *  - `next` can consume pipelined arguments and produce pipelined results.
 *  - `finalize` can produce materialized results.
 *
 * ## Ordered Results
 *
 * An ordered result is a complete relational result with an established row sequence. The
 * guarantee does not depend on row values, rank ownership, or whether the result contains rows.
 * `ordering_policy` describes whether a task establishes or preserves this sequence. Tasks whose
 * semantics have not been verified default to no ordering guarantee.
 *
 * A preserving task inherits the guarantee of one relational predecessor. Multiple relational
 * inputs have no common sequence without an explicit alignment contract, so normalization treats
 * them conservatively. Task-private initialization state is not another relational input.
 *
 * Preservation describes relative row order within an input, not ordering among concurrent
 * `next` calls. Normalization must ensure that an ordered preservation path receives at most one
 * present relational chunk per execution, including at materialization boundaries.
 *
 * The QEP contract does not require chunks to be processed in relational order. An executor need
 * not enqueue all chunks on the same CUDA stream or enqueue them in relational order on any given
 * stream. A CUDA stream preserves submission order, but the contract does not require that order
 * to match the relational sequence.
 *
 * ## Early Exit
 *
 * The early-exit protocol applies only to tasks whose `next` produces a result. Not all do, as a
 * `next` may instead only update the task's state. A result-producing `next` returns
 * `std::optional`, which encodes the protocol:
 *
 *  - An engaged result is the task's output for one `next` input. It always carries the schema.
 *    In particular, an input whose rows are all filtered out produces a schema-correct zero-row
 *    result.
 *  - `std::nullopt` signals an early exit. The task produces no output for this input and for
 *    every future input of the stream, for example because a row quota tracked in the accumulator
 *    is full. This mirrors the end-of-stream semantics of `iterate_task::next`.
 *
 * The executor owns the propagation of an early exit. After observing `std::nullopt`, it stops
 * feeding the task and skips the remainder of the pipeline, including downstream tasks. It never
 * invokes `next` with inputs that are unavailable due to an upstream early exit, so task
 * implementations do not handle or propagate the exit themselves. Concurrent `next` calls that
 * were already in flight are permitted per the thread-safety contract and follow the same
 * protocol.
 *
 * The executor may also skip tasks inside the exited pipeline segment entirely. A skipped task
 * observes no further `next` calls, but state that was initialized is still finalized exactly
 * once. `finalize` exists because a destructor cannot easily be stream-ordered, so teardown must
 * not fall to destruction.
 *
 * # Thread Safety
 *
 * ## Terms
 *
 * **Internal thread-safety**: Access to state within the function. Example: Parallel GPU threads
 * that mutate the state.
 *
 * **External thread-safety**: Access to state from the outside of the function. Example: Multiple
 * executor workers calling the `next` function concurrently on the same state.
 *
 * ## Host Thread Coordination
 *
 * The `next` function must be thread-safe. It can be called concurrently by multiple workers. The
 * task implementation is responsible for ensuring **external and internal thread-safety**.
 *
 * `initialize` and `finalize` functions are not thread-safe. The executor guarantees that these
 * functions are called only once on a given state. However, the executor is permitted to call these
 * functions concurrently on different states. The task implementation is responsible for ensuring
 * **internal thread-safety**.
 *
 * ## CUDA Stream Coordination
 *
 * The executor is responsible for forward stream coordination. Tasks do not host-synchronize the
 * streams passed to their lifecycle functions solely to coordinate input or output storage
 * lifetimes. A task may still synchronize when required by its algorithm. The executor and tasks
 * coordinate through the following contract.
 *
 * The executor ensures that producers are stream-ordered before consumers of their outputs. It
 * also retains each call's input storage until that call's device work is complete, or orders the
 * input's deallocation streams after that work before releasing the storage. A task has the same
 * responsibility for state that it retains or releases internally.
 *
 * A task preserves the deallocation-stream metadata of state it retains and records complete and
 * accurate metadata for device storage it creates. In particular, owning state created on the
 * call's `stream_handle` records that capability rather than the default stream. Recording metadata
 * does not rebind a buffer's deallocation stream.
 *
 * Column-bearing state retains each recorded `query_stream`, keeping an explicit deallocation
 * stream valid until every buffer that records it has been deallocated. Opaque `task_private` and
 * `keyed_accumulator` state may instead rely on the executor keeping their `query_context` alive;
 * such state must not escape the QEP task graph unless it independently retains every stream it
 * requires.
 *
 * `next` may be invoked concurrently on multiple different streams. Each call's
 * device-side work and any state it produces (e.g., `cudf::column`s stored in the accumulator
 * or iterator state) are stream-ordered on the call's `stream_handle`.
 *
 * `initialize` is invoked once to produce state used by `next` on different streams. The executor
 * must guarantee stream order of `initialize` before all `next` tasks.
 *
 * `finalize` is invoked once to consume state of `next` on different streams. The executor must
 * guarantee stream order of all `next` tasks before `finalize`.
 */
class task {
 public:
  virtual ~task() = default;

  /**
   * @brief Return this task's intrinsic row-ordering policy.
   */
  [[nodiscard]] task_ordering ordering_policy() const noexcept { return _ordering; }

  /**
   * @brief Return an independent deep copy of this task.
   *
   * QEP tasks are stateless pure functors; the clone holds the same parameters and behaves
   * identically. Useful when multiple equivalent instances of the same logical task are
   * needed.
   *
   * @return A new task equivalent to `*this`.
   */
  [[nodiscard]] virtual std::unique_ptr<task> clone() const = 0;

  /**
   * @brief Dispatch a visitor to this task's concrete type.
   *
   * @param[in,out] visitor The visitor to dispatch.
   */
  virtual void accept(qep_visitor& visitor) const = 0;

  /**
   * @brief `true` when this task is a pipeline breaker.
   *
   * A pipeline breaker emits its output only after consuming every input — its outgoing
   * edges are cross-pipeline boundaries during `qep::partition_into_pipelines`.
   */
  [[nodiscard]] virtual bool is_pipeline_breaker() const noexcept { return false; }

 protected:
  /** @brief Construct a task with no row-ordering guarantee. */
  task() = default;

  /**
   * @brief Construct a task with its intrinsic row-ordering policy.
   *
   * @param[in] ordering Intrinsic effect on the relational result's row sequence.
   */
  explicit task(task_ordering ordering) : _ordering(ordering) {}

 private:
  task_ordering _ordering = task_ordering::unordered;
};

/**
 * @brief Transparent hash for `std::unique_ptr<task>` that also accepts `task*`.
 *
 * Useful as the hash functor for an unordered associative container storing owned tasks
 * whose lookup interface should accept non-owning `task*` views.
 */
struct task_ptr_hash {
  using is_transparent = void;
  std::size_t operator()(task const* t) const noexcept { return std::hash<task const*>{}(t); }
  std::size_t operator()(std::unique_ptr<task> const& t) const noexcept
  {
    return std::hash<task const*>{}(t.get());
  }
};

/**
 * @brief Transparent equality for `std::unique_ptr<task>` that also accepts `task const*`.
 */
struct task_ptr_equal {
  using is_transparent = void;
  bool operator()(std::unique_ptr<task> const& a, std::unique_ptr<task> const& b) const noexcept
  {
    return a.get() == b.get();
  }
  bool operator()(std::unique_ptr<task> const& a, task const* b) const noexcept
  {
    return a.get() == b;
  }
  bool operator()(task const* a, std::unique_ptr<task> const& b) const noexcept
  {
    return a == b.get();
  }
};

/**
 * @brief Filter and map task.
 *
 * # Semantics
 *
 * The `stateless_transform_task` keeps no state across `next` calls. It takes an input stream and
 * produces an output stream. Each output carries at least the schema, even when no rows remain.
 * The optional return of `next` follows the early-exit protocol documented in `task`, in which
 * `std::nullopt` signals an early exit.
 *
 * # Use Case Examples
 *
 * - Filter.
 * - Project.
 */
class stateless_transform_task : public task {
 public:
  void accept(qep_visitor& visitor) const override;

  /**
   * @brief Process the next input in the stream.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return The schema-bearing output state, or `std::nullopt` to signal an early exit per the
   *         protocol in the `task` class documentation.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

 protected:
  /** @brief Allow derived tasks to select their intrinsic row-ordering policy. */
  using task::task;
};

/**
 * @brief Fold task.
 *
 * # Semantics
 *
 * The `fold_task` is stateful. It takes an input stream and returns its final state as its output.
 *
 * The fold task is a **pipeline breaker**.
 *
 * # Use Case Examples
 *
 * - Write to a table.
 * - Materialize intermediate results.
 * - Hash join build phase.
 * - Bloom filter build phase.
 * - Aggregate.
 */
class fold_task : public task {
 public:
  void accept(qep_visitor& visitor) const override;
  [[nodiscard]] bool is_pipeline_breaker() const noexcept override { return true; }

  /**
   * @brief Initialize the accumulator state.
   *
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return Initial accumulator state.
   */
  [[nodiscard]] virtual state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Process the next input in the stream and update the accumulator state.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in,out] accumulator Reference to the accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   */
  virtual void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the accumulator state.
   *
   * The returned state is the task's output.
   *
   * @param[in] accumulator Moved accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return The task's output state.
   */
  [[nodiscard]] virtual state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 protected:
  /** @brief Allow derived tasks to select their intrinsic row-ordering policy. */
  using task::task;
};

/**
 * @brief Iterate task.
 *
 * # Semantics
 *
 * The `iterate_task` is stateful. It iterates over state and produces an output stream.
 *
 * # Use Case Examples
 *
 * - Read from a table.
 * - Iterate over a state (e.g., a cuDF table, a hash map with aggregation results).
 */
class iterate_task : public task {
 public:
  void accept(qep_visitor& visitor) const override;

  /**
   * @brief Initialize the iterator state.
   *
   * @param[in] inputs Reference to the inputs.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return Initial iterator state.
   */
  [[nodiscard]] virtual state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Retrieve the next output in the stream.
   *
   * Returns `std::nullopt` to signal end-of-stream. The executor must stop
   * calling `next` after the iterator returns `std::nullopt`.
   *
   * @param[in] iterator Reference to the iterator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return Optional output state. `std::nullopt` indicates end-of-stream.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the iterator state.
   *
   * The iterator state is destroyed when this function returns.
   *
   * @param[in] iterator Moved iterator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   */
  virtual void finalize(
    state_container&& iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 protected:
  /** @brief Allow derived tasks to select their intrinsic row-ordering policy. */
  using task::task;
};

/**
 * @brief Scan task.
 *
 * # Semantics
 *
 * The `stateful_transform_task` maintains an accumulator state. It takes an input stream and
 * returns an output stream. The optional return of `next` follows the early-exit protocol
 * documented in `task`.
 *
 * # Use Case Examples
 *
 * - Hash join probe phase.
 * - Bloom filter probe phase.
 * - HyperLogLog cardinality estimation.
 * - Fetch (SQL `LIMIT`/`OFFSET`) row quota.
 */
class stateful_transform_task : public task {
 public:
  void accept(qep_visitor& visitor) const override;

  /**
   * @brief Initialize the accumulator state.
   *
   * @param[in] inputs Reference to the inputs. Operators that have no init-time inputs
   *   (e.g., HyperLogLog) receive an empty container.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return Initial accumulator state.
   */
  [[nodiscard]] virtual state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Process the next input in the stream and update the accumulator state.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in,out] accumulator Reference to the accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   * @return The schema-bearing output state, or `std::nullopt` to signal an early exit per the
   *         protocol in the `task` class documentation.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the accumulator state.
   *
   * The accumulator state is destroyed when this function returns.
   *
   * @param[in] accumulator Moved accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   */
  virtual void finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 protected:
  /** @brief Allow derived tasks to select their intrinsic row-ordering policy. */
  using task::task;
};

/**
 * @brief Clone a `qep::task` and downcast to a concrete `qep::task`-derived type.
 *
 * Convenience wrapper around `qep::task::clone()` that returns the desired derived type. The
 * caller guarantees the dynamic type of `source` is a `T`.
 *
 * @tparam T A type derived from `qep::task`.
 *
 * @param[in] source The task to clone.
 *
 * @return An owning pointer to the cloned task as `T`.
 */
template <typename T>
[[nodiscard]] std::unique_ptr<T> clone_qep_task_as(task const& source)
{
  // `static_cast<T*>(task*)` already requires T to derive from `task` for compilation;
  // this assert just surfaces the constraint up front with a clearer error.
  static_assert(std::is_base_of_v<task, T>,
                "clone_qep_task_as requires T to be derived from qep::task");
  std::unique_ptr<task> base = source.clone();
  assert(dynamic_cast<T*>(base.get()) != nullptr);
  return std::unique_ptr<T>(static_cast<T*>(base.release()));
}

}  // namespace qep
}  // namespace gqe
