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

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>

namespace gqe {
namespace qep {

class qep_visitor;

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
 * The executor is responsible for stream synchronization. Tasks do not synchronize the stream(s).
 * Both the exector and tasks coordinate by following contract.
 *
 * In general, the executor is responsible for ensuring that producers are stream-ordered before the
 * consumers of their outputs.
 *
 * `next` may be invoked concurrently on multiple different streams. Each call's
 * device-side work and any state it produces (e.g., `cudf::column`s stored in the accumulator
 * or iterator state) are stream-ordered on the call's `stream`.
 *
 * `initialize` is invoked once to produce state used by `next` on different streams. The executor
 * must guarantee stream order of `initialize` before all `next` tasks.
 *
 * `finalize` is invoked once to to consume state of `next` on different streams. The executor must
 * guarantee stream order of all `next` tasks before `finalize`.
 */
class task {
 public:
  virtual ~task() = default;

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
 * The `optional_transform_task` is stateless. It takes an input stream and produces an output
 * stream.
 *
 * For a given call to `next`, the output is optional because inputs may be filtered out.
 *
 * # Use Case Examples
 *
 * - Filter.
 * - Project.
 */
class optional_transform_task : public task {
 public:
  void accept(qep_visitor& visitor) const override;

  /**
   * @brief Process the next input in the stream.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Optional output state.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;
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
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Initial accumulator state.
   */
  [[nodiscard]] virtual state_container initialize(
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Process the next input in the stream and update the accumulator state.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in,out] accumulator Reference to the accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   */
  virtual void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the accumulator state.
   *
   * The returned state is the task's output.
   *
   * @param[in] accumulator Moved accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return The task's output state.
   */
  [[nodiscard]] virtual state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;
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
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Initial iterator state.
   */
  [[nodiscard]] virtual state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Retrieve the next output in the stream.
   *
   * Returns `std::nullopt` to signal end-of-stream. The executor must stop
   * calling `next` after the iterator returns `std::nullopt`.
   *
   * @param[in] iterator Reference to the iterator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Optional output state. `std::nullopt` indicates end-of-stream.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view iterator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the iterator state.
   *
   * The iterator state is destroyed when this function returns.
   *
   * @param[in] iterator Moved iterator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   */
  virtual void finalize(
    state_container&& iterator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;
};

/**
 * @brief Scan task.
 *
 * # Semantics
 *
 * The `stateful_transform_task` maintains an accumulator state. It takes an input stream and
 * returns an output stream.
 *
 * For a given call to `next`, the output is optional because inputs may be filtered out.
 *
 * # Use Case Examples
 *
 * - Hash join probe phase.
 * - Bloom filter probe phase.
 * - HyperLogLog cardinality estimation.
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
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Initial accumulator state.
   */
  [[nodiscard]] virtual state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Process the next input in the stream and update the accumulator state.
   *
   * @param[in] inputs Reference to the inputs from dependent tasks.
   * @param[in,out] accumulator Reference to the accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   * @return Optional output state.
   */
  [[nodiscard]] virtual std::optional<state_container> next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const = 0;

  /**
   * @brief Finalize the accumulator state.
   *
   * The accumulator state is destroyed when this function returns.
   *
   * @param[in] accumulator Moved accumulator state.
   * @param[in] ctx_ref Context reference.
   * @param[in] stream CUDA stream.
   * @param[in] mr Memory resource.
   */
  virtual void finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;
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
