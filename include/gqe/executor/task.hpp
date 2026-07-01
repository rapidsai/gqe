/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <proto/task.pb.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

namespace gqe {

namespace result_kind {

struct owned {
  std::unique_ptr<cudf::table> table;
};

struct borrowed {
  cudf::table_view view;
};

/**
 * @brief A QEP state container result.
 *
 * Used by adapter tasks that wrap a `qep::task`. The container is the task's emitted result, which
 * may consist of any supported state kinds.
 */
struct qep_state {
  qep::state_container container;
};

using type = std::variant<result_kind::owned, result_kind::borrowed, result_kind::qep_state>;

}  // namespace result_kind

// forward declaration for friend function declaration
class task_graph;

class task {
  friend void execute_task_graph_single_process(context_reference, task_graph const*);
  friend void execute_task_graph_multi_process(context_reference, task_graph const*);
  friend class task_migration_service;
  friend class task_migration_client;

 public:
  /**
   * @brief Status of a task.
   */
  using status_type = ::proto::TaskStatus;

  /**
   * @brief Construct a new task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] dependencies Dependent tasks of the new task. These tasks are the children nodes in
   * the task graph.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_tasks[i]`.
   */
  task(context_reference ctx_ref,
       int32_t task_id,
       int32_t stage_id,
       std::vector<std::shared_ptr<task>> dependencies,
       std::vector<std::shared_ptr<task>> subquery_tasks);

  virtual ~task()              = default;
  task(const task&)            = delete;
  task& operator=(const task&) = delete;

  /**
   * @brief Execute the task on the local GPU.
   *
   * After executing this function, `result()` will return a valid table view.
   */
  virtual void execute() = 0;

  /**
   * @brief Return the task result as a table view, if available.
   *
   * If the result is a `result_kind::qep_state`, then it is returned as a `cudf::table_view` by
   * routing through `qep::to_table_view`. Pipelines that emit non-convertible shapes must
   * terminate before reaching a legacy `result()` caller (e.g. by inserting a `gather_task`).
   *
   * @throws std::logic_error If the result is a `qep::state_container` that is not convertible to
   *         a `cudf::table_view` (e.g. holds a mask or task-private slot).
   */
  [[nodiscard]] std::optional<cudf::table_view> result() const;

  /**
   * @brief Indicate whether the result of the task is owned or borrowed.
   * @return True, if result is owned; false, if result is borrowed. std::nullopt if task has not
   * finished.
   *
   * `qep_state` results are reported as not owned, since the container may hold borrowed
   * `cudf_column_view` slots.
   */
  [[nodiscard]] std::optional<bool> is_result_owned() const noexcept;

  /**
   * @brief Return the task result as a shallow-copied QEP state container, if available.
   *
   * `owned` and `borrowed` results are wrapped one `cudf_column_view` slot per column;
   * `qep_state` results return a shallow copy of the underlying container.
   *
   * @return The container, or `std::nullopt` if the task has not finished.
   */
  [[nodiscard]] std::optional<qep::state_container> qep_state_result() const;

  /**
   * @brief Return the task ID.
   *
   * @note Task ID is the globally unique identifier of the task.
   */
  [[nodiscard]] int32_t task_id() const noexcept { return _task_id; }

  /** @brief Return the query ID that owns this task. */
  [[nodiscard]] utility::uuid query_id() const noexcept
  {
    return _ctx_ref._query_context->query_id;
  }

  /**
   * @brief Return the stage ID of this task.
   */
  [[nodiscard]] int32_t stage_id() const noexcept { return _stage_id; }

  /**
   * @brief Returns the set of pipelines where this task should be executed. Pipeline ids are unique
   * within a stage.
   */
  [[nodiscard]] std::unordered_set<int32_t> pipeline_ids() const noexcept { return _pipeline_ids; }

  /**
   * @brief Traverses dependencies and subqueries starting at the current task and assigns supplied
   * pipeline_id. Tasks can belong in more than one pipeline.
   */
  void assign_pipeline(int32_t pipeline_id) noexcept;

  /**
   * @brief Mark this task for broadcast execution on all ranks.
   */
  void set_execute_on_all_ranks(bool execute_on_all_ranks) noexcept
  {
    _execute_on_all_ranks = execute_on_all_ranks;
  }

  /**
   * @brief Return whether this task should be executed on all ranks.
   */
  [[nodiscard]] bool is_execute_on_all_ranks() const noexcept { return _execute_on_all_ranks; }

 protected:
  /**
   * @name Result emitters
   *
   * Every `emit_result` overload host-synchronises the default stream before publishing
   * `_status = finished`. This is **load-bearing** under cuDF's per-thread default streams
   * (PTDS): the producing worker thread and the consuming worker thread use different
   * per-thread default streams, so the GPU work that wrote the result is *not* implicitly
   * ordered before the consumer's GPU reads. Without the host sync, a consumer can:
   *
   *  - read partially-written or stale data (data corruption), or
   *  - hit use-after-free when the producer task is destroyed and its
   *    `rmm::device_buffer::deallocate_async` returns memory to the pool while reads on the
   *    consumer's stream are still queued.
   *
   * Do not remove the `synchronize()` call from any overload unless you have replaced PTDS
   * with explicit cross-thread stream ordering (e.g. CUDA events).
   * @{
   */

  /**
   * @brief Emit an owned result.
   *
   * Sets the result as a new table instance. The task owns the result.
   *
   * @note Use this function when the local GPU gets  copy of the task result, for
   * example, after the local GPU finishes executing the task, or after the local GPU explicitly
   * copies the result from a remote GPU.
   */
  void emit_result(std::unique_ptr<cudf::table> new_result);

  /**
   * @brief Emit a borrowed result.
   *
   * Sets the result as a reference to an existing table. The caller retains
   * ownership of the data.
   */
  void emit_result(cudf::table_view new_result);

  /**
   * @brief Emit a QEP state container result.
   *
   * Sets the result as the emitted state of a wrapped `qep::task`. Lifetime of any borrowed slots
   * inside the container is the caller's responsibility (typically a base table owned by the
   * database catalog).
   */
  void emit_result(qep::state_container&& new_result);
  /** @} */

  /**
   * @brief Set task status to failed.
   */
  void fail();

  /**
   * @brief Return the dependent tasks.
   *
   * The dependent tasks are the children node of the current task in the task graph.
   *
   * @note The returned tasks do not share ownership. This object must be kept alive for the
   * returned tasks to be valid.
   */
  [[nodiscard]] std::vector<task*> dependencies() const noexcept;

  /**
   * @brief Return the subquery tasks
   *
   * The subquery tasks are those indexed by any subquery expression(s) that are part of the current
   * task.
   *
   * @note The returned tasks do not share ownership. This object must be kept alive for the
   * returned tasks to be valid.
   */
  [[nodiscard]] std::vector<task*> subqueries() const noexcept;

  /**
   * @brief Return the query context.
   */
  [[nodiscard]] query_context* get_query_context() const noexcept
  {
    return _ctx_ref._query_context;
  }

  /**
   * @brief Return the context reference.
   */
  [[nodiscard]] context_reference get_context_reference() const noexcept { return _ctx_ref; }

  /**
   * @brief Make the results of all dependencies (including subqueries) available to the local GPU.
   */
  void prepare_dependencies();

  /**
   * @brief Remove all dependencies from this task.
   *
   * This function should be called after the current task has a valid result, as the dependencies
   * are no longer needed.
   */
  void remove_dependencies() noexcept
  {
    cudf::get_default_stream().synchronize();
    _dependencies.clear();
    _subqueries.clear();
  }

 private:
  void prepare_dependent_tasks(std::vector<std::shared_ptr<task>>& dependent_tasks);

  context_reference _ctx_ref;
  int32_t _task_id;
  int32_t _stage_id;
  std::unordered_set<int32_t> _pipeline_ids;
  std::vector<std::shared_ptr<task>> _dependencies;
  std::vector<std::shared_ptr<task>> _subqueries;
  result_kind::type _result;
  std::atomic<status_type> _status;
  bool _execute_on_all_ranks = false;
};

}  // namespace gqe
