/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/query_context.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {

class task {
 public:
  /**
   * @brief Status of a task.
   */
  enum class status_type {
    not_started,  ///< Task has not been started on the current GPU.
    in_progress,  ///< Task is currently executing on or migrating to the current GPU.
    finished      ///< Task result is available to the current GPU.
  };

  /**
   * @brief Construct a new task.
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] dependencies Dependent tasks of the new task. These tasks are the children nodes in
   * the task graph.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_tasks[i]`.
   */
  task(query_context* query_context,
       int32_t task_id,
       int32_t stage_id,
       std::vector<std::shared_ptr<task>> dependencies,
       std::vector<std::shared_ptr<task>> subquery_tasks);

  virtual ~task()   = default;
  task(const task&) = delete;
  task& operator=(const task&) = delete;

  /**
   * @brief Execute the task on the local GPU.
   *
   * After executing this function, `result()` will return a valid table view.
   */
  virtual void execute() = 0;

  /**
   * @brief Migrate the task result from a remote GPU.
   *
   * After executing this function, `result()` will return a valid table view.
   */
  void migrate();

  /**
   * @brief Return the task result.
   */
  [[nodiscard]] std::optional<cudf::table_view> result() const noexcept { return _result; }

  /**
   * @brief Return the task ID.
   *
   * @note Task ID is the globally unique identifier of the task.
   */
  [[nodiscard]] int32_t task_id() const noexcept { return _task_id; }

  /**
   * @brief Return the stage ID of this task.
   */
  [[nodiscard]] int32_t stage_id() const noexcept { return _stage_id; }

 protected:
  /**
   * @brief Update the local result cache.
   *
   * @note Use this function when the local GPU gets a copy of the task result, for
   * example, after the local GPU finishes executing the task, or after the local GPU explicitly
   * copies the result from a remote GPU.
   */
  void update_result_cache(std::unique_ptr<cudf::table> new_result);

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
  [[nodiscard]] query_context& get_query_context() const noexcept;

  /**
   * @brief Return the optimization parameters.
   */
  [[nodiscard]] const optimization_parameters& get_optimization_parameters() const noexcept;

  /**
   * @brief Make the results of all dependencies available to the local GPU.
   */
  void prepare_dependencies();

  /**
   * @brief Make the results of all subqueries available to the local GPU.
   */
  void prepare_subqueries();

  /**
   * @brief Remove all dependencies from this task.
   *
   * This function should be called after the current task has a valid result, as the dependencies
   * are no longer needed.
   */
  void remove_dependencies() noexcept { _dependencies.clear(); }

  void remove_subqueries() noexcept { _subqueries.clear(); }

 private:
  void prepare_dependent_tasks(std::vector<std::shared_ptr<task>>& dependent_tasks);

  query_context* _query_context;
  int32_t _task_id;
  int32_t _stage_id;
  std::vector<std::shared_ptr<task>> _dependencies;
  std::vector<std::shared_ptr<task>> _subqueries;
  std::optional<cudf::table_view> _result;
  // This field could hold the result after execution, or the migrated table from a remote GPU. Note
  // that it is possible that this field is empty but `_result` contains the valid view, when
  // directly accessing a remote GPU's mapped memory.
  std::unique_ptr<cudf::table> _result_cache;
  std::atomic<status_type> _status;
};

}  // namespace gqe
