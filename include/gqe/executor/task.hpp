/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace gqe {

class task {
 public:
  enum class result_status_type { available, not_available };

  /**
   * @brief Construct a new task.
   *
   * @param[in] dependencies Dependent tasks of the new task. These tasks are the children nodes in
   * the task graph.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   */
  task(std::vector<std::shared_ptr<task>> const& dependencies, int32_t task_id, int32_t stage_id);

  virtual ~task();
  task(const task&)            = delete;
  task& operator=(const task&) = delete;

  /**
   * @brief Execute the task on the local GPU.
   */
  virtual void execute() = 0;

  /**
   * @brief Migrate the task result from a remote GPU.
   */
  void migrate();

  /**
   * @brief Return whether the task result is available to the local GPU.
   *
   * @return result_status_type::available The task result is available.
   * @return result_status_type::not_available The task result is unavailable.
   */
  result_status_type result_status() const noexcept { return _result_status; }

  /**
   * @brief Return the task result.
   *
   * @note This function will not check whether the result is available to the local GPU. The
   * user can check that by calling `result_status()`. If the result is unavailable, this function
   * has undefined behavior.
   */
  cudf::table_view result() const noexcept { return _result; }

  /**
   * @brief Return the task ID.
   *
   * @note Task ID is the globally unique identifier of the task.
   */
  int32_t task_id() const noexcept { return _task_id; }

  /**
   * @brief Return the stage ID of this task.
   */
  int32_t stage_id() const noexcept { return _stage_id; }

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
   * @note The depedent tasks are the children node of the current task in the task graph.
   */
  std::vector<std::shared_ptr<task>> dependencies() const noexcept { return _dependencies; }

 private:
  result_status_type _result_status;
  cudf::table_view _result;
  // This field could hold the result after execution, or the migrated table from a remote GPU. Note
  // that it is possible that this field is empty but `_result` contains the valid view, when
  // directly accessing a remote GPU's mapped memory.
  std::unique_ptr<cudf::table> _result_cache;
  std::vector<std::shared_ptr<task>> _dependencies;
  int32_t _task_id;
  int32_t _stage_id;
};

}  // namespace gqe
