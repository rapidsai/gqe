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
#include <optional>
#include <vector>

namespace gqe {

class task {
 public:
  /**
   * @brief Construct a new task.
   *
   * @param[in] dependencies Dependent tasks of the new task. These tasks are the children nodes in
   * the task graph.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   */
  task(std::vector<std::shared_ptr<task>> dependencies, int32_t task_id, int32_t stage_id);

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
   * @brief Make the results of all dependencies available to the local GPU.
   */
  void prepare_dependencies();

 private:
  std::optional<cudf::table_view> _result;
  // This field could hold the result after execution, or the migrated table from a remote GPU. Note
  // that it is possible that this field is empty but `_result` contains the valid view, when
  // directly accessing a remote GPU's mapped memory.
  std::unique_ptr<cudf::table> _result_cache;
  std::vector<std::shared_ptr<task>> _dependencies;
  int32_t _task_id;
  int32_t _stage_id;
};

}  // namespace gqe
