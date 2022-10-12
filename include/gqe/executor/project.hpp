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

#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>

#include <memory>
#include <vector>

namespace gqe {

class project_task : public task {
 public:
  /**
   * @brief Construct a projection task.
   *
   * The projection result contains the same number of columns as the length of
   * `output_expressions`. The column `i` in the projection result is constructed by evaluating
   * `output_expressions[i]` on the `input`.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table on which `output_expressions` are evaluated.
   * @param[in] output_expressions Expressions for the result columns.
   */
  project_task(int32_t task_id,
               int32_t stage_id,
               std::shared_ptr<task> input,
               std::vector<std::unique_ptr<expression>> output_expressions);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _output_expressions;
};

}  // namespace gqe
