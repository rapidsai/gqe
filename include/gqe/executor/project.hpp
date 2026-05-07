/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table on which `output_expressions` are evaluated.
   * @param[in] output_expressions Expressions for the result columns.
   */
  project_task(context_reference ctx_ref,
               int32_t task_id,
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
