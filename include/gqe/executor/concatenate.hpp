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

#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Combine multiple tables along the row axis.
 *
 * For example, if there are three input tables with 2, 4, 3 rows respectively, the concatenated
 * result will have 2 + 4 + 3 = 9 rows. All input tables should have the same number of columns with
 * the same data types.
 */
class concatenate_task : public task {
 public:
  /**
   * @brief Construct a concatenate task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] inputs Input tables to be concatenated.
   */
  concatenate_task(context_reference ctx_ref,
                   int32_t task_id,
                   int32_t stage_id,
                   std::vector<std::shared_ptr<task>> inputs)
    : task(ctx_ref, task_id, stage_id, std::move(inputs), {})
  {
  }

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

}  // namespace gqe
