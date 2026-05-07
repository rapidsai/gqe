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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class gen_ident_col_task : public task {
 public:
  /**
   * @brief Construct a new gen_ident_col task.
   *
   * A gen_ident_col task appends a new column to the end of the input table. Its values are 64-bit
   * integers whose higher 32 bits are the task ID and lower 32 bits are the row index. This
   * provides a unique row/partition identifier for each row. Used for window relation
   * implementation.
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be appended to.
   */
  gen_ident_col_task(context_reference ctx_ref,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> input);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

}  // namespace gqe
