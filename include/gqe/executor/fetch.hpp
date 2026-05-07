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

class fetch_task : public task {
 public:
  /**
   * @brief Construct a fetch task.
   *
   * The fetch task retrieves `count` consecutive rows from the input table starting at the row
   * with index `offset`. If `offset` is greater than or equal to the number of rows,
   * the resulting table is empty. If the number of rows after `offset` is less than `count`,
   * all rows from `offset` to the end of the table are retrieved.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task
   * @param[in] stage_id Stage of the current task
   * @param[in] input Input table to fetch from
   * @param[in] offset The row index from which the fetch starts
   * @param[in] count The number of rows to retrieve starting from offset
   */
  fetch_task(context_reference ctx_ref,
             int32_t task_id,
             int32_t stage_id,
             std::shared_ptr<task> input,
             cudf::size_type offset,
             cudf::size_type count);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  cudf::size_type _offset;
  cudf::size_type _count;
};

}  // namespace gqe
