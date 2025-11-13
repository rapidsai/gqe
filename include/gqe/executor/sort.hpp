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

class sort_task : public task {
 public:
  /**
   * @brief Construct a sort task.
   *
   * The sort task reorders the rows of `input` according to the lexicographic ordering of the key
   * table, which is produced by evaluating `keys` on `input`.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to reorder.
   * @param[in] keys Expressions evaluated on `input` to determine the ordering.
   * @param[in] column_orders Desired order for each column in `keys`. The size of this argument
   * must be the same as the size of `keys`.
   * @param[in] null_precedences Whether a null element is smaller or larger than other elements.
   * The size of this argument must be the same as the size of `keys`.
   */
  sort_task(context_reference ctx_ref,
            int32_t task_id,
            int32_t stage_id,
            std::shared_ptr<task> input,
            std::vector<std::unique_ptr<expression>> keys,
            std::vector<cudf::order> column_orders,
            std::vector<cudf::null_order> null_precedences);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_precedences;
};

}  // namespace gqe
