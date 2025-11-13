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

class filter_task : public task {
 public:
  /**
   * @brief Construct a new filter task.
   *
   * A filter task eliminates rows from the result of `input` based on a boolean filter expression
   * `condition`.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be filtered.
   * @param[in] condition A boolean expression evaluated on `input` to represent the filter
   * condition.
   * @param[in] projection_indices Column indices to materialize after the filter.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   */
  filter_task(context_reference ctx_ref,
              int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<task> input,
              std::unique_ptr<expression> condition,
              std::vector<cudf::size_type> projection_indices,
              std::vector<std::shared_ptr<task>> subquery_tasks = {});

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace gqe
