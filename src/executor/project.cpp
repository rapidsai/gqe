/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/executor/project.hpp>

#include <gqe/context_reference.hpp>
#include <gqe/executor/eval.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace gqe {

project_task::project_task(context_reference ctx_ref,
                           int32_t task_id,
                           int32_t stage_id,
                           std::shared_ptr<task> input,
                           std::vector<std::unique_ptr<expression>> output_expressions)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}),
    _output_expressions(std::move(output_expressions))
{
}

void project_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range project_task_range("project_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  auto [eval_columns, column_cache] = evaluate_expressions(
    get_query_context()->parameters, input_table, utility::to_const_raw_ptrs(_output_expressions));
  cudf::table_view eval_table(eval_columns);

  // FIXME: For the current implementation, the result table is copied from `eval_table`, which
  // could be inefficient.
  // In theory, a copy is not necessary in certain scenarios. For example, if a project
  // relation is only used to reorder columns, we could simply let the input task release its
  // ownership of the table, change the order of the columns and assemble into a new table.
  auto result = std::make_unique<cudf::table>(eval_table);

  GQE_LOG_TRACE("Execute project task: task_id={}, stage_id={}, output_size={}.",
                task_id(),
                stage_id(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
