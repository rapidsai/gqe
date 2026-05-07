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

#include <gqe/executor/sort.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

sort_task::sort_task(context_reference ctx_ref,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> input,
                     std::vector<std::unique_ptr<expression>> keys,
                     std::vector<cudf::order> column_orders,
                     std::vector<cudf::null_order> null_precedences)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}),
    _keys(std::move(keys)),
    _column_orders(std::move(column_orders)),
    _null_precedences(std::move(null_precedences))
{
}

void sort_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range sort_task_range("sort_task");

  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto values               = dependent_tasks[0]->result().value();
  auto [keys, column_cache] = evaluate_expressions(
    get_query_context()->parameters, values, utility::to_const_raw_ptrs(_keys));

  auto result =
    cudf::sort_by_key(values, cudf::table_view(keys), _column_orders, _null_precedences);

  GQE_LOG_TRACE("Execute sort task: task_id={}, stage_id={}, output_size={}.",
                task_id(),
                stage_id(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
