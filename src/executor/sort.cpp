/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/eval.hpp>
#include <gqe/executor/sort.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

sort_task::sort_task(query_context* query_context,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> input,
                     std::vector<std::unique_ptr<expression>> keys,
                     std::vector<cudf::order> column_orders,
                     std::vector<cudf::null_order> null_precedences)
  : task(query_context, task_id, stage_id, {std::move(input)}, {}),
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
  auto [keys, column_cache] = evaluate_expressions(values, utility::to_const_raw_ptrs(_keys));

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
