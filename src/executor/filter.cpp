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
#include <gqe/executor/filter.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

filter_task::filter_task(query_context* query_context,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<task> input,
                         std::unique_ptr<expression> condition,
                         std::vector<cudf::size_type> projection_indices,
                         std::vector<std::shared_ptr<task>> subquery_tasks)
  : task(query_context, task_id, stage_id, {std::move(input)}, std::move(subquery_tasks)),
    _condition(std::move(condition)),
    _projection_indices(std::move(projection_indices))
{
}

void filter_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range filter_task_range("filter_task");

  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  std::vector<expression const*> condition_expr{_condition.get()};

  auto input_table = dependent_tasks[0]->result().value();

  auto [mask, column_cache] = evaluate_expressions(input_table, condition_expr);

  input_table = input_table.select(_projection_indices);

  auto result = cudf::apply_boolean_mask(input_table, mask[0]);

  GQE_LOG_TRACE(
    "Execute filter task: task_id={}, stage_id={}, input_rows={}, output_rows={}, output_cols={}",
    task_id(),
    stage_id(),
    input_table.num_rows(),
    result->num_rows(),
    result->num_columns());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
