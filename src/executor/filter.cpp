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

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

filter_task::filter_task(query_context* query_context,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<task> input,
                         std::unique_ptr<expression> condition,
                         std::vector<std::shared_ptr<task>> subquery_tasks)
  : task(query_context, task_id, stage_id, {std::move(input)}, std::move(subquery_tasks)),
    _condition(std::move(condition))
{
}

void filter_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();

  // TODO: Evaluate possible subquery

  assert(dependent_tasks.size() == 1);

  std::vector<expression const*> condition_expr{_condition.get()};

  auto input_table          = dependent_tasks[0]->result().value();
  auto [mask, column_cache] = evaluate_expressions(input_table, condition_expr);

  emit_result(cudf::apply_boolean_mask(input_table, mask[0]));
}

}  // namespace gqe
