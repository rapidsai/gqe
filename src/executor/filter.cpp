/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

filter_task::filter_task(int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<task> input,
                         std::unique_ptr<expression> condition)
  : task(task_id, stage_id, {std::move(input)}), _condition(std::move(condition))
{
}

void filter_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();

  assert(dependent_tasks.size() == 1);

  std::vector<expression const*> condition_expr{_condition.get()};

  auto values               = dependent_tasks[0]->result().value();
  auto [mask, column_cache] = evaluate_expressions(values, condition_expr);

  update_result_cache(cudf::apply_boolean_mask(values, mask[0]));
}

}  // namespace gqe