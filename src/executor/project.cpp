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
#include <gqe/executor/project.hpp>
#include <gqe/utility.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace gqe {

project_task::project_task(int32_t task_id,
                           int32_t stage_id,
                           std::shared_ptr<task> input,
                           std::vector<std::unique_ptr<expression>> output_expressions)
  : task(task_id, stage_id, {std::move(input)}), _output_expressions(std::move(output_expressions))
{
}

void project_task::execute()
{
  prepare_dependencies();
  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  auto [eval_columns, column_cache] =
    evaluate_expressions(input_table, utility::to_const_raw_ptrs(_output_expressions));
  cudf::table_view eval_table(eval_columns);

  // FIXME: For the current implementation, the result table is copied from `eval_table`, which
  // could be inefficient.
  // In theory, a copy is not necessary in certain scenarios. For example, if a project
  // relation is only used to reorder columns, we could simply let the input task release its
  // ownership of the table, change the order of the columns and assemble into a new table.
  update_result_cache(std::make_unique<cudf::table>(eval_table));
  remove_dependencies();
}

}  // namespace gqe
