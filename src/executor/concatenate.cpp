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

#include <gqe/executor/concatenate.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/concatenate.hpp>

void gqe::concatenate_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range concatenate_task_range("concatenate_task");

  auto dependent_tasks = dependencies();

  std::vector<cudf::table_view> tables_to_concatenate;
  tables_to_concatenate.reserve(dependent_tasks.size());
  for (auto const& dependent_task : dependent_tasks) {
    auto depedent_task_result = dependent_task->result();
    assert(depedent_task_result.has_value());
    tables_to_concatenate.push_back(depedent_task_result.value());
  }

  auto result = cudf::concatenate(tables_to_concatenate);

  GQE_LOG_TRACE("Execute concatenate task: task_id={}, stage_id={}, output_size={}.",
                task_id(),
                stage_id(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}
