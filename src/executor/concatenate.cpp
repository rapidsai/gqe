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
