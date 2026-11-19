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

#include <gqe/executor/partition_merge.hpp>

#include <gqe/executor/partition.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

namespace gqe {

partition_merge_task::partition_merge_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::vector<std::shared_ptr<task>> inputs,
                                           int32_t partition_idx)
  : task(ctx_ref, task_id, stage_id, {std::move(inputs)}, {}), _partition_idx(partition_idx)
{
}

void gqe::partition_merge_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range partition_merge_task_range("partition_merge_task");

  GQE_LOG_TRACE("Execute partition merge task: input partition offsets for partition {} :",
                _partition_idx);
  std::vector<cudf::table_view> tables_to_concat;
  for (auto dependent_task : dependencies()) {
    auto input_view = *dependent_task->result();

    auto part_task = dynamic_cast<partition_task*>(dependent_task);
    // make sure the child task is partition_task
    GQE_EXPECTS(part_task != nullptr, "The child task is not a partition_task");
    auto const start_idx = part_task->partition_offset(_partition_idx);
    auto const end_idx   = part_task->partition_offset(_partition_idx + 1);

    GQE_LOG_TRACE("input partition offset for input task id {} : start_idx={}, end_idx={}",
                  dependent_task->task_id(),
                  start_idx,
                  end_idx);
    tables_to_concat.push_back(cudf::slice(input_view, {start_idx, end_idx})[0]);
  }

  auto result = cudf::concatenate(tables_to_concat);

  GQE_LOG_TRACE(
    "Execute partition merge task: task_id={}, stage_id={}, output_rows={}, output_cols={}",
    task_id(),
    stage_id(),
    result->num_rows(),
    result->num_columns());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
