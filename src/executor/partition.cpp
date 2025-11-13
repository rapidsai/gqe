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

#include <gqe/executor/partition.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/partitioning.hpp>

namespace gqe {

partition_task::partition_task(context_reference ctx_ref,
                               int32_t task_id,
                               int32_t stage_id,
                               std::shared_ptr<task> input,
                               std::vector<std::unique_ptr<expression>> partition_cols,
                               int32_t num_partitions)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}),
    _partition_cols(std::move(partition_cols)),
    _num_partitions(num_partitions)
{
}

void partition_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range partition_task_range("partition_task");

  auto dependent_tasks = dependencies();
  GQE_EXPECTS(dependent_tasks.size() == 1, "Partition task expects exactly one input task");
  auto input_view = *dependent_tasks[0]->result();

  GQE_EXPECTS(!_partition_cols.empty(), "Partition columns cannot be empty");
  std::vector<cudf::size_type> columns_to_hash;
  GQE_LOG_TRACE("Partition columns: ");
  for (const auto& partition_col : _partition_cols) {
    GQE_EXPECTS(partition_col->type() == expression::expression_type::column_reference,
                "Currently, partition column must be a column reference");
    auto column_ref = static_cast<column_reference_expression*>(partition_col.get());
    GQE_LOG_TRACE("Partition column: {}", column_ref->to_string());
    columns_to_hash.push_back(column_ref->column_idx());
  }

  auto [result, partition_offsets] =
    cudf::hash_partition(input_view, columns_to_hash, _num_partitions);
  // get the end of the last partition offset
  partition_offsets.push_back(input_view.num_rows());

  _partition_offsets = std::move(partition_offsets);

  GQE_LOG_TRACE(
    "Execute partition task: task_id={}, stage_id={}, input_rows={}, output_rows={}, "
    "output_cols={}",
    task_id(),
    stage_id(),
    input_view.num_rows(),
    result->num_rows(),
    result->num_columns());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
