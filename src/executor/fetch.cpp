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

#include <gqe/context_reference.hpp>
#include <gqe/executor/eval.hpp>
#include <gqe/executor/fetch.hpp>
#include <gqe/utility/cuda.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

fetch_task::fetch_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       cudf::size_type offset,
                       cudf::size_type count)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}), _offset(offset), _count(count)
{
}

void fetch_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range fetch_task_range("fetch_task");

  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();

  auto start_idx = _offset;

  if (std::numeric_limits<cudf::size_type>::max() - _count < _offset) {
    throw std::overflow_error("End row index in fetch relation overflows cudf::size_type.\n");
  }

  auto end_idx = _offset + _count;
  if (end_idx > input_table.num_rows()) { end_idx = input_table.num_rows(); }

  std::unique_ptr<cudf::table> result;
  if (start_idx >= end_idx) {
    result = cudf::empty_like(input_table);
  } else {
    result = std::make_unique<cudf::table>(cudf::slice(input_table, {start_idx, end_idx})[0]);
  }

  GQE_LOG_TRACE("Execute fetch task: task_id={}, stage_id={}, output_size={}.",
                task_id(),
                stage_id(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
