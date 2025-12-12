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
#include <gqe/executor/gen_ident_col.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/table/table_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cassert>
#include <numeric>

namespace gqe {

gen_ident_col_task::gen_ident_col_task(context_reference ctx_ref,
                                       int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<task> input)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {})
{
}

void gen_ident_col_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range gen_ident_col_task_range("gen_ident_col_task");

  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();

  rmm::device_uvector<uint64_t> row_id_vec{static_cast<std::size_t>(input_table.num_rows()),
                                           cudf::get_default_stream()};
  thrust::sequence(thrust::device,
                   row_id_vec.data(),
                   row_id_vec.data() + input_table.num_rows(),
                   static_cast<uint64_t>(task_id()) << 32);

  auto row_id_col = std::make_unique<cudf::column>(std::move(row_id_vec), rmm::device_buffer{}, 0);

  cudf::table initial_table(input_table);
  auto result_cols = initial_table.release();
  result_cols.push_back(std::move(row_id_col));
  auto result = std::make_unique<cudf::table>(std::move(result_cols));

  GQE_LOG_TRACE("Execute gen_ident_col task: task_id={}, stage_id={}, output_size={}.",
                task_id(),
                stage_id(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
