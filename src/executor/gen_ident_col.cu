/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/gen_ident_col.hpp>

#include <cudf/table/table_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cassert>
#include <numeric>

namespace gqe {

gen_ident_col_task::gen_ident_col_task(int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<task> input)
  : task(task_id, stage_id, {std::move(input)}, {})
{
}

void gen_ident_col_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();

  rmm::device_uvector<uint64_t> row_id_vec{static_cast<std::size_t>(input_table.num_rows()),
                                           rmm::cuda_stream_default};
  thrust::sequence(thrust::device,
                   row_id_vec.data(),
                   row_id_vec.data() + input_table.num_rows(),
                   static_cast<uint64_t>(task_id()) << 32);

  auto row_id_col = std::make_unique<cudf::column>(std::move(row_id_vec));

  cudf::table initial_table(input_table);
  auto result_cols = initial_table.release();
  result_cols.push_back(std::move(row_id_col));
  update_result_cache(std::make_unique<cudf::table>(std::move(result_cols)));
}

}  // namespace gqe