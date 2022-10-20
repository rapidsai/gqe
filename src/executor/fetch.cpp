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
#include <gqe/executor/fetch.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>

namespace gqe {

fetch_task::fetch_task(int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       cudf::size_type offset,
                       cudf::size_type count)
  : task(task_id, stage_id, {std::move(input)}), _offset(offset), _count(count)
{
}

void fetch_task::execute()
{
  prepare_dependencies();
  auto dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);

  auto input_table = dependent_tasks[0]->result().value();

  auto start_idx = _offset;

  if (std::numeric_limits<cudf::size_type>::max() - _count < _offset) {
    throw std::overflow_error("End row index in fetch relation overflows cudf::size_type.\n");
  }

  auto end_idx = _offset + _count;

  if (start_idx >= input_table.num_rows() || start_idx >= end_idx) {
    update_result_cache(cudf::empty_like(input_table));
    return;
  }

  if (end_idx > input_table.num_rows()) { end_idx = input_table.num_rows(); }

  update_result_cache(
    std::make_unique<cudf::table>(cudf::slice(input_table, {start_idx, end_idx})[0]));
  remove_dependencies();
}

}  // namespace gqe