/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/query_context.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class fetch_task : public task {
 public:
  /**
   * @brief Construct a fetch task.
   *
   * The fetch task retrieves `count` consecutive rows from the input table starting at the row
   * with index `offset`. If `offset` is greater than or equal to the number of rows,
   * the resulting table is empty. If the number of rows after `offset` is less than `count`,
   * all rows from `offset` to the end of the table are retrieved.
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task
   * @param[in] stage_id Stage of the current task
   * @param[in] input Input table to fetch from
   * @param[in] offset The row index from which the fetch starts
   * @param[in] count The number of rows to retrieve starting from offset
   */
  fetch_task(query_context* query_context,
             int32_t task_id,
             int32_t stage_id,
             std::shared_ptr<task> input,
             cudf::size_type offset,
             cudf::size_type count);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  cudf::size_type _offset;
  cudf::size_type _count;
};

}  // namespace gqe
