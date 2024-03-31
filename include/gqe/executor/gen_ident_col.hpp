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

#pragma once

#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/query_context.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class gen_ident_col_task : public task {
 public:
  /**
   * @brief Construct a new gen_ident_col task.
   *
   * A gen_ident_col task appends a new column to the end of the input table. Its values are 64-bit
   * integers whose higher 32 bits are the task ID and lower 32 bits are the row index. This
   * provides a unique row/partition identifier for each row. Used for window relation
   * implementation.
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be appended to.
   */
  gen_ident_col_task(query_context* query_context,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> input);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

}  // namespace gqe
