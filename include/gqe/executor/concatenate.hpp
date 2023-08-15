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

#include <memory>
#include <vector>

namespace gqe {

/**
 * @brief Combine multiple tables along the row axis.
 *
 * For example, if there are three input tables with 2, 4, 3 rows respectively, the concatenated
 * result will have 2 + 4 + 3 = 9 rows. All input tables should have the same number of columns with
 * the same data types.
 */
class concatenate_task : public task {
 public:
  /**
   * @brief Construct a concatenate task.
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] inputs Input tables to be concatenated.
   */
  concatenate_task(query_context* query_context,
                   int32_t task_id,
                   int32_t stage_id,
                   std::vector<std::shared_ptr<task>> inputs)
    : task(query_context, task_id, stage_id, std::move(inputs), {})
  {
  }

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;
};

}  // namespace gqe
