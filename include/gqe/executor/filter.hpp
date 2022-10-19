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

#pragma once

#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class filter_task : public task {
 public:
  /**
   * @brief Construct a new filter task.
   *
   * A filter task eliminates rows from the result of `input` based on a boolean filter expression
   * `condition`.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be filtered.
   * @param[in] condition A boolean expression evaluated on `input` to represent the filter
   * condition.
   */
  filter_task(int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<task> input,
              std::unique_ptr<expression> condition);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::unique_ptr<expression> _condition;
};

}  // namespace gqe
