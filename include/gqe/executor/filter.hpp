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
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be filtered.
   * @param[in] condition A boolean expression evaluated on `input` to represent the filter
   * condition.
   * @param[in] projection_indices Column indices to materialize after the filter.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   */
  filter_task(query_context* query_context,
              int32_t task_id,
              int32_t stage_id,
              std::shared_ptr<task> input,
              std::unique_ptr<expression> condition,
              std::vector<cudf::size_type> projection_indices,
              std::vector<std::shared_ptr<task>> subquery_tasks = {});

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace gqe
