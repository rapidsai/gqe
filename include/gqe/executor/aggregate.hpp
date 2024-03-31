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
#include <gqe/query_context.hpp>

#include <cudf/aggregation.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe {

/**
 * @brief Partition the rows into groups and combine the values.
 */
class aggregate_task : public task {
 public:
  /**
   * @brief Construct an aggregate task.
   *
   * An aggregate task can either represent a reduction (when `keys` are empty) or a groupby (when
   * `keys` are not empty).
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be aggregated.
   * @param[in] keys Expressions evaluated on `input` to represent groups. Rows with the same keys
   * will be grouped together. Note that this argument can be an empty vector. In that case, all
   * rows belong to the same group (reductions).
   * @param[in] values List of `(op, expr)` pairs such that each `expr` will be evaluated on `input`
   * and then rows of the evaluated result in the same group will be combined together using `op`.
   */
  aggregate_task(
    query_context* query_context,
    int32_t task_id,
    int32_t stage_id,
    std::shared_ptr<task> input,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _values;
};

}  // namespace gqe
