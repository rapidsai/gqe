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
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace gqe {

class join_task : public task {
 public:
  /**
   * @brief Construct a new join task.
   *
   * In `condition`, the column indices of the right table come after the left table. For example,
   * suppose the left table has 3 columns. The equality join of column 1 from the left table with
   * the column 0 from the right table and column 0 from the left table with column 2 from the right
   * table can be represented by
   * `AND(Equal(ColumnReference 1, ColumnReference 3), Equal(ColumnReference 0, ColumnReference 5))`
   *
   * For each Equal expression ("=") in `condition`, the left child expression is evaluated on the
   * left table, and the right child expression is evaluated on the right table. Trying to reference
   * a right (left) table column in the left (right) child expression is an undefined behavior. In
   * the example above, `ColumnReference 1` and `ColumnReference 0` would be evaluated on the left
   * table, whereas `ColumnReference 3` and `ColumnReference 5` are evaluated on the right table.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] left Left table to be joined.
   * @param[in] right Right table to be joined.
   * @param[in] join_type Type of the join.
   * @param[in] condition A boolean expression to define when a left tuple matches with a right
   * tuple.
   * @param[in] projection_indices Column indices to materialize after the join. The rest of columns
   * are discarded.
   */
  join_task(int32_t task_id,
            int32_t stage_id,
            std::shared_ptr<task> left,
            std::shared_ptr<task> right,
            join_type_type join_type,
            std::unique_ptr<expression> condition,
            std::vector<cudf::size_type> projection_indices);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  join_type_type _join_type;
  std::unique_ptr<expression> _condition;
  std::vector<cudf::size_type> _projection_indices;
};

}  // namespace gqe
