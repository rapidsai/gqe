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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {

class sort_task : public task {
 public:
  /**
   * @brief Construct a sort task.
   *
   * The sort task reorders the rows of `input` according to the lexicographic ordering of the key
   * table, which is produced by evaluating `keys` on `input`.
   *
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to reorder.
   * @param[in] keys Expressions evaluated on `input` to determine the ordering.
   * @param[in] column_orders Desired order for each column in `keys`. The size of this argument
   * must be the same as the size of `keys`.
   * @param[in] null_precedences Whether a null element is smaller or larger than other elements.
   * The size of this argument must be the same as the size of `keys`.
   */
  sort_task(query_context* query_context,
            int32_t task_id,
            int32_t stage_id,
            std::shared_ptr<task> input,
            std::vector<std::unique_ptr<expression>> keys,
            std::vector<cudf::order> column_orders,
            std::vector<cudf::null_order> null_precedences);

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_precedences;
};

}  // namespace gqe
