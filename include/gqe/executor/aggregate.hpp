/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/expression/expression.hpp>

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
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table to be aggregated.
   * @param[in] keys Expressions evaluated on `input` to represent groups. Rows with the same keys
   * will be grouped together. Note that this argument can be an empty vector. In that case, all
   * rows belong to the same group (reductions).
   * @param[in] values List of `(op, expr)` pairs such that each `expr` will be evaluated on `input`
   * and then rows of the evaluated result in the same group will be combined together using `op`.
   * @param[in] condition An optional boolean expression evaluated on `input` to filter rows before
   * performing aggregation. Note: That this is currently only supported for groupby and not for
   * pure reductions
   * @param[in] perfect_hashing Whether to use perfect hashing.
   */
  aggregate_task(
    context_reference ctx_ref,
    int32_t task_id,
    int32_t stage_id,
    std::shared_ptr<task> input,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values,
    std::unique_ptr<expression> condition = nullptr,
    bool perfect_hashing                  = false);

  /**
   * @brief Return a boolean indicating whether to use perfect hashing.
   */
  [[nodiscard]] bool perfect_hashing() const noexcept { return _perfect_hashing; }

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _values;
  std::unique_ptr<expression> _condition;
  bool _perfect_hashing;
};

}  // namespace gqe
