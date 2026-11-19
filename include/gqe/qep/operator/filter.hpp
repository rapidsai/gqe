/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/executor/eval.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/qep/task.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Evaluate a Boolean predicate and attach it to the input as a valid-row mask.
 *
 * The task accepts every column-bearing `logical_table_view` shape. Only unique predicate operands
 * are materialized for evaluation. The output is always masked and passes every original logical
 * data-column state through without copying or changing direct/indirect representation. Constant
 * predicates must be eliminated before constructing the task.
 */
class filter_task : public stateless_transform_task {
 public:
  /**
   * @brief Construct a filter task.
   *
   * @param[in] condition Boolean predicate used to produce the valid-row mask.
   *
   * @throws std::invalid_argument If `condition` is null or references no logical column.
   */
  explicit filter_task(std::unique_ptr<gqe::expression> condition);

  /**
   * @copydoc stateless_transform_task::next
   */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::unique_ptr<gqe::expression> _condition;
  std::vector<cudf::size_type> _condition_column_indices;
  expression_binding _expression_binding;
};

}  // namespace qep
}  // namespace gqe
