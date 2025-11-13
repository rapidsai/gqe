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
#include <gqe/executor/read.hpp>
#include <gqe/expression/expression.hpp>

#include <memory>

namespace gqe {

namespace storage {

/**
 * @brief Data access method to read the data in a table.
 *
 * The readable view provides a uniform interface to read the data in a table.
 * Subtypes implement the read method for the concrete table kind.
 */
class readable_view {
 public:
  /**
   * @brief Parameters forwarded to exactly one read task constructor.
   */
  struct task_parameters {
    int32_t task_id; /**< Globally unique identifier of the task. */
    std::unique_ptr<gqe::expression> partial_filter =
      nullptr; /**< Used to support predicate pushdown. Note that a row that satisfies
                * the predicate is guaranteed to be included in the loaded table, but a row that
                * does not satisfy the predicate may or may not be excluded. If such exclusion needs
                * to be guaranteed, an extra filter task is needed. If this argument is nullptr, no
                * rows will be filtered out.
                */
    std::vector<std::shared_ptr<gqe::task>> subquery_tasks =
      {}; /**< Subquery tasks that may be referenced by a subquery expression. A
           * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
           */
  };

  virtual ~readable_view() = default;

  /**
   * @brief Return multiple read tasks for the table kind.
   *
   * @param[in] task_parameters The parameters per read task.
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] stage_id Stage of the current task.
   * @param[in] column_names Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   *
   * == Thread Safety ==
   *
   * Implementations guarantee thread safety while writes to the table occur. In particular,
   * repeated reads of the table size may be inconsistent while table appends are underway.
   */
  virtual std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) = 0;
};

};  // namespace storage

};  // namespace gqe
