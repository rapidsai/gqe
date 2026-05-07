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
#include <gqe/executor/write.hpp>
#include <gqe/optimizer/statistics.hpp>

#include <memory>

namespace gqe {

namespace storage {

/** @brief Data access method to write data to a table.
 *
 * The writeable view provides a uniform interface to write data to a table.
 * Subtypes implement the write method for a concrete table kind.
 */
class writeable_view {
 public:
  /**
   * @brief Parameters forwarded to exactly one write task constructor.
   */
  struct task_parameters {
    int32_t task_id;                  /**< Globally unique identifier of the task. */
    std::shared_ptr<gqe::task> input; /** The input table to be written. */
  };

  virtual ~writeable_view() = default;

  /**
   * @brief Return multiple write tasks for the table kind.
   *
   * @param[in] task_parameters The parameters per write task.
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] stage_id Stage of the current task.
   * @param[in] column_names Columns to be stored.
   * @param[in] data_types Expected data types of each column. If the actual data type of a stored
   * column is different from expected, a `std::invalid_argument` exception will
   * be thrown at runtime. Must have the same length as `column_names`.
   *  @param[in] statistics Statistics manager of the table
   *
   * == Thread Safety ==
   *
   * Implementations guarantee thread safety while writes to the table occur.
   */
  virtual std::vector<std::unique_ptr<write_task_base>> get_write_tasks(
    std::vector<task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    table_statistics_manager* statistics) = 0;
};

};  // namespace storage

};  // namespace gqe
