/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/executor/read.hpp>
#include <gqe/storage/pruning.hpp>
#include <gqe/types.hpp>

namespace gqe {
namespace storage {
class row_group_interface;
}  // namespace storage
}  // namespace gqe

namespace gqe {

namespace storage {

/**
 * @brief A read task for loading data from memory.
 */
class in_memory_read_task : public read_task_base {
 public:
  /**
   * @brief Construct an in-memory read task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] row_groups The row groups assigned to this task.
   * @param[in] column_indexes Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_indexes`.
   * @param[in] memory_kind The memory kind used by the input table.
   * @param[in] partial_filter Used to support predicate pushdown. Note that a row that satisfies
   * the predicate is guaranteed to be included in the loaded table, but a row that does not satisfy
   * the predicate may or may not be excluded. If such exclusion needs to be guaranteed, an extra
   * filter task is needed. If this argument is nullptr, no rows will be filtered out.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   * @param[in] force_zero_copy_disable Override the zero-copy optimization parameter with a disable
   * command. Used, e.g., for unit testing or when zero-copy is unsafe due to the system
   * configuration.
   */
  in_memory_read_task(context_reference ctx_ref,
                      int32_t task_id,
                      int32_t stage_id,
                      std::vector<const row_group_interface*> row_groups,
                      std::vector<cudf::size_type> column_indexes,
                      std::vector<cudf::data_type> data_types,
                      memory_kind::type memory_kind,
                      std::optional<arrow::compute::Expression> partial_filter = std::nullopt,
                      std::vector<std::shared_ptr<task>> subquery_tasks        = {},
                      bool force_zero_copy_disable                             = false);

  in_memory_read_task(const in_memory_read_task&)            = delete;
  in_memory_read_task& operator=(const in_memory_read_task&) = delete;

  void execute() override;

 private:
  /// Determine if a zero-copy read is possible, based on the number of row group partitions and
  /// environment parameters.
  [[nodiscard]] bool is_zero_copy_possible(const pruning_results_t& pruning_results) const;

  /// Construct an empty table with the schema of the read task and emit it as an owned result.
  void emit_empty_table();

  /// Emit a single row group partition as a borrowed result, which is accessed via zero-copy.
  void emit_zero_copy_result(std::unique_ptr<pruning_results_t> pruning_results);

  /// Copy one or more row groups to GPU memory and emit an owned result.
  void emit_copied_result(std::unique_ptr<pruning_results_t> pruning_results);

  std::vector<const row_group_interface*>
    _row_groups; /**< Non-owning references to the row groups assigned to this task. */
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  memory_kind::type _memory_kind;
  std::optional<arrow::compute::Expression> _partial_filter;
  bool _force_zero_copy_disable;
};

}  // namespace storage
}  // namespace gqe
