/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/read.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/query_context.hpp>

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
   * @param[in] query_context The query context in which the current task is running in.
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
    query_context* query_context,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) = 0;
};

};  // namespace storage

};  // namespace gqe
