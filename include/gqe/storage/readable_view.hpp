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
  virtual ~readable_view() = default;

  /**
   * @brief Return a read task for the table kind.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] parallelism The number of parallel read task instances that will
   * be instantiated.
   * @param[in] instance_id The unique identifier of this parallel instance.
   * @param[in] column_names Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   * @param[in] partial_filter Used to support predicate pushdown. Note that a row that satisfies
   * the predicate is guaranteed to be included in the loaded table, but a row that does not satisfy
   * the predicate may or may not be excluded. If such exclusion needs to be guaranteed, an extra
   * filter task is needed. If this argument is nullptr, no rows will be filtered out.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   */
  virtual std::unique_ptr<read_task_base> get_read_task(
    int32_t task_id,
    int32_t stage_id,
    int32_t parallelism,
    int32_t instance_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    std::unique_ptr<gqe::expression> partial_filter   = nullptr,
    std::vector<std::shared_ptr<task>> subquery_tasks = {}) = 0;
};

};  // namespace storage

};  // namespace gqe
