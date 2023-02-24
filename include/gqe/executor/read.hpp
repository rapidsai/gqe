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
#include <gqe/expression/subquery.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace gqe {

class read_task : public task {
 public:
  /**
   * @brief Construct a read task.
   *
   * A read task is used for loading a table from a file.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] file_paths Paths of the files to be read.
   * @param[in] file_format Format of the file.
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
  read_task(int32_t task_id,
            int32_t stage_id,
            std::vector<std::string> file_paths,
            file_format_type file_format,
            std::vector<std::string> column_names,
            std::vector<cudf::data_type> data_types,
            std::unique_ptr<gqe::expression> partial_filter   = nullptr,
            std::vector<std::shared_ptr<task>> subquery_tasks = {});

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  [[nodiscard]] std::unique_ptr<cudf::table> table_from_parquet(
    std::vector<std::string> const& file_paths) const;

  [[nodiscard]] std::string print_column_names() const;

  std::vector<std::string> _file_paths;
  file_format_type _file_format;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
  std::unique_ptr<gqe::expression> _partial_filter;
};

}  // namespace gqe
