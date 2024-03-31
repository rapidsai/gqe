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
#include <gqe/executor/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gqe {

namespace storage {

/**
 * @brief A table consisting of one of more Parquet files.
 *
 * Optionally, the Parquet files may be stored as Hive partitions.
 */
class parquet_table : public table {
 public:
  /**
   * @brief Create a new Parquet table with existing files.
   *
   * @param[in] file_paths Paths to Parquet files containing data.
   */
  parquet_table(std::vector<std::string> file_paths);

  /**
   * @copydoc gqe::storage::table::is_readable()
   */
  [[nodiscard]] bool is_readable() const override;

  /**
   * @copydoc gqe::storage::table::is_writeable()
   */
  [[nodiscard]] bool is_writeable() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_readers()
   */
  [[nodiscard]] int32_t max_concurrent_readers() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_writers()
   */
  [[nodiscard]] int32_t max_concurrent_writers() const override;

  /**
   * @copydoc gqe::storage::table::readable_view()
   */
  std::unique_ptr<storage::readable_view> readable_view() override;

  /**
   * @copydoc gqe::storage::table::writeable_view()
   */
  std::unique_ptr<storage::writeable_view> writeable_view() override;

 private:
  std::shared_ptr<std::vector<std::string>> _file_paths;
};

class parquet_read_task : public read_task_base {
 public:
  /**
   * @brief Construct a Parquet read task.
   *
   * A read task is used for loading a table from a file.
   *
   * @param[in] query_context The query context in which the current task is running in.
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
  parquet_read_task(query_context* query_context,
                    int32_t task_id,
                    int32_t stage_id,
                    std::vector<std::string> file_paths,
                    std::vector<std::string> column_names,
                    std::vector<cudf::data_type> data_types,
                    std::unique_ptr<gqe::expression> partial_filter   = nullptr,
                    std::vector<std::shared_ptr<task>> subquery_tasks = {});

  parquet_read_task(const parquet_read_task&) = delete;
  parquet_read_task& operator=(const parquet_read_task&) = delete;

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  [[nodiscard]] std::unique_ptr<cudf::table> table_from_parquet(
    std::vector<std::string> const& file_paths) const;

  [[nodiscard]] std::string print_column_names() const;

  std::vector<std::string> _file_paths;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
  std::unique_ptr<gqe::expression> _partial_filter;
};

class parquet_write_task : public write_task_base {
 public:
  parquet_write_task(query_context* query_context,
                     int32_t task_id,
                     int32_t stage_id,
                     std::shared_ptr<task> input,
                     std::vector<std::string> file_paths,
                     std::vector<std::string> column_names,
                     std::vector<cudf::data_type> data_types);

  parquet_write_task(const parquet_write_task&) = delete;
  parquet_write_task& operator=(const parquet_write_task&) = delete;

  /**
   * @copydoc gqe::task::execute()
   */
  void execute() override;

 private:
  std::vector<std::string> _file_paths;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
};

/**
 * @brief Data access method to read a Parquet table.
 */
class parquet_readable_view : public readable_view {
  friend parquet_table;

 public:
  /**
   * @copydoc gqe::storage::readable_view::get_read_tasks()
   */
  std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<readable_view::task_parameters>&& task_parameters,
    query_context* query_context,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

 private:
  parquet_readable_view(std::vector<std::string>* non_owning_file_paths);

  std::vector<std::string>*
    _non_owning_file_paths /**< Non-owning reference to paths owned by `parquet_table`. */;
};

/**
 * @brief Data access method to write a Parquet table.
 */
class parquet_writeable_view : public writeable_view {
  friend parquet_table;

 public:
  /**
   * @copydoc gqe::storage::writeable_view::get_write_tasks()
   */
  std::vector<std::unique_ptr<write_task_base>> get_write_tasks(
    std::vector<writeable_view::task_parameters>&& task_parameters,
    query_context* query_context,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

 private:
  parquet_writeable_view(std::vector<std::string>* non_owning_file_paths);

  std::vector<std::string>*
    _non_owning_file_paths /**< Non-owning reference to paths owned by `parquet_table`. */;
};

};  // namespace storage

};  // namespace gqe
