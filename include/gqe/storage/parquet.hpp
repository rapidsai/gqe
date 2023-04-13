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
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/utility.hpp>

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
   * @copydoc gqe::storage::table::readable_view()
   */
  std::unique_ptr<storage::readable_view> readable_view() override;

 private:
  std::shared_ptr<std::vector<std::string>> _file_paths;
};

/**
 * @brief Data access method to read a Parquet table.
 */
class parquet_readable_view : public readable_view {
  friend parquet_table;

 public:
  /**
   * @copydoc gqe::storage::readable_view::get_read_task()
   */
  std::unique_ptr<read_task_base> get_read_task(
    int32_t task_id,
    int32_t stage_id,
    uint32_t parallelism,
    uint32_t instance_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    std::unique_ptr<gqe::expression> partial_filter   = nullptr,
    std::vector<std::shared_ptr<task>> subquery_tasks = {}) override;

 private:
  parquet_readable_view(std::vector<std::string>* non_owning_file_paths);

  std::vector<std::string>*
    _non_owning_file_paths /**< Non-owning reference to paths owned by `parquet_table`. */;
};

};  // namespace storage

};  // namespace gqe
