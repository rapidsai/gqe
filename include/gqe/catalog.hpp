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

#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {

/**
 * @brief Store the metadata of the tables.
 *
 * Before a table can be referenced in a SQL query, it must be registered in the catalog. The stored
 * metadata includes the table name, column names and their data types.
 */
class catalog {
 public:
  /**
   * @brief Register a new table into the catalog.
   *
   * @throw std::logic_error if `table_name` is already registered.
   *
   * @param[in] table_name Name of the table to create.
   * @param[in] columns A collection of (column name, column data type) pairs.
   * @param[in] storage Storage hint to phyiscally store the table's data.
   * @param[in] partitioning_schema Partitioning schema with which the table's data are divided.
   */
  void register_table(std::string table_name,
                      std::vector<std::pair<std::string, cudf::data_type>> const& columns,
                      storage_kind::type storage,
                      partitioning_schema_kind::type partitioning_schema);

  /**
   * @brief Register a new table into the catalog.
   *
   * @throw std::logic_error if `table_name` is already registered.
   *
   * @param[in] table_name Name of the table to register.
   * @param[in] columns A collection of (column name, column data type) pairs.
   * @param[in] file_paths List of files for containing data of `table_name`.
   * @param[in] file_format Format of files in `file_paths`.
   */
  void register_table(std::string table_name,
                      std::vector<std::pair<std::string, cudf::data_type>> const& columns,
                      std::vector<std::string> const& file_paths,
                      file_format_type file_format);

  /**
   * @brief Return the data type of a column in the catalog.
   *
   * @throw std::logic_error if the table or the column is not found in the catalog.
   *
   * @param[in] table_name Table name of the query column.
   * @param[in] column_name Column name of the query column.
   */
  cudf::data_type column_type(std::string const& table_name, std::string const& column_name) const;

  /**
   * @brief Return the storage type of a table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  storage_kind::type storage_kind(std::string const& table_name) const;

  /**
   * @brief Return the estimated statistics of a table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  table_statistics statistics(std::string const& table_name) const;

  /**
   * @brief Return the number of read tasks to be generated for this table
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  size_t num_partitions(std::string const& table_name) const;

  /**
   * @brief Return whether the table is readable or not.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  bool is_readable(std::string const& table_name) const;

  /**
   * @brief Return a readable view to the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::unique_ptr<storage::readable_view> readable_view(std::string const& table_name) const;

 private:
  struct table_info_type {
    std::unordered_map<std::string, cudf::data_type>
      _column_name_to_type;  ///< map from column names to their data types
    storage_kind::type _storage;
    partitioning_schema_kind::type _partitioning_schema;
    table_statistics _statistics;
    size_t _num_partitions;  ///< number of read tasks to be generated for this table
  };

  struct table_entry {
    table_info_type info;                     ///< the table's metadata
    std::unique_ptr<storage::table> storage;  ///< the data's data
  };

  std::unordered_map<std::string, table_entry>
    _table_entries;  ///< map from the table name to its catalog entry
};

}  // namespace gqe
