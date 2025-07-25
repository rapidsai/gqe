/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/optimizer/statistics.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {

/**
 * @brief Struct for consolidating column traits.
 */
struct column_traits {
  enum class column_property { unique };

  /**
   * @brief Constructor for `column_traits`.
   *
   * @param[in] name_ Name of the column.
   * @param[in] data_type_ Data type of the column.
   * @param[in] props Brace-enclosed initializer list of all properties (see `enum class
   * column_property`) that the column possesses. Any duplicates are ignored.
   */
  column_traits(std::string const& name_,
                cudf::data_type const& data_type_,
                std::vector<column_property> const& props = {});

  std::string name;           // name of the column
  cudf::data_type data_type;  // data type of the column's elements
  bool is_unique =
    false;  // whether the column possesses the property that all elements are always unique
};

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
   * @param[in] columns A collection of `column_traits`.
   * @param[in] storage Storage hint to phyiscally store the table's data.
   * @param[in] partitioning_schema Partitioning schema with which the table's data are divided.
   */
  void register_table(std::string const& table_name,
                      std::vector<column_traits> const& columns,
                      storage_kind::type storage,
                      partitioning_schema_kind::type partitioning_schema);

  /**
   * @brief Return the names of all columns in the table in the user-defined order.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  const std::vector<std::string>& column_names(std::string const& table_name) const;

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
   * @brief Return the data types of all columns of a table.
   * @param table_name A name of a table in the catalog.
   * @return A vector with the data types of the columns in the order of the column names.
   */
  std::vector<cudf::data_type> column_types(std::string const& table_name) const;

  /**
   * @brief Returns whether the specified column in the specified table is unique
   *
   * @param table_name Name of the table to check uniqueness
   * @param column_name Name of the column to to check uniqueness
   * @return true If the specified column in the specified table is unique
   * @return false Otherwise
   */
  bool column_is_unique(std::string const& table_name, std::string const& column_name) const;

  /**
   * @brief Return the storage type of a table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  storage_kind::type storage_kind(std::string const& table_name) const;

  /**
   * @brief Returns unowned pointer to statistics manager of the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  table_statistics_manager* statistics(std::string const& table_name) const;

  /**
   * @brief Return whether the table is readable or not.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  bool is_readable(std::string const& table_name) const;

  /**
   * @brief Return if the table is writeable.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  bool is_writeable(std::string const& table_name) const;

  /**
   * @brief Return the maximum number of concurrent readers.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  int32_t max_concurrent_readers(std::string const& table_name) const;

  /**
   * @brief Return the maximum number of concurrent writers.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  int32_t max_concurrent_writers(std::string const& table_name) const;

  /**
   * @brief Return a readable view to the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::unique_ptr<storage::readable_view> readable_view(std::string const& table_name) const;

  /**
   * @brief Return a writeable view to the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::unique_ptr<storage::writeable_view> writeable_view(std::string const& table_name) const;

 private:
  struct table_info_type {
    std::vector<std::string> _column_names;  ///< column names in the user-defined order
    std::unordered_map<std::string, cudf::data_type>
      _column_name_to_type;  ///< map from column names to their data types
    std::unordered_map<std::string, bool>
      _column_name_to_uniq;  ///< map from column names whether they are unique
    storage_kind::type _storage;
    partitioning_schema_kind::type _partitioning_schema;
    std::unique_ptr<table_statistics_manager> _statistics;
  };

  struct table_entry {
    table_info_type info;                     ///< the table's metadata
    std::unique_ptr<storage::table> storage;  ///< the data's data
  };

  std::unordered_map<std::string, table_entry>
    _table_entries;  ///< map from the table name to its catalog entry
};

}  // namespace gqe
