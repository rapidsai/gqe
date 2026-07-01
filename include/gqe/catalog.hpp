/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/statistics.hpp>
#include <gqe/storage/descriptor.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {

// Forward declaration
//
// The forward declaration avoids pulling in `mpi.h`. Some tests wouldn't build without linking
// against MPI_CXX.
class task_manager_context;

/**
 * @brief Struct for consolidating column traits.
 */
struct column_traits {
  /**
   * @brief Constructor for `column_traits`.
   *
   * @param[in] name_ Name of the column.
   * @param[in] data_type_ Data type of the column.
   */
  column_traits(std::string const& name_, cudf::data_type const& data_type_);

  std::string name;           // name of the column
  cudf::data_type data_type;  // data type of the column's elements
};

/**
 * @brief Store the metadata of the tables.
 *
 * Before a table can be referenced in a SQL query, it must be registered in the catalog. The stored
 * metadata includes the table name, column names and their data types.
 */
class catalog : public storage::table_provider {
 public:
  /**
   * @brief Construct a catalog with a task manager context.
   *
   * @param[in] ctx Non-owning pointer to the task manager context. The context must outlive the
   catalog.

   * @pre `ctx` must not be null.
   */
  explicit catalog(task_manager_context* ctx);

  /**
   * @brief Register a new table into the catalog.
   *
   * @throw std::logic_error if `table_name` is already registered.
   *
   * @param[in] table_name Name of the table to create.
   * @param[in] columns A collection of `column_traits`.
   * @param[in] storage Storage hint to phyiscally store the table's data.
   * @param[in] partitioning_schema Partitioning schema with which the table's data are divided.
   * @param[in] unique_keys All UNIQUE / PRIMARY KEY constraints. Each inner vector is one key-set:
   *            size 1 = single-column unique, size >= 2 = composite unique.
   */
  void register_table(std::string const& table_name,
                      std::vector<column_traits> const& columns,
                      storage_kind::type storage,
                      partitioning_schema_kind::type partitioning_schema,
                      std::vector<std::vector<std::string>> const& unique_keys = {});

  /**
   * @brief Remove a table from the catalog.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   *
   * @param[in] table_name Name of the table to remove.
   */
  void unregister_table(std::string const& table_name);

  /**
   * @brief Check whether a table is registered in the catalog.
   *
   * @param[in] table_name Name of the table to check.
   * @return true if the table exists, false otherwise.
   */
  bool has_table(std::string const& table_name) const;

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
   * @brief Return all unique key-sets for a table. Each inner vector is one key: size 1 =
   * single-column unique, size >= 2 = composite unique.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::vector<std::vector<std::string>> const& unique_keys(std::string const& table_name) const;

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
   * @brief Returns shared pointer to statistics manager of the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::shared_ptr<table_statistics_manager> table_statistics(std::string const& table_name) const;

  /**
   * @brief Returns shared pointer to the table's storage.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  std::shared_ptr<storage::table> table_storage(std::string const& table_name) const;

  /** @copydoc storage::table_provider::get_table */
  [[nodiscard]] std::shared_ptr<storage::table> get_table(
    std::string_view table_name) const override;

  /**
   * @brief Write all current in-memory row groups under @p serialized_data_path.
   *
   * Intended to run after write tasks have finished. Creates @p serialized_data_path if needed.
   *
   * @param[in] serialized_data_path Table serialized-data root (`zmps-*` folder, e.g. from @ref
   *            gqe::utility::serialized_table_root).
   * @throw std::invalid_argument if @p serialized_data_path is empty.
   * @throw std::logic_error if the table is missing or storage is not @ref
   * storage::in_memory_table.
   * @throw std::runtime_error if directory creation or serialization fails.
   *
   * @note Lifetime: @p table_name must stay registered until this call returns. Do not @ref
   *       unregister_table for that table (or destroy the only `shared_ptr` to its storage) while
   *       serialization is in progress.
   */
  void serialize_table(std::string const& table_name,
                       std::string const& serialized_data_path) const;

  /**
   * @brief Load serialized row-group snapshots to disk from @p serialized_data_path into the
   * in-memory table.
   *
   * Intended to run after @ref register_table for an empty in-memory table.
   *
   * @param[in] serialized_data_path Table serialized-data root (`zmps-*` folder) with per-row-group
   *                                 `rg-*` directories.
   * @throw std::invalid_argument if @p serialized_data_path is empty.
   * @throw std::logic_error if the table is missing or storage is not @ref
   * storage::in_memory_table.
   * @throw std::runtime_error if deserialization fails.
   *
   * @note Lifetime: @p table_name must stay registered until this call returns. Do not @ref
   *       unregister_table for that table (or destroy the only `shared_ptr` to its storage) while
   *       deserialization is in progress.
   */
  void deserialize_table(std::string const& table_name,
                         std::string const& serialized_data_path) const;

  /**
   * @brief Returns a storage descriptor for the table.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  storage::descriptor storage_descriptor(std::string const& table_name) const;

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

  /**
   * @brief Return the names of all tables in the catalog.
   */
  std::vector<std::string> table_names() const;

  /**
   * @brief Format every registered table's basic stats and unique-key constraints
   * as a multi-line string, one indented line per table. Intended for trace-level
   * logging.
   */
  std::string to_string() const;

 private:
  task_manager_context*
    _task_manager_context;  ///< Non-owning pointer to the task manager context. The context field
                            ///< must be on top to ensure that it outlives the other catalog fields.

  struct table_info_type {
    std::vector<std::string> _column_names;  ///< column names in the user-defined order
    std::unordered_map<std::string, cudf::data_type>
      _column_name_to_type;  ///< map from column names to their data types
    std::vector<std::vector<std::string>>
      _unique_keys;  ///< all unique key-sets; each inner vec is one key (size 1 = single-col)
    storage_kind::type _storage;
    partitioning_schema_kind::type _partitioning_schema;
  };

  struct table_entry {
    table_info_type info;                                  ///< the table's metadata
    std::shared_ptr<storage::table> storage;               ///< the table's data
    std::shared_ptr<table_statistics_manager> statistics;  ///< the table's statistics
  };

  std::unordered_map<std::string, table_entry>
    _table_entries;  ///< map from table name to catalog entry; not thread-safe — external sync
                     ///< required if accessed concurrently
};

}  // namespace gqe
