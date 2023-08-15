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

#include <gqe/catalog.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace gqe {

void catalog::register_table(std::string table_name,
                             const std::vector<std::pair<std::string, cudf::data_type>>& columns,
                             storage_kind::type storage,
                             partitioning_schema_kind::type partitioning_schema)
{
  if (_table_entries.find(table_name) != _table_entries.end())
    throw std::logic_error("table \"" + table_name + "\" is already registered");

  table_info_type table_info;
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  column_names.reserve(columns.size());
  column_types.reserve(columns.size());
  for (auto const& column : columns) {
    auto const& column_name = column.first;
    auto const& column_type = column.second;
    if (table_info._column_name_to_type.find(column_name) != table_info._column_name_to_type.end())
      throw std::logic_error("column name already exists when registering table");
    table_info._column_name_to_type[column_name] = column_type;
    column_names.push_back(std::move(column_name));
    column_types.push_back(std::move(column_type));
  }
  table_info._column_names        = column_names;
  table_info._storage             = storage;
  table_info._partitioning_schema = partitioning_schema;

  int64_t num_rows = 0;
  if (auto file = std::get_if<storage_kind::parquet_file>(&storage)) {
    // FIXME: Parse the table metadata for a more accurate estimation
    constexpr int64_t default_num_rows_per_file = 10000;
    num_rows = default_num_rows_per_file * file->file_paths.size();
  }

  table_statistics stats;
  stats.num_rows         = num_rows;
  table_info._statistics = stats;

  std::unique_ptr<storage::table> table = std::visit(
    utility::overloaded{
      [](storage_kind::parquet_file file) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::parquet_table>(std::move(file.file_paths));
      },
      [&](storage_kind::system_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::system{}, column_names, column_types);
      },
      [&](storage_kind::numa_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::numa{memory.numa_node_set, memory.page_kind}, column_names, column_types);
      },
      [&](storage_kind::pinned_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::pinned{}, column_names, column_types);
      },
      [&](storage_kind::device_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::device{memory.device_id}, column_names, column_types);
      },
      [&](storage_kind::managed_memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::managed{}, column_names, column_types);
      }},
    storage);

  table_entry entry = {std::move(table_info), std::move(table)};

  _table_entries[table_name] = std::move(entry);
}

std::vector<std::string> catalog::column_names(std::string const& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second.info._column_names;
}

cudf::data_type catalog::column_type(std::string const& table_name,
                                     std::string const& column_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  auto const& column_name_to_type = table_info_iter->second.info._column_name_to_type;

  auto const column_type_iter = column_name_to_type.find(column_name);
  if (column_type_iter == column_name_to_type.end())
    throw std::logic_error("cannot find column \"" + column_name + "\" of table \"" + table_name +
                           "\" in the catalog");

  return column_type_iter->second;
}

storage_kind::type catalog::storage_kind(const std::string& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second.info._storage;
}

table_statistics catalog::statistics(std::string const& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second.info._statistics;
}

bool catalog::is_readable(std::string const& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->is_readable();
}

bool catalog::is_writeable(std::string const& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("Cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->is_writeable();
}

int32_t catalog::max_concurrent_readers(std::string const& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("Cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->max_concurrent_readers();
}

int32_t catalog::max_concurrent_writers(std::string const& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("Cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->max_concurrent_writers();
}

std::unique_ptr<storage::readable_view> catalog::readable_view(std::string const& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->readable_view();
}

std::unique_ptr<storage::writeable_view> catalog::writeable_view(
  const std::string& table_name) const
{
  auto const table_storage_iter = _table_entries.find(table_name);

  if (table_storage_iter == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_storage_iter->second.storage->writeable_view();
}

}  // namespace gqe
