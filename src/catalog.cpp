/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/catalog.hpp>

#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace gqe {

column_traits::column_traits(std::string const& name_,
                             cudf::data_type const& data_type_,
                             std::vector<column_property> const& props)
  : name(name_), data_type(data_type_)
{
  for (auto prop : props) {
    if (prop == column_property::unique) is_unique = true;
  }
}

catalog::catalog(task_manager_context* ctx) : _task_manager_context(ctx)
{
  if (!ctx) { throw std::invalid_argument("Catalog requires a non-null task_manager_context"); }
}

void catalog::register_table(std::string const& table_name,
                             std::vector<column_traits> const& columns,
                             storage_kind::type storage,
                             partitioning_schema_kind::type partitioning_schema)
{
  if (_table_entries.count(table_name))
    throw std::logic_error("table \"" + table_name + "\" is already registered");

  table_info_type table_info;
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  column_names.reserve(columns.size());
  column_types.reserve(columns.size());
  for (auto const& column : columns) {
    auto& column_name = column.name;
    auto& column_type = column.data_type;
    if (table_info._column_name_to_type.count(column_name))
      throw std::logic_error("column name already exists when registering table");
    table_info._column_name_to_type[column_name] = column_type;
    table_info._column_name_to_uniq[column_name] = column.is_unique;
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

  table_info._statistics =
    std::make_unique<table_statistics_manager>(table_statistics(num_rows, column_types));

  std::unique_ptr<storage::table> table = std::visit(
    utility::overloaded{
      [](storage_kind::parquet_file file) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::parquet_table>(std::move(file.file_paths));
      },
      [&](storage_kind::system_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::system{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::numa_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::numa{memory.numa_node_set, memory.page_kind},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::numa_pinned_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::numa_pinned{memory.numa_node_set, memory.page_kind},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::pinned_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::pinned{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::device_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::device{memory.device_id}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::managed_memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::managed{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::boost_shared_memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::boost_shared{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::numa_pool_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::numa_pool{memory.numa_node_id},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::shared_numa_pool_memory memory) -> std::unique_ptr<storage::table> {
        return std::make_unique<storage::in_memory_table>(
          memory_kind::shared_numa_pool{memory.numa_node_id},
          column_names,
          column_types,
          _task_manager_context);
      }},
    storage);

  table_entry entry = {std::move(table_info), std::move(table)};

  _table_entries[table_name] = std::move(entry);
}

const std::vector<std::string>& catalog::column_names(std::string const& table_name) const
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

std::vector<cudf::data_type> catalog::column_types(std::string const& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  // Maybe it would be better to store the data types when registering the table or use a dedicated
  // column type.
  auto const& column_names        = table_info_iter->second.info._column_names;
  auto const& column_name_to_type = table_info_iter->second.info._column_name_to_type;
  std::vector<cudf::data_type> column_types;
  column_types.reserve(column_names.size());
  std::for_each(column_names.begin(), column_names.end(), [&](std::string const& column_name) {
    const auto& column_type = column_name_to_type.at(column_name);
    column_types.push_back(column_type);
  });
  return column_types;
}

bool catalog::column_is_unique(std::string const& table_name, std::string const& column_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  auto const& column_name_to_uniq = table_info_iter->second.info._column_name_to_uniq;

  auto const column_uniq_iter = column_name_to_uniq.find(column_name);
  if (column_uniq_iter == column_name_to_uniq.end())
    throw std::logic_error("cannot find column \"" + column_name + "\" of table \"" + table_name +
                           "\" in the catalog");

  return column_uniq_iter->second;
}

storage_kind::type catalog::storage_kind(const std::string& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second.info._storage;
}

table_statistics_manager* catalog::statistics(std::string const& table_name) const
{
  auto const table_info_iter = _table_entries.find(table_name);

  if (table_info_iter == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return table_info_iter->second.info._statistics.get();
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

std::vector<std::string> catalog::table_names() const
{
  std::vector<std::string> table_names;
  for (auto const& [table_name, _] : _table_entries) {
    table_names.push_back(table_name);
  }
  return table_names;
}

}  // namespace gqe
