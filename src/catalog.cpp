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

#include <gqe/catalog.hpp>

#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/serialization.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace gqe {

column_traits::column_traits(std::string const& name_, cudf::data_type const& data_type_)
  : name(name_), data_type(data_type_)
{
}

catalog::catalog(task_manager_context* ctx) : _task_manager_context(ctx)
{
  if (!ctx) { throw std::invalid_argument("Catalog requires a non-null task_manager_context"); }
}

void catalog::register_table(std::string const& table_name,
                             std::vector<column_traits> const& columns,
                             storage_kind::type storage,
                             partitioning_schema_kind::type partitioning_schema,
                             std::vector<std::vector<std::string>> const& unique_keys)
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
    column_names.push_back(std::move(column_name));
    column_types.push_back(std::move(column_type));
  }
  table_info._column_names        = column_names;
  table_info._unique_keys         = unique_keys;
  table_info._storage             = storage;
  table_info._partitioning_schema = partitioning_schema;

  int64_t num_rows = 0;
  if (auto file = std::get_if<storage_kind::parquet_file>(&storage)) {
    // FIXME: Parse the table metadata for a more accurate estimation
    constexpr int64_t default_num_rows_per_file = 10000;
    num_rows = default_num_rows_per_file * file->file_paths.size();
  }

  auto stats =
    std::make_shared<table_statistics_manager>(gqe::table_statistics(num_rows, column_types));

  std::shared_ptr<storage::table> table = std::visit(
    utility::overloaded{
      [](storage_kind::parquet_file file) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::parquet_table>(std::move(file.file_paths));
      },
      [&](storage_kind::system_memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::system{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::numa_memory memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::numa{memory.numa_node_set, memory.page_kind},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::numa_pinned_memory memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::numa_pinned{memory.numa_node_set, memory.page_kind},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::pinned_memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::pinned{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::device_memory memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::device{memory.device_id}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::managed_memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::managed{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::boost_shared_memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::boost_shared{}, column_names, column_types, _task_manager_context);
      },
      [&](storage_kind::numa_pool_memory memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::numa_pool{memory.numa_node_id},
          column_names,
          column_types,
          _task_manager_context);
      },
      [&](storage_kind::shared_numa_pool_memory memory) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::shared_numa_pool{memory.numa_node_id},
          column_names,
          column_types,
          _task_manager_context);
      }},
    storage);

  table_entry entry = {std::move(table_info), std::move(table), std::move(stats)};

  _table_entries[table_name] = std::move(entry);
}

bool catalog::has_table(std::string const& table_name) const
{
  return _table_entries.find(table_name) != _table_entries.end();
}

void catalog::unregister_table(std::string const& table_name)
{
  auto it = _table_entries.find(table_name);
  if (it == _table_entries.end())
    throw std::logic_error("cannot unregister table \"" + table_name +
                           "\": not found in the catalog");
  _table_entries.erase(it);
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

std::vector<std::vector<std::string>> const& catalog::unique_keys(
  std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);
  if (it == _table_entries.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  return it->second.info._unique_keys;
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

  return table_info_iter->second.statistics.get();
}

std::shared_ptr<table_statistics_manager> catalog::table_statistics(
  std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);

  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return it->second.statistics;
}

std::shared_ptr<storage::table> catalog::table_storage(std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);

  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return it->second.storage;
}

std::shared_ptr<storage::table> catalog::get_table(std::string_view table_name) const
{
  auto const it = _table_entries.find(std::string{table_name});
  if (it == _table_entries.end()) {
    throw std::logic_error(std::format("Table '{}' not found in catalog", table_name));
  }
  return it->second.storage;
}

void catalog::serialize_table(std::string const& table_name,
                              std::string const& serialized_data_path) const
{
  // Lifetime: `table_name` must remain registered (and its storage object kept alive via the
  // catalog entry or another owning `shared_ptr`) until this function returns.
  if (serialized_data_path.empty()) {
    throw std::invalid_argument("serialize_table requires a non-empty serialized_data_path");
  }

  auto const iterator = _table_entries.find(table_name);
  if (iterator == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  auto* in_memory_table = dynamic_cast<storage::in_memory_table*>(iterator->second.storage.get());
  if (in_memory_table == nullptr) {
    throw std::logic_error("serialize_table requires an in-memory table: \"" + table_name + "\"");
  }

  std::filesystem::create_directories(std::filesystem::path(serialized_data_path));
  GQE_LOG_TRACE("Created tables serialized data root: {}", serialized_data_path);

  auto stream = cudf::get_default_stream();
  stream.synchronize();
  in_memory_table->serialize_table_to_disk(serialized_data_path, table_name, stream);
}

void catalog::deserialize_table(std::string const& table_name,
                                std::string const& serialized_data_path) const
{
  // Lifetime: `table_name` must remain registered (and its storage object kept alive via the
  // catalog entry or another owning `shared_ptr`) until this function returns.
  if (serialized_data_path.empty()) {
    throw std::invalid_argument("deserialize_table requires a non-empty serialized_data_path");
  }

  auto const iterator = _table_entries.find(table_name);
  if (iterator == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  auto* in_memory_table = dynamic_cast<storage::in_memory_table*>(iterator->second.storage.get());
  if (in_memory_table == nullptr) {
    throw std::logic_error("deserialize_table requires an in-memory table: \"" + table_name + "\"");
  }

  auto stream = cudf::get_default_stream();
  stream.synchronize();
  in_memory_table->deserialize_table_from_disk(serialized_data_path, table_name, stream);
}

storage::descriptor catalog::storage_descriptor(std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);
  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  return storage::make_descriptor(
    it->second.info._storage, table_name, it->second.info._column_names, column_types(table_name));
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

std::string catalog::to_string() const
{
  std::string s;
  bool first = true;
  for (auto const& [table_name, entry] : _table_entries) {
    auto const& stats  = entry.statistics->statistics();
    auto const& uks    = entry.info._unique_keys;
    std::string uk_str = "[";
    for (size_t i = 0; i < uks.size(); ++i) {
      uk_str += '[';
      for (size_t j = 0; j < uks[i].size(); ++j) {
        uk_str += uks[i][j];
        if (j + 1 < uks[i].size()) uk_str += ',';
      }
      uk_str += ']';
      if (i + 1 < uks.size()) uk_str += ',';
    }
    uk_str += ']';
    if (!first) s += '\n';
    first = false;
    s += std::format("  {}: num_rows={}, num_row_groups={}, num_columns={}, unique_keys={}",
                     table_name,
                     stats.num_rows,
                     stats.num_row_groups,
                     stats.num_columns,
                     uk_str);
  }
  return s;
}

}  // namespace gqe
