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
#include <gqe/storage/shared_memory.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/serialization.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <system_error>
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
  register_table_impl(table_name,
                      columns,
                      std::move(storage),
                      std::move(partitioning_schema),
                      unique_keys,
                      true,
                      true);
}

void catalog::register_system_table(std::string const& table_name,
                                    std::vector<column_traits> const& columns,
                                    storage_kind::type storage,
                                    partitioning_schema_kind::type partitioning_schema,
                                    std::vector<std::vector<std::string>> const& unique_keys)
{
  register_table_impl(table_name,
                      columns,
                      std::move(storage),
                      std::move(partitioning_schema),
                      unique_keys,
                      false,
                      false);
}

void catalog::register_table_impl(std::string const& table_name,
                                  std::vector<column_traits> const& columns,
                                  storage_kind::type storage,
                                  partitioning_schema_kind::type partitioning_schema,
                                  std::vector<std::vector<std::string>> const& unique_keys,
                                  bool allow_user_ddl,
                                  bool allow_user_writes)
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
      [&](storage_kind::shared_segment_memory) -> std::shared_ptr<storage::table> {
        auto& segment = _task_manager_context->shared_memory_segment();
        shared_memory_allocator alloc(segment.get_segment_manager());
        auto* table_storage = segment.construct<storage::shared_in_memory_table_storage>(
          boost::interprocess::anonymous_instance)(alloc);
        try {
          auto const table_handle = segment.get_handle_from_address(table_storage);
          return std::make_shared<storage::shared_in_memory_table>(
            &segment, column_names, column_types, table_handle);
        } catch (...) {
          segment.destroy_ptr(table_storage);
          throw;
        }
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

  table_entry entry = {std::move(table_info),
                       std::move(table),
                       std::move(stats),
                       utility::uuid::generate(),
                       allow_user_ddl,
                       allow_user_writes,
                       false};

  _table_entries[table_name] = std::move(entry);
  _stats_version.fetch_add(1, std::memory_order_relaxed);
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

  if (!it->second.allow_user_ddl) {
    throw std::logic_error("cannot unregister system-managed table \"" + table_name + "\"");
  }

  // TODO: Tables can still leak memory during construction.
  // https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/402
  if (auto in_memory_table =
        std::dynamic_pointer_cast<storage::in_memory_table_interface>(it->second.storage)) {
    auto stream = cudf::get_default_stream();
    auto mr = _task_manager_context->get_table_memory_resource(in_memory_table->get_memory_kind());
    it->second.storage->deallocate(stream, mr);
    GQE_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }

  _table_entries.erase(it);
  _stats_version.fetch_add(1, std::memory_order_relaxed);
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

  auto* in_memory_table =
    dynamic_cast<storage::in_memory_table_interface*>(iterator->second.storage.get());
  if (in_memory_table == nullptr) {
    throw std::logic_error("serialize_table requires an in-memory table: \"" + table_name + "\"");
  }

  std::filesystem::path const root(serialized_data_path);
  auto const marker = utility::serialized_snapshot_marker_path(root);

  std::error_code remove_marker_error;
  std::filesystem::remove(marker, remove_marker_error);
  if (remove_marker_error) {
    throw std::runtime_error(std::format(
      "Cannot remove the completion marker '{}' of table '{}': {}. The previous snapshot is "
      "unchanged.",
      marker.string(),
      table_name,
      remove_marker_error.message()));
  }

  // From here on, a failure must not leave files that a later snapshot could be mistaken to
  // include.
  try {
    // Replace the snapshot as a whole, so no file of the previous one outlives it.
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    GQE_LOG_TRACE("Created tables serialized data root: {}", serialized_data_path);

    auto stream = cudf::get_default_stream();
    stream.sync();
    in_memory_table->serialize_table_to_disk(serialized_data_path,
                                             table_name,
                                             _task_manager_context->get_optimization_parameters(),
                                             stream);

    std::ofstream marker_file(marker);
    marker_file.close();
    if (!marker_file) {
      throw std::runtime_error(std::format(
        "Cannot write the completion marker '{}' of table '{}'", marker.string(), table_name));
    }
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    if (cleanup_error) {
      GQE_LOG_ERROR("Cannot remove the incomplete snapshot '{}' of table '{}': {}",
                    serialized_data_path,
                    table_name,
                    cleanup_error.message());
    }
    throw;
  }
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
  auto& entry = iterator->second;

  auto* in_memory_table = dynamic_cast<storage::in_memory_table_interface*>(entry.storage.get());
  if (in_memory_table == nullptr) {
    throw std::logic_error("deserialize_table requires an in-memory table: \"" + table_name + "\"");
  }

  if (!std::filesystem::is_regular_file(
        utility::serialized_snapshot_marker_path(serialized_data_path))) {
    throw std::invalid_argument(std::format(
      "The snapshot of table '{}' at '{}' has no completion marker. It is incomplete, or a GQE "
      "version without completion markers wrote it. Regenerate it by loading the table with COPY "
      "while serialize_tables_to_disk is enabled.",
      table_name,
      serialized_data_path));
  }

  auto stream = cudf::get_default_stream();
  stream.sync();
  // Source the table's memory resource here, because the catalog owns the task_manager_context and
  // the shared table does not. The context owns the segment and caches the resource borrowing it.
  auto const mr =
    _task_manager_context->get_table_memory_resource(in_memory_table->get_memory_kind());
  GQE_LOG_TRACE("catalog::deserialize_table: table='{}' path='{}' -> deserialize_table_from_disk",
                table_name,
                serialized_data_path);
  in_memory_table->deserialize_table_from_disk(serialized_data_path,
                                               table_name,
                                               _task_manager_context->get_optimization_parameters(),
                                               mr,
                                               stream,
                                               entry.statistics.get());
}

storage::descriptor catalog::storage_descriptor(std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);
  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }

  auto const& entry           = it->second;
  auto const types            = column_types(table_name);
  auto const storage_revision = entry.storage->storage_revision();
  return std::visit(
    utility::overloaded{
      [&](storage_kind::shared_segment_memory const&) -> storage::descriptor {
        auto shared_table =
          std::dynamic_pointer_cast<storage::shared_in_memory_table>(entry.storage);
        if (!shared_table) {
          throw std::logic_error(
            std::format("Table '{}' uses shared_segment_memory storage but is not backed by a "
                        "shared_in_memory_table",
                        table_name));
        }
        return storage::shared_memory_descriptor{table_name,
                                                 entry.table_id,
                                                 storage_revision,
                                                 entry.info._column_names,
                                                 types,
                                                 shared_table->shared_table_handle()};
      },
      [&](storage_kind::shared_numa_pool_memory const& memory) -> storage::descriptor {
        return storage::shared_numa_pool_memory_descriptor{table_name,
                                                           entry.table_id,
                                                           storage_revision,
                                                           memory.numa_node_id,
                                                           entry.info._column_names,
                                                           types};
      },
      [&](storage_kind::parquet_file const& file) -> storage::descriptor {
        return storage::parquet_file_descriptor{
          table_name, entry.table_id, storage_revision, file.file_paths};
      },
      [&](auto const&) -> storage::descriptor {
        // Process-local memory kinds (system, device, pinned, managed, etc.)
        return storage::local_memory_descriptor{table_name, entry.table_id};
      }},
    entry.info._storage);
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

  return table_storage_iter->second.allow_user_writes &&
         table_storage_iter->second.storage->is_writeable();
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

  if (!table_storage_iter->second.allow_user_writes) {
    throw std::logic_error("cannot write to system-managed table \"" + table_name + "\"");
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

std::uint64_t catalog::stats_version() const
{
  return _stats_version.load(std::memory_order_relaxed);
}

void catalog::append_table_statistics(std::string const& table_name,
                                      gqe::table_statistics const& stats)
{
  statistics(table_name)->append_table_statistics(stats);
  _stats_version.fetch_add(1, std::memory_order_relaxed);
}

void catalog::mark_unusable(std::string const& table_name)
{
  auto const it = _table_entries.find(table_name);
  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }
  it->second.is_unusable = true;
}

bool catalog::is_unusable(std::string const& table_name) const
{
  auto const it = _table_entries.find(table_name);
  if (it == _table_entries.end()) {
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  }
  return it->second.is_unusable;
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
