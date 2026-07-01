/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/task_manager/remote_table_cache.hpp>

#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/utility/helpers.hpp>

#include <format>

namespace gqe::task_manager {

namespace {

/**
 * @brief Construct a storage::table from a descriptor.
 *
 * @param[in] desc The descriptor that specifies how to access the table.
 * @param[in] ctx  Task manager context used to construct in-memory tables.
 * @return A newly created table.
 * @throws std::logic_error If the descriptor kind is not supported.
 */
std::shared_ptr<storage::table> make_table(storage::descriptor const& desc,
                                           task_manager_context* ctx)
{
  return std::visit(
    utility::overloaded{
      [](storage::parquet_file_descriptor const& d) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::parquet_table>(d.file_paths);
      },
      [ctx](storage::boost_shared_memory_descriptor const& d) -> std::shared_ptr<storage::table> {
        return std::make_shared<storage::in_memory_table>(
          memory_kind::boost_shared{}, d.column_names, d.data_types, ctx);
      },
      [](storage::shared_numa_pool_memory_descriptor const& d) -> std::shared_ptr<storage::table> {
        throw std::logic_error(std::format(
          "Table '{}': shared NUMA pool memory storage not yet implemented", d.table_name));
      },
      [](storage::local_memory_descriptor const& d) -> std::shared_ptr<storage::table> {
        throw std::logic_error(std::format(
          "Table '{}': local memory descriptor cannot be used in a task manager", d.table_name));
      }},
    desc);
}

}  // namespace

remote_table_cache::remote_table_cache(task_manager_context* ctx) : _ctx(ctx) {}

void remote_table_cache::update(std::vector<storage::descriptor> descriptors)
{
  for (auto& desc : descriptors) {
    if (_descriptors.contains(desc)) { continue; }

    // Descriptor is new or changed. Remove the old one for the same table name
    // and eagerly create the new table.
    auto name = storage::table_name_of(desc);
    std::erase_if(_descriptors,
                  [&name](auto const& d) { return storage::table_name_of(d) == name; });
    _tables[name] = make_table(desc, _ctx);
    _descriptors.insert(std::move(desc));
  }
}

std::shared_ptr<storage::table> remote_table_cache::get_table(std::string_view table_name) const
{
  auto it = _tables.find(std::string{table_name});
  if (it == _tables.end()) {
    throw std::logic_error(std::format("Table '{}' not found in remote table cache", table_name));
  }
  return it->second;
}

}  // namespace gqe::task_manager
