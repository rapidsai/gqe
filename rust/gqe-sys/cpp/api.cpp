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

#include <cxx_gqe/api.hpp>

#include <gqe/types.hpp>

#include <rmm/cuda_device.hpp>

#include <cxx_gqe/types.hpp>

// Include wrappers of Rust gqe-rs shared types.
#include <gqe-sys/src/lib.rs.h>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <iterator>
#include <memory>
#include <vector>

namespace cxx_gqe {

void catalog::register_table(const rust::Str table_name,
                             const rust::Slice<const column_schema> columns,
                             const storage_kind_type storage_type,
                             const void* storage_info,
                             const partitioning_schema_kind_type partitioning_schema_type,
                             const void* partitioning_schema_info)
{
  std::string cxx_name(table_name);

  std::vector<std::pair<std::string, cudf::data_type>> cxx_columns;
  cxx_columns.reserve(columns.size());
  std::transform(columns.begin(), columns.end(), std::back_inserter(cxx_columns), [](auto& cs) {
    return std::make_pair<std::string, cudf::data_type>(std::string(cs.column_name),
                                                        cudf::data_type(cs.data_type));
  });

  gqe::storage_kind::type cxx_storage;
  switch (storage_type) {
    case storage_kind_type::system_memory: {
      cxx_storage = gqe::storage_kind::system_memory{};
    } break;
    case storage_kind_type::numa_memory: {
      auto info             = reinterpret_cast<const numa_memory_info*>(storage_info);
      auto cxx_raw_node_set = reinterpret_cast<const cpu_set_t*>(info->numa_node_set.data());
      constexpr int32_t BITS_PER_BYTE = 8;
      gqe::cpu_set cxx_node_set(*cxx_raw_node_set, info->numa_node_set_bytes * BITS_PER_BYTE);
      cxx_storage = gqe::storage_kind::numa_memory{std::move(cxx_node_set), info->page_kind};
    } break;
    case storage_kind_type::pinned_memory: {
      cxx_storage = gqe::storage_kind::pinned_memory{};
    } break;
    case storage_kind_type::device_memory: {
      auto info = reinterpret_cast<const device_memory_info*>(storage_info);
      cxx_storage =
        gqe::storage_kind::device_memory{std::move(rmm::cuda_device_id(info->device_id))};
    } break;

    case storage_kind_type::managed_memory: {
      cxx_storage = gqe::storage_kind::managed_memory{};
    } break;
    case storage_kind_type::parquet_file: {
      auto info = reinterpret_cast<const parquet_file_info*>(storage_info);

      std::vector<std::string> cxx_files;
      cxx_files.reserve(info->file_paths.size());
      std::transform(info->file_paths.begin(),
                     info->file_paths.end(),
                     std::back_inserter(cxx_files),
                     [](auto& p) { return std::string(p); });

      cxx_storage = gqe::storage_kind::parquet_file{std::move(cxx_files)};
    } break;
    default: assert(false && "Got an unknown storage kind type from Rust.");
  }

  gqe::partitioning_schema_kind::type cxx_partitioning_schema;
  switch (partitioning_schema_type) {
    case partitioning_schema_kind_type::automatic: {
      cxx_partitioning_schema = gqe::partitioning_schema_kind::automatic{};
    } break;
    case partitioning_schema_kind_type::none: {
      cxx_partitioning_schema = gqe::partitioning_schema_kind::none{};
    } break;
    case partitioning_schema_kind_type::key: {
      auto info = reinterpret_cast<const key_schema_info*>(partitioning_schema_info);

      std::vector<std::string> cxx_columns;
      cxx_columns.reserve(info->columns.size());
      std::transform(
        info->columns.begin(), info->columns.end(), std::back_inserter(cxx_columns), [](auto& cs) {
          return std::string(cs);
        });
      cxx_partitioning_schema = gqe::partitioning_schema_kind::key{std::move(cxx_columns)};
    } break;
    default: assert(false && "Got an unknown partitioning schema kind type from Rust.");
  }

  _catalog.register_table(std::move(cxx_name),
                          std::move(cxx_columns),
                          std::move(cxx_storage),
                          std::move(cxx_partitioning_schema));
}

[[nodiscard]] std::unique_ptr<std::vector<std::string>> catalog::column_names(
  const rust::Str table_name) const
{
  std::string cxx_name(table_name);
  return std::make_unique<std::vector<std::string>>(_catalog.column_names(cxx_name));
}

[[nodiscard]] type_id catalog::column_type(const rust::Str table_name,
                                           const rust::Str column_name) const
{
  std::string cxx_table_name(table_name);
  std::string cxx_column_name(column_name);
  return _catalog.column_type(cxx_table_name, cxx_column_name).id();
}

std::unique_ptr<catalog> new_catalog() noexcept { return std::make_unique<catalog>(); }

}  // namespace cxx_gqe
