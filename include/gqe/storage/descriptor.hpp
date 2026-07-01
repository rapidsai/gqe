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

#pragma once

#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace gqe::storage {

/**
 * @brief Descriptor for tables stored in boost shared memory.
 *
 * Carries enough information to reconstruct a storage::table in another
 * process via the shared IPC segment.
 */
struct boost_shared_memory_descriptor {
  std::string table_name;                   ///< Key to locate the shared_table in the IPC segment.
  std::vector<std::string> column_names;    ///< Column names for reconstructing the table schema.
  std::vector<cudf::data_type> data_types;  ///< Column types for reconstructing the table schema.

  /** @brief Compute a hash over all fields. */
  [[nodiscard]] std::size_t hash() const;
  bool operator==(boost_shared_memory_descriptor const&) const = default;
};

/**
 * @brief Descriptor for tables stored in a shared NUMA pool.
 */
struct shared_numa_pool_memory_descriptor {
  std::string table_name;                   ///< Key to locate the shared_table.
  int numa_node_id;                         ///< NUMA node that owns the memory pool.
  std::vector<std::string> column_names;    ///< Column names for reconstructing the table schema.
  std::vector<cudf::data_type> data_types;  ///< Column types for reconstructing the table schema.

  /** @brief Compute a hash over all fields. */
  [[nodiscard]] std::size_t hash() const;
  bool operator==(shared_numa_pool_memory_descriptor const&) const = default;
};

/**
 * @brief Descriptor for tables backed by Parquet files.
 */
struct parquet_file_descriptor {
  std::string table_name;               ///< Logical table name.
  std::vector<std::string> file_paths;  ///< Paths to the Parquet files.

  /** @brief Compute a hash over all fields. */
  [[nodiscard]] std::size_t hash() const;
  bool operator==(parquet_file_descriptor const&) const = default;
};

/**
 * @brief Descriptor for process-local in-memory tables (system_memory,
 * device_memory, pinned_memory, etc.).
 *
 * These are only usable in single-process mode where the factory resolves
 * them by table name via the catalog.
 */
struct local_memory_descriptor {
  std::string table_name;  ///< Logical table name.

  /** @brief Compute a hash over all fields. */
  [[nodiscard]] std::size_t hash() const;
  bool operator==(local_memory_descriptor const&) const = default;
};

/** @brief A self-contained description of how to access a table's storage. */
using descriptor = std::variant<boost_shared_memory_descriptor,
                                shared_numa_pool_memory_descriptor,
                                parquet_file_descriptor,
                                local_memory_descriptor>;

/**
 * @brief Extract the table name from any descriptor variant.
 *
 * @param[in] desc The descriptor to query.
 * @return The table name stored in the descriptor.
 */
[[nodiscard]] std::string table_name_of(descriptor const& desc);

/**
 * @brief Create a storage descriptor from a storage_kind, table name, and schema.
 *
 * Column names and types are embedded in shared-memory descriptors so that
 * a task manager can reconstruct the table without access to a catalog.
 *
 * @param[in] kind         The storage kind to describe.
 * @param[in] table_name   Logical name of the table.
 * @param[in] column_names Names of the columns.
 * @param[in] column_types Data types of the columns.
 * @return A descriptor matching the given storage kind.
 */
[[nodiscard]] descriptor make_descriptor(storage_kind::type const& kind,
                                         std::string const& table_name,
                                         std::vector<std::string> const& column_names,
                                         std::vector<cudf::data_type> const& column_types);

}  // namespace gqe::storage

template <>
struct std::hash<gqe::storage::descriptor> {
  std::size_t operator()(gqe::storage::descriptor const& desc) const
  {
    return std::visit([](auto const& d) { return d.hash(); }, desc);
  }
};
