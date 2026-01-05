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

#pragma once

#include <gqe/catalog.hpp>

#include <cudf/types.hpp>

#include <cxx_gqe/types.hpp>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <memory>
#include <string>
#include <vector>

namespace cxx_gqe {

/*
 * @brief Catalog wrapper
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * catalog.
 */
class catalog {
  friend std::unique_ptr<catalog> new_catalog() noexcept;

 public:
  /**
   * @brief Register a new table into the catalog.
   *
   * @throw std::logic_error if `table_name` is already registered.
   *
   * @param[in] table_name Name of the table to create.
   * @param[in] columns A collection of (column name, column data type) pairs.
   * @param[in] storage_type A storage hint about how to physically store the
   * table's data.
   * @param[in] storage_info The struct belonging to the `storage_type` enum.
   * Combined, these parameters are used to construct a
   * `gqe::storage_kind::type` variant.
   * @param[in] partitioning_schema_type The partitioning schema with which the
   * table's data are divided.
   * @param[in] partitioning_schema_info The struct belonging to the
   * `partitioning_schema_type` enum. Combined, these parameters are used to
   * construct a `gqe::partitioning_schema_kind::type` variant.
   */
  void register_table(const rust::Str table_name,
                      const rust::Slice<const column_schema> columns,
                      const storage_kind_type storage_type,
                      const void* storage_info,
                      const partitioning_schema_kind_type partitioning_schema_type,
                      const void* partitioning_schema_info);

  /*
   * @brief Returns the C++ catalog.
   *
   * This is a helper method used in the C++ bindings to convert from the wrapper to the actual
   * object.
   */
  [[nodiscard]] inline gqe::catalog& get() { return _catalog; }

  /*
   * @brief Returns the C++ catalog as `const`.
   *
   * This is a helper method used in the C++ bindings to convert from the wrapper to the actual
   * object.
   */
  [[nodiscard]] inline const gqe::catalog& get_const() const { return _catalog; }

  /**
   * @brief Return the names of all columns in the table in the user-defined order.
   *
   * @throw std::logic_error if the table is not found in the catalog.
   */
  [[nodiscard]] std::unique_ptr<std::vector<std::string>> column_names(
    const rust::Str table_name) const;

  /**
   * @brief Return the data type of a column in the catalog.
   *
   * @throw std::logic_error if the table or the column is not found in the catalog.
   *
   * @param[in] table_name Table name of the query column.
   * @param[in] column_name Column name of the query column.
   */
  [[nodiscard]] type_id column_type(const rust::Str table_name, const rust::Str column_name) const;

 private:
  gqe::catalog _catalog;
};

/*
 * @brief Returns a new catalog wrapper.
 */
std::unique_ptr<catalog> new_catalog() noexcept;

}  // namespace cxx_gqe
