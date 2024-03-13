/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cxx_gqe/types.hpp>

#include <gqe/catalog.hpp>

#include <cudf/types.hpp>

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
