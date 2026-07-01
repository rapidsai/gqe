/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cudf/types.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gqe {
namespace optimizer {

namespace column_property {
enum class property_id { compressed, sorted };

std::string to_string(column_property::property_id prop);
}  // namespace column_property

/**
 * @brief Tracks per-column and per-key-set properties for a relation's output schema.
 *
 * Two orthogonal API families:
 *  - Per-column properties (compressed, sorted): `add/remove/check_column_property`,
 *    `get_column_properties`, `get_columns_with_property`.
 *  - Unique key-sets (single-column and composite): `add/remove_unique_key`,
 *    `covers_unique_key`, `unique_keys`.
 */
class relation_properties {
 public:
  /**
   * @brief Construct a new relation properties object
   *
   */
  relation_properties() {}

  /**
   * @brief Add `property` to the list of properties column `col_idx` exhibits
   *
   * @param col_idx Index of the column to add the property to
   * @param property Property to add
   */
  void add_column_property(cudf::size_type col_idx, column_property::property_id property) noexcept;

  /**
   * @brief Remove `property` from the list of column `col_idx`'s properties
   *
   * @param col_idx Index of the column to remove the property from
   * @param property Property to remove
   */
  void remove_column_property(cudf::size_type col_idx,
                              column_property::property_id property) noexcept;

  /**
   * @brief Get all properties column `col_idx` exhibits
   *
   * @param col_idx Column to retrieve properties from
   * @return List of properties for the specified column
   */
  std::vector<column_property::property_id> get_column_properties(
    cudf::size_type col_idx) const noexcept;

  /**
   * @brief Get the list of all of the columns that exhibit the specified `property`
   *
   * @param property Property to search
   * @return Set of columns that match
   */
  std::unordered_set<cudf::size_type> get_columns_with_property(
    column_property::property_id property) const noexcept;

  /**
   * @brief Check if the column `col_idx` exhibits the specified `property`
   *
   * @param col_idx Column to check
   * @param property Property to check
   * @return true If the column `col_idx` exhibits `property`
   * @return false Otherwise
   */
  bool check_column_property(cudf::size_type col_idx,
                             column_property::property_id property) const noexcept;

  /**
   * @brief Add a unique key-set. Each element of `key` is a column index into this relation's
   * output schema. Size 1 = single-column unique; size >= 2 = composite unique.
   * Empty keys are ignored; duplicate key-sets are skipped (idempotent).
   */
  void add_unique_key(std::vector<cudf::size_type> key);

  /**
   * @brief Remove the unique key-set that exactly matches `key` (after sorting).
   * No-op if not found.
   */
  void remove_unique_key(std::vector<cudf::size_type> const& key) noexcept;

  /**
   * @brief Return true if `cols` covers at least one registered unique key-set, i.e. some
   * registered key-set is a subset of `cols`.
   *
   * A superset of a unique key is itself unique, so any grouping that includes all columns of a
   * unique key-set is also unique. Used by the `aggregate_perfect_hash` rule to decide whether
   * a set of group-by columns yields unique tuples.
   */
  bool covers_unique_key(std::unordered_set<cudf::size_type> const& cols) const noexcept;

  /**
   * @brief Return all registered unique key-sets (single-column and composite).
   */
  std::vector<std::vector<cudf::size_type>> const& unique_keys() const noexcept
  {
    return _unique_keys;
  }

  bool operator==(const relation_properties& other) const noexcept
  {
    return _columns_properties == other._columns_properties && _unique_keys == other._unique_keys;
  }

  std::string to_string() const;

  std::unordered_map<column_property::property_id, std::unordered_set<cudf::size_type>> const&
  columns_properties() const noexcept
  {
    return _columns_properties;
  }

 private:
  std::unordered_map<column_property::property_id, std::unordered_set<cudf::size_type>>
    _columns_properties;
  std::vector<std::vector<cudf::size_type>> _unique_keys;  ///< composite unique key-sets
};

}  // namespace optimizer
}  // namespace gqe
