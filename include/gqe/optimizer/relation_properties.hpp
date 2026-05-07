/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/types.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gqe {
namespace optimizer {

namespace column_property {
// Currently only using `unique`
enum class property_id { compressed, sorted, unique };

std::string to_string(column_property::property_id prop);
}  // namespace column_property

/**
 * @brief A class to keep track of properties that each column in a relation exhibits.
 *
 * @note The class is designed to answer the questions
 * 1. Given the property I'm interested in, which columns have this property?
 * 2. Given the column I'm interested in, which properties does the column exhibit?
 * 3. Does column i has property x?
 */
// TODO: Track compound keys. These occur in DISTINCT A, B and GROUP BY A, B, but also in the
// schema as PRIMARY KEY (A, B) and UNIQUE (A, B). If has a single group by key, mark the
// group-by key as unique
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

  bool operator==(const relation_properties& other) const noexcept
  {
    return _columns_properties == other._columns_properties;
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
};

}  // namespace optimizer
}  // namespace gqe
