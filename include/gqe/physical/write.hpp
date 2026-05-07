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

#include <gqe/physical/relation.hpp>

#include <memory>
#include <vector>

namespace gqe {

namespace physical {

/**
 * @brief Physical relation for storing data to tables.
 */
class write_relation : public relation {
 public:
  /**
   * @brief Construct a new physical write relation.
   *
   * @param[in] input Input table that is written out.
   * @param[in] column_names Names of the columns to be loaded.
   * @param[in] table_name Name of the table to be written to.
   */
  write_relation(std::shared_ptr<relation> input,
                 std::vector<std::string> column_names,
                 std::string table_name)
    : relation({std::move(input)}, {}),
      _column_names(std::move(column_names)),
      _table_name(std::move(table_name))
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the name of the table to store.
   */
  [[nodiscard]] std::string table_name() const noexcept { return _table_name; }

  /**
   * @brief Return the names of the columns to store.
   */
  [[nodiscard]] std::vector<std::string> column_names() const noexcept { return _column_names; }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  std::vector<std::string> _column_names;
  std::string _table_name;
};

}  // namespace physical
}  // namespace gqe
