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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace optimizer {
class optimization_rule;
}  // namespace optimizer
namespace logical {

class read_relation : public relation {
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new read relation object
   *
   * @param subquery_relations Subquery relations that are referenced within the
   * `partial_filter` expression
   * @param column_names Names of the columns to be loaded
   * @param column_types Types of the columns to be loaded
   * @param table_name Name of the table to be loaded
   * @param partial_filter Expression determining which rows of the input data are to be loaded.
   * Note that this is an optimization hint and the read relation makes no guarantee that this
   * filter will be applied.
   */
  read_relation(std::vector<std::shared_ptr<relation>> subquery_relations,
                std::vector<std::string> column_names,
                std::vector<cudf::data_type> column_types,
                std::string table_name,
                std::unique_ptr<expression> partial_filter);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::read; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    return this->_data_types;  // initialized in constructor
  }
  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the name of the table to read from
   *
   * @return Name of table to read from
   */
  const std::string& table_name() const { return _table_name; }

  /**
   * @brief Return the names of the columns to read
   *
   * @return List of columns to read
   */
  const std::vector<std::string>& column_names() const { return _column_names; }

  /**
   * @brief Return a raw pointer to partial filter
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   *
   * @return Filter hint for read relation
   */
  [[nodiscard]] expression* partial_filter_unsafe() const { return _partial_filter.get(); }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
  std::string _table_name;
  std::unique_ptr<expression> _partial_filter;
};

}  // namespace logical
}  // namespace gqe