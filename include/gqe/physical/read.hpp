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

#include <gqe/expression/expression.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/physical/relation.hpp>

#include <string>
#include <vector>

namespace gqe {

namespace physical {

/**
 * @brief Physical relation for loading data from tables.
 */
class read_relation : public relation {
 public:
  /**
   * @brief Construct a new physical read relation.
   * @param[in] subquery_relations Subquery relations that are referenced within the
   * `partial_filter` expression.
   * @param[in] column_names Names of the columns to be loaded.
   * @param[in] table_name Name of the table to be loaded.
   * @param[in] partial_filter Expression determining which rows of the input data are to be loaded.
   * Note that this is an optimization hint and the read relation makes no guarantee that this
   * filter will be applied.
   * @param[in] data_types the data types for each column
   */
  read_relation(std::vector<std::shared_ptr<relation>> subquery_relations,
                std::vector<std::string> column_names,
                std::string table_name,
                std::unique_ptr<expression> partial_filter,
                std::vector<cudf::data_type> data_types)
    : relation({}, std::move(subquery_relations)),
      _column_names(std::move(column_names)),
      _table_name(std::move(table_name)),
      _partial_filter(std::move(partial_filter)),
      _data_types(std::move(data_types))
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the name of the table to load.
   */
  [[nodiscard]] std::string table_name() const noexcept { return _table_name; }

  /**
   * @brief Return the names of the columns to load.
   */
  [[nodiscard]] std::vector<std::string> column_names() const noexcept { return _column_names; }

  expression* partial_filter_unsafe() const { return _partial_filter.get(); }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override
  {
    return _data_types;
  }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  std::vector<std::string> _column_names;
  std::string _table_name;
  std::unique_ptr<expression> _partial_filter;
  std::vector<cudf::data_type> _data_types;
};

}  // namespace physical
}  // namespace gqe
