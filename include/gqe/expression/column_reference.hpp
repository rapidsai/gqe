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

#include <cudf/types.hpp>
#include <string>

namespace gqe {

/**
 * @brief An expression referencing a column of the associated table.
 */
class column_reference_expression : public expression {
 public:
  /**
   * @brief Construct a column reference expression.
   *
   * @param[in] column_idx Column index of the referenced column (zero-based).
   */
  column_reference_expression(cudf::size_type column_idx) : expression({}), _column_idx(column_idx)
  {
  }

  /**
   * @brief Construct a column reference expression with names.
   *
   * @param[in] column_idx Column index of the referenced column (zero-based).
   * @param[in] column_name Name of the referenced column (can be empty for benchmarking).
   */
  column_reference_expression(cudf::size_type column_idx, std::string column_name)
    : expression({}), _column_idx(column_idx), _column_name(std::move(column_name))
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept override
  {
    return expression_type::column_reference;
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const override
  {
    return column_types[_column_idx];
  }

  /**
   * @brief Return the column index.
   */
  [[nodiscard]] cudf::size_type column_idx() const noexcept { return _column_idx; }

  /**
   * @brief Return the column name.
   */
  [[nodiscard]] std::string const& column_name() const noexcept { return _column_name; }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept override
  {
    if (_column_name.empty()) {
      return "column_reference(" + std::to_string(_column_idx) + ")";
    } else {
      return _column_name;
    }
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<column_reference_expression>(*this);
  }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override
  {
    if (this->type() != other.type()) { return false; }
    auto other_cast_expr = dynamic_cast<const column_reference_expression&>(other);
    if (this->column_idx() != other_cast_expr.column_idx()) { return false; }
    return true;
  }

 private:
  cudf::size_type _column_idx;
  std::string _column_name;
};

}  // namespace gqe
