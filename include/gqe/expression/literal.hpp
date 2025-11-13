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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <ctime>
#include <sstream>
#include <string>
#include <typeinfo>

namespace gqe {

/**
 * @brief An expression representing a fixed value in the query.
 *
 * A literal can be of different primitive and variable-size types.
 * Example, "SELECT col1 FROM a WHERE a.col2 = 'foo' AND a.col3 < 10;"
 *   - 'foo' is a string literal
 *   - 10 is an integer literal
 */
template <typename T>
class literal_expression : public gqe::expression {
 public:
  /**
   * @brief Constructs a literal expression.
   *
   * @param[in] type Type of the literal expression
   */
  literal_expression(T value, bool is_null = false)
    : expression({}), _data_type(_cudf_type_from_cpp_type(value)), _value(value), _is_null(is_null)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept override { return expression_type::literal; }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& input_types) const override
  {
    return _data_type;
  }

  /**
   * @brief Return stored value for this literal expression
   *
   * @return The value of this expression
   */
  [[nodiscard]] T value() const noexcept { return _value; }

  /**
   * @brief Return whether this literal is null
   *
   * @return true if this literal is null
   * @return false otherwise
   */
  [[nodiscard]] bool is_null() const noexcept { return _is_null; }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept override
  {
    std::string value_string;
    if (_is_null) {
      value_string = "NULL";
    } else if constexpr (std::is_convertible_v<T, std::string>) {
      value_string = _value;
    } else if constexpr (std::is_convertible_v<T, cudf::timestamp_D>) {
      std::time_t time_since_epoch = cuda::std::chrono::system_clock::to_time_t(_value);
      // Use POSIX localtime_r because C++11 std::localtime may not be thread-safe.
      // C++20 provides std::format as a more concise alternative.
      std::tm tm{};
      localtime_r(&time_since_epoch, &tm);
      std::stringstream ss;
      ss << std::put_time(&tm, "%Y-%m-%d");
      value_string = ss.str();
    } else {
      value_string = std::to_string(_value);
    }
    return "literal(" + cudf::type_to_name(_data_type) + " " + value_string + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<literal_expression>(*this);
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
    if (this->_data_type != other.data_type({})) { return false; }
    auto other_literal_expr = dynamic_cast<const literal_expression<T>&>(other);
    if (this->value() != other_literal_expr.value()) { return false; }
    return true;
  }

 private:
  /**
   * @brief Return the corresponding cuDF type from a C++ type
   *
   * @param value Literal value with C++ type
   * @return cuDF type of `value`
   */
  static inline cudf::data_type _cudf_type_from_cpp_type(T value)
  {
    if constexpr (std::is_convertible_v<T, std::string>) {
      return cudf::data_type(cudf::type_id::STRING);
    } else if constexpr (cudf::is_fixed_point<T>()) {
      return cudf::data_type(cudf::type_to_id<T>(), value.scale());
    } else {
      return cudf::data_type(cudf::type_to_id<T>());
    }
  }

  cudf::data_type _data_type;
  T _value;
  bool _is_null;
};

}  // namespace gqe
