/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

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
  T value() const noexcept { return _value; }

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
    if constexpr (std::is_convertible_v<T, std::string>) {
      value_string = _value;
    } else {
      value_string = std::to_string(_value);
    }
    return "literal(" + cudf::type_dispatcher(_data_type, cudf::type_to_name{}) + " " +
           value_string + ")";
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
