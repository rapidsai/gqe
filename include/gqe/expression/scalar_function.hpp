/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/logical/utility.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <stdexcept>

namespace gqe {

/**
 * @brief Scalar function expressions are created from Substrait ScalarFunction expressions
 *
 * @note The consumer will check whether the function name can be mapped to a
 * `gqe::scalar_function_expression::function_kind`. If so, it will translate the Substrait
 * `ScalarFunction` expression into `gqe::scalar_function_expression`. Otherwise, it will check if
 * the Substrait `ScalarFunction` can be mapped to a cudf binary or unary op. If so, it will be
 * parsed into `gqe::binary_op_expression`s or `gqe::unary_op_expression`s (`TODO`. Tracked in
 * https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/60). If not, the consumer will
 * throw an exception.
 */
class scalar_function_expression : public expression {
 public:
  enum class function_kind {
    // date_part(input, component): return the `component` part of timestamp `input`. For example
    // date_part(timestamp_D(2023-02-04), "year") returns 2023
    datepart,
    // substr(size_type start, size_type length): return a substring of input starting from the
    // character at the `start` index to `start+length-1`
    substr,
    // round(input, decimal_places): round the input to the specified number of decimal places
    round
  };

  /**
   * @brief Construct a new scalar function expression object
   *
   * @param fn_kind Scalar function kind for this expression
   * @param arguments List of argument expressions
   */
  scalar_function_expression(function_kind fn_kind,
                             std::vector<std::shared_ptr<expression>> arguments)
    : expression(std::move(arguments)), _fn_kind(fn_kind)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept final
  {
    return expression_type::scalar_function;
  }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @brief Return the scalar function kind of this expression
   */
  [[nodiscard]] function_kind fn_kind() const { return _fn_kind; }

 private:
  function_kind _fn_kind;
};

class datepart_expression : public scalar_function_expression {
 public:
  enum class datetime_component {
    year,
    month,
    day,
    weekday,
    hour,
    minute,
    second,
    millisecond,
    nanosecond
  };

  /**
   * @brief Construct a new datepart expression object
   *
   * @param input Expression to extract date part from
   * @param component Type of date part to extract
   */
  datepart_expression(std::shared_ptr<expression> input, datetime_component component)
    : scalar_function_expression(function_kind::datepart, {std::move(input)}), _component(component)
  {
  }

  /**
   * @brief Returns the component type of the input timestamp to extract
   */
  [[nodiscard]] datetime_component component() const noexcept { return _component; }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final
  {
    // All `extract_<datatime_component>()` functions in cudf (v23.06) return INT16
    return cudf::data_type(cudf::type_id::INT16);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final;

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<datepart_expression>(*this);
  }

 private:
  datetime_component _component;
};

class round_expression : public scalar_function_expression {
 public:
  /**
   * @brief Construct a new round expression object
   *
   * @param input Expression to round
   * @param decimal_places Number of decimal places to round the `input` to
   */
  round_expression(std::shared_ptr<expression> input, cudf::size_type decimal_places)
    : scalar_function_expression(function_kind::round, {std::move(input)}),
      _decimal_places(decimal_places)
  {
  }

  /**
   * @brief Returns number of decimal places to round the input to
   */
  [[nodiscard]] cudf::size_type decimal_places() const noexcept { return _decimal_places; }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final
  {
    auto const child_exprs = children();
    assert(child_exprs.size() == 1);
    return child_exprs[0]->data_type(column_types);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 1);
    return "round(" + child_exprs[0]->to_string() + ", " + std::to_string(_decimal_places) + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<round_expression>(*this);
  }

 private:
  cudf::size_type _decimal_places;
};

class substr_expression : public scalar_function_expression {
 public:
  /**
   * @brief Construct a new substr expression object
   *
   * @param input Expression to extract the susbtraing from
   * @param start Start index in `input` as the first character of the result substring
   * @param length The length of the result substring
   */
  substr_expression(std::shared_ptr<expression> input,
                    cudf::size_type start,
                    cudf::size_type length)
    : scalar_function_expression(function_kind::substr, {std::move(input)}),
      _start(start),
      _length(length)
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final
  {
    return cudf::data_type(cudf::type_id::STRING);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 1);
    return "substr(" + child_exprs[0]->to_string() + ", " + std::to_string(_start) + ", " +
           std::to_string(_length) + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<substr_expression>(*this);
  }

  /**
   * @brief Returns the start index of the substring
   */
  [[nodiscard]] cudf::size_type start() const noexcept { return _start; }

  /**
   * @brief Returns the length of the substring
   */
  [[nodiscard]] cudf::size_type length() const noexcept { return _length; }

 private:
  cudf::size_type _start;
  cudf::size_type _length;
};

}  // namespace gqe