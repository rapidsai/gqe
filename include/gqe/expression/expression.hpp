/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gqe {

class binary_op_expression;
class cast_expression;
class column_reference_expression;
class if_then_else_expression;
template <typename T>
class literal_expression;
class subquery_expression;

/**
 * @brief Base interface for an expression visitor.
 *
 * A concrete visitor needs to override these methods to customize the behavior.
 */
struct expression_visitor {
  virtual void visit(column_reference_expression const* expression)
  {
    throw std::logic_error("Visiting column_reference_expression is not implemented");
  }

  // The following functions resolve the instantiation problem of the
  // `literal_expression<T>` template for now by explicitly instantiating for
  // numeric types we target in the substrait consumer.
  template <typename T>
  void visit(literal_expression<T> const* expression)
  {
    throw std::logic_error("Visiting a generic literal_expression<T> is not implemented");
  }

  virtual void visit(literal_expression<int32_t> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<int32_t> is not implemented");
  }

  virtual void visit(literal_expression<int64_t> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<int64_t> is not implemented");
  }

  virtual void visit(literal_expression<float> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<float> is not implemented");
  }

  virtual void visit(literal_expression<double> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<double> is not implemented");
  }

  virtual void visit(literal_expression<std::string> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<string> is not implemented");
  }

  virtual void visit(binary_op_expression const* expression)
  {
    throw std::logic_error("Visiting binary_op_expression is not implemented");
  }

  virtual void visit(subquery_expression const* expression)
  {
    throw std::logic_error("Visiting subquery_expression is not implemented");
  }

  virtual void visit(if_then_else_expression const* expression)
  {
    throw std::logic_error("Visiting if_then_else_expression is not implemented");
  }

  virtual void visit(cast_expression const* expression)
  {
    throw std::logic_error("Visiting cast_expression is not implemented");
  }

  virtual void visit(literal_expression<cudf::timestamp_D> const* expression)
  {
    throw std::logic_error("Visiting literal_expression<cudf::timestamp_D> is not implemented");
  }
};

/**
 * @brief Abstract base class of all expression nodes.
 *
 * In GQE, an expression is represented by a DAG. For example, the following tree represents a
 * boolean expression that evaluates to true if column0 is equal to 15 and column1 is greater
 * than 10.
 *
 *             AND
 *           /     \
 *       Equal     Greater
 *       /   \     /      \
 * ColRef0   15  ColRef1   10
 *
 * This `expression` class serves as an abstract base class for all expression nodes.
 *
 * Note that in general an expression is only meaningful if it is associated with a table. For
 * example, a `column_reference_expression` only knows which column it is referred to if a table is
 * provided.
 */
class expression {
 public:
  enum class expression_type {
    // An expression referencing a column
    column_reference,
    // A fixed value in the source code, e.g., a string literal or a number
    literal,
    // Evaluate to true if the input expression is NULL and false otherwise
    is_null,
    // Cast the input expression to a specific type
    cast,
    // A ternary expression that returns the then expression if the predicate is true, and return
    // the else expression otherwise
    if_then_else,
    // A cuDF unary operator
    unary_op,
    // A cuDF binary operator
    binary_op,
    // Subquery
    subquery,
    // Create row ID column for window function
    gen_ident_col
  };

  /**
   * @brief Construct an expression node.
   *
   * @param[in] children Child nodes of the new expression node.
   */
  expression(std::vector<std::shared_ptr<expression>> children) : _children(std::move(children)) {}

  virtual ~expression() = default;

  /**
   * @brief Construct a new expression by deep copying `expr`.
   */
  expression(const expression& expr)
  {
    _children.reserve(expr._children.size());
    for (auto const& child : expr._children)
      _children.push_back(child->clone());
  }

  /**
   * @brief Assign to the current object by deep copying `expr`.
   */
  expression& operator=(const expression& expr)
  {
    if (this != &expr) {
      _children.clear();
      _children.reserve(expr._children.size());
      for (auto const& child : expr._children)
        _children.push_back(child->clone());
    }
    return *this;
  }

  /**
   * @brief Return the operator type of the expression.
   */
  [[nodiscard]] virtual expression_type type() const noexcept = 0;

  /**
   * @brief Return the output data type of this expression.
   *
   * @param[in] column_types Column types of the associated table.
   */
  [[nodiscard]] virtual cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const = 0;

  /**
   * @brief Return the children nodes.
   *
   * @note The returned expressions do not share ownership. This object must be kept alive for the
   * returned expressions to be valid.
   */
  [[nodiscard]] std::vector<expression*> children() const noexcept
  {
    return utility::to_raw_ptrs(_children);
  }

  /**
   * @brief Accept a visitor.
   *
   * Implement the visitor pattern (https://en.wikipedia.org/wiki/Visitor_pattern) through double
   * dispatch.
   */
  virtual void accept(expression_visitor& visitor) const = 0;

  /**
   * @brief Return the string representation of the expression.
   */
  [[nodiscard]] virtual std::string to_string() const noexcept { return "base gqe::expression"; };

  /**
   * @brief Return a deep copy of the current object.
   */
  [[nodiscard]] virtual std::unique_ptr<expression> clone() const = 0;

 private:
  // Child nodes of the current expression
  std::vector<std::shared_ptr<expression>> _children;

  friend class expression_evaluator;
};

}  // namespace gqe
