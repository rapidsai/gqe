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

#include <cudf/binaryop.hpp>

#include <cassert>
#include <memory>

namespace gqe {

/**
 * @brief A binary-operator expression supported by cuDF.
 */
class binary_op_expression : public expression {
 public:
  /**
   * @brief Construct a binary-operator expression.
   *
   * @param[in] binary_operator Binary operator of the expression.
   * @param[in] lhs Left-hand side of the expression.
   * @param[in] rhs Right-hand side of the expression.
   */
  binary_op_expression(cudf::binary_operator binary_operator,
                       std::shared_ptr<expression> lhs,
                       std::shared_ptr<expression> rhs)
    : expression({std::move(lhs), std::move(rhs)}), _binary_operator(binary_operator)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept final { return expression_type::binary_op; }

  /**
   * @brief Return the type of the binary operator.
   */
  [[nodiscard]] cudf::binary_operator binary_operator() const noexcept { return _binary_operator; }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;

 private:
  cudf::binary_operator _binary_operator;
};

// operator +
class add_expression : public binary_op_expression {
 public:
  add_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::ADD, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final;

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " + " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<add_expression>(*this);
  }
};

// operator -
class subtract_expression : public binary_op_expression {
 public:
  subtract_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::SUB, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final;

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " - " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<subtract_expression>(*this);
  }
};

// operator *
class multiply_expression : public binary_op_expression {
 public:
  multiply_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::MUL, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final;

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " * " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<multiply_expression>(*this);
  }
};

// operator /
class divide_expression : public binary_op_expression {
 public:
  divide_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::TRUE_DIV, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::FLOAT64);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " / " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<divide_expression>(*this);
  }
};

// operator &&
class logical_and_expression : public binary_op_expression {
 public:
  logical_and_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::NULL_LOGICAL_AND, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " && " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<logical_and_expression>(*this);
  }
};

// operator ||
class logical_or_expression : public binary_op_expression {
 public:
  logical_or_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::NULL_LOGICAL_OR, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " || " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<logical_or_expression>(*this);
  }
};

// operator ==
class equal_expression : public binary_op_expression {
 public:
  equal_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::EQUAL, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " = " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<equal_expression>(*this);
  }
};

// Returns true when both operands are null; false when one is null; the result of equality when
// both are non-null
class nulls_equal_expression : public binary_op_expression {
 public:
  nulls_equal_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::NULL_EQUALS, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " is_not_distinct_from " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<nulls_equal_expression>(*this);
  }
};

// operator !=
class not_equal_expression : public binary_op_expression {
 public:
  not_equal_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::NOT_EQUAL, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " != " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<not_equal_expression>(*this);
  }
};

// operator <
class less_expression : public binary_op_expression {
 public:
  less_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::LESS, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " < " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<less_expression>(*this);
  }
};

// operator >
class greater_expression : public binary_op_expression {
 public:
  greater_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::GREATER, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " > " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<greater_expression>(*this);
  }
};

// operator <=
class less_equal_expression : public binary_op_expression {
 public:
  less_equal_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::LESS_EQUAL, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " <= " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<less_equal_expression>(*this);
  }
};

// operator >=
class greater_equal_expression : public binary_op_expression {
 public:
  greater_equal_expression(std::shared_ptr<expression> lhs, std::shared_ptr<expression> rhs)
    : binary_op_expression(cudf::binary_operator::GREATER_EQUAL, std::move(lhs), std::move(rhs))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(std::vector<cudf::data_type> const&) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 2);
    return child_exprs[0]->to_string() + " >= " + child_exprs[1]->to_string();
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<greater_equal_expression>(*this);
  }
};

}  // namespace gqe
