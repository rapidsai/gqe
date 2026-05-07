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

#include <cudf/unary.hpp>

#include <memory>

namespace gqe {

/**
 * @brief A unary-operator expression supported by cuDF.
 */
class unary_op_expression : public expression {
 public:
  unary_op_expression(cudf::unary_operator unary_operator, std::shared_ptr<expression> input)
    : expression({std::move(input)}), _unary_operator(unary_operator)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept final { return expression_type::unary_op; }

  /**
   * @brief Return the type of the unary operator.
   */
  [[nodiscard]] cudf::unary_operator unary_operator() const noexcept { return _unary_operator; }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;

 private:
  cudf::unary_operator _unary_operator;
};

// operator !
class not_expression : public unary_op_expression {
 public:
  not_expression(std::shared_ptr<expression> input)
    : unary_op_expression(cudf::unary_operator::NOT, std::move(input))
  {
  }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const final
  {
    return cudf::data_type(cudf::type_id::BOOL8);
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 1);
    return "!(" + child_exprs[0]->to_string() + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<not_expression>(*this);
  }
};

}  // namespace gqe
