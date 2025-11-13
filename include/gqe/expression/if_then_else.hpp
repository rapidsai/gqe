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

namespace gqe {

class if_then_else_expression : public expression {
 public:
  /**
   * @brief Construct a new if-then-else expression object
   *
   * @param if_expr The condition expression to determine whether the `then` or `else` branch should
   * be taken
   * @param then_expr The expression to be evaluated if `if_expr` condition returns true
   * @param else_expr The expression to be evaluated if `if_expr` condition returns false
   */
  if_then_else_expression(std::shared_ptr<expression> if_expr,
                          std::shared_ptr<expression> then_expr,
                          std::shared_ptr<expression> else_expr)
    : expression({std::move(if_expr), std::move(then_expr), std::move(else_expr)})
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept final
  {
    return expression_type::if_then_else;
  }

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& input_types) const final
  {
    auto const then_type                  = children()[1]->data_type(input_types);
    [[maybe_unused]] auto const else_type = children()[2]->data_type(input_types);
    assert(then_type == else_type);
    return then_type;
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept final
  {
    auto child_exprs             = children();
    std::string if_then_else_str = "";
    if_then_else_str += "IF (" + child_exprs[0]->to_string() + ") ";
    if_then_else_str += "THEN (" + child_exprs[1]->to_string() + ") ";
    if_then_else_str += "ELSE (" + child_exprs[2]->to_string() + ")";
    return if_then_else_str;
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<if_then_else_expression>(*this);
  }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;
};

}  // namespace gqe
