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
#include <cudf/utilities/type_dispatcher.hpp>

namespace gqe {

/**
 * @brief An expression to cast from one data type to another.
 */
class cast_expression : public expression {
 public:
  /**
   * @brief Construct a cast expression.
   *
   * @param[in] input Expression to cast
   * @param[in] out_type Data type to cast to
   */
  cast_expression(std::shared_ptr<expression> input, cudf::data_type out_type)
    : expression({std::move(input)}), _out_type(out_type)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept override { return expression_type::cast; }

  /**
   * @copydoc gqe::expression::data_type(std::vector<cudf::data_type> const&)
   */
  [[nodiscard]] cudf::data_type data_type(
    std::vector<cudf::data_type> const& column_types) const override
  {
    return _out_type;
  }

  /**
   * @copydoc gqe::expression::to_string()
   */
  [[nodiscard]] std::string to_string() const noexcept override
  {
    auto child_exprs = children();
    assert(child_exprs.size() == 1);
    return "cast(" + child_exprs[0]->to_string() + " as " + cudf::type_to_name(_out_type) + ")";
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<cast_expression>(*this);
  }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @brief Return the data type to cast to
   */
  [[nodiscard]] cudf::data_type out_type() const noexcept { return _out_type; }

 private:
  cudf::data_type _out_type;
};

}  // namespace gqe
