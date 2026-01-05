/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
class subquery_expression : public expression {
 public:
  enum class subquery_type_type { in_predicate, scalar, set_predicate, set_comparison };

  subquery_expression(std::vector<std::shared_ptr<expression>> child_expressions,
                      cudf::size_type relation_index)
    : expression(std::move(child_expressions)), _relation_index(relation_index)
  {
  }

  /**
   * @copydoc gqe::expression::type()
   */
  [[nodiscard]] expression_type type() const noexcept final { return expression_type::subquery; }

  /**
   * @brief Return the type of the subquery
   */
  [[nodiscard]] virtual subquery_type_type subquery_type() const noexcept = 0;

  /**
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @brief Return the index to input relation stored in the associated relation
   */
  [[nodiscard]] std::size_t relation_index() const noexcept { return _relation_index; }

  /**
   * @copydoc expression::operator==(const relation& other)
   */
  bool operator==(const expression& other) const override;

 private:
  std::size_t _relation_index;
};

class in_predicate_expression : public subquery_expression {
 public:
  /**
   * @brief Construct a new in predicate expression object
   *
   * The in_predicate_expression checks that the needles expression(s) is contained in the
   * haystack subquery. The haystack can be accessed in the `_subqueries` field of the relation
   * associated with this expression.
   *
   * Examples:
   * x IN (SELECT * FROM t)
   * (x, y) IN (SELECT a, b FROM t)
   *
   * @param needles Expressions who existence will be checked
   * @param haystack_relation_index Index to the subquery to check in the associated relation's
   * `_subqueries` field
   */
  in_predicate_expression(std::vector<std::shared_ptr<expression>> needles,
                          cudf::size_type haystack_relation_index)
    : subquery_expression(std::move(needles), haystack_relation_index)
  {
  }

  /**
   * @copydoc gqe::subquery_expression::subtype()
   */
  [[nodiscard]] subquery_type_type subquery_type() const noexcept override
  {
    return subquery_type_type::in_predicate;
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
    auto child_exprs             = children();
    std::string in_predicate_str = "(";
    for (auto child_expr : child_exprs) {
      in_predicate_str += child_expr->to_string();
    }
    in_predicate_str += ") IN parent's subquery indexed " + std::to_string(relation_index());
    return in_predicate_str;
  }

  /**
   * @copydoc gqe::expression::clone()
   */
  [[nodiscard]] std::unique_ptr<expression> clone() const override
  {
    return std::make_unique<in_predicate_expression>(*this);
  }
};
}  // namespace gqe
