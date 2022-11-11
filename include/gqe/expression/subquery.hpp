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

#include <cstddef>
#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

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
   * @copydoc gqe::expression::accept()
   */
  void accept(expression_visitor& visitor) const override { visitor.visit(this); }

  /**
   * @brief Return the index to input relation stored in the associated relation
   */
  [[nodiscard]] std::size_t relation_index() const noexcept { return _relation_index; }

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
