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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {

class expression_evaluator : public expression_visitor {
 public:
  /**
   * @brief Create an cudf::ast expression evaluator.
   *
   * @param[in] root_expression Non-owning pointer to the root of the AST in gqe::expression format.
   * @param[in] column_reference_offset Offset for column reference expressions. For example,
   * if this argument is 2, col_ref(3) refers to column 1 (3 - 2).
   *
   */
  expression_evaluator(expression const* root_expression,
                       cudf::size_type column_reference_offset = 0);

  /**
   * @brief Evaluate an expression on a table using the cudf::ast module.
   *
   * @param[in] table Table on which to evaluate the expression.
   *
   * @return A pair of [evaluated_result, column_cache] where `evaluated_result` is the result of
   * evaluating `expression` on `table`, and `column_cache` must be kept alive for
   * `evaluated_result` to be valid.
   */
  [[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>> evaluate(
    cudf::table_view const& table) const;

  /**
   * @copydoc gqe::expression_visitor::visit(column_reference_expression const*)
   */
  void visit(column_reference_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<int32_t> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<int64_t> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<float> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<double> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(binary_op_expression const*)
   */
  void visit(binary_op_expression const* expression) override;

 private:
  cudf::size_type _column_reference_offset;  ///< Global column offset.
  std::vector<std::unique_ptr<cudf::ast::expression const>>
    _converted_expressions;  ///< AST nodes in cudf::ast format.
  std::vector<std::unique_ptr<cudf::scalar>> _converted_scalars;  ///< Scalars in cudf::ast format.

  static inline const std::unordered_map<cudf::binary_operator, cudf::ast::ast_operator>
    _operator_map = {
      {cudf::binary_operator::EQUAL, cudf::ast::ast_operator::EQUAL},
      {cudf::binary_operator::LOGICAL_AND,
       cudf::ast::ast_operator::LOGICAL_AND}  // TODO add more operators
    };  ///> Emum mapper between cudf::binary_op and cudf::ast::ast_operator.
};

/**
 * @brief Evaluate a batch of expressions on a table.
 *
 * @param[in] table Table on which to evaluate the expressions.
 * @param[in] exprs Expressions to be evaluated.
 * @param[in] column_reference_offset Offset for column reference expressions. For example,
 * if this argument is 2, col_ref(3) refers to column 1 (3 - 2).
 *
 * @return A pair of [evaluated_results, column_cache] where `evaluated_results` are the results of
 * evaluating `exprs` on `table`, and `column_cache` must be kept alive for `evaluated_results` to
 * be valid.
 */
std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset = 0);

}  // namespace gqe
