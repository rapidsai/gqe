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
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

class expression_evaluator : public expression_visitor {
 public:
  /**
   * @brief Create an cudf::ast expression evaluator.
   *
   * @param[in] table Table on which to evaluate the expression.
   * @param[in] root_expression Non-owning pointer to the root of the AST in gqe::expression format.
   * @param[in] column_reference_offset Offset for column reference expressions. For example,
   * if this argument is 2, col_ref(3) refers to column 1 (3 - 2).
   *
   */
  expression_evaluator(cudf::table_view const& table,
                       expression const* root_expression,
                       cudf::size_type column_reference_offset = 0);

  /**
   * @brief Evaluate an expression on a table.
   *
   * This function uses cudf's AST module to efficiently evaluate expressions.
   * Since `cudf::ast` only implements a subset of expressions that we need,
   * we evaluate unsupported subexpressions separately using a fallback strategy.
   *
   * @return A pair of [evaluated_result, column_cache] where `evaluated_result` is the result of
   * evaluating `expression` on `table`, and `column_cache` must be kept alive for
   * `evaluated_result` to be valid.
   */
  [[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>> evaluate() const;

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
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<std::string> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(binary_op_expression const*)
   */
  void visit(binary_op_expression const* expression) override;

 private:
  /**
   * @brief Helper function for convertion a variant of `shared_ptr`to a variant of raw pointers.
   *
   * @param expr A variant of `shared_ptr` types.
   *
   * @return A variant of raw pointers.
   */
  [[nodiscard]] std::variant<cudf::ast::expression*, gqe::expression*> to_raw_expr_ptr(
    std::variant<std::shared_ptr<cudf::ast::expression>, std::shared_ptr<gqe::expression>> const&
      expr) const;

  /**
   * @brief Helper function which concatenated the input table with the current
   *        intermediate result columns and returns a new `cudf::table_view` object.
   *
   * @param intermediate_results Additional columns that are concatenated to the input table.
   *
   * @return New table view of the two concatenated tables.
   */
  [[nodiscard]] cudf::table_view concat_input_table(
    std::vector<std::unique_ptr<cudf::column>> const& intermediate_results) const;

  /**
   * @brief Helper function to extract the data type of each column from a table view.
   *
   * @param table Table view.
   *
   * @return Vector of column data types.
   */
  [[nodiscard]] std::vector<cudf::data_type> column_types(cudf::table_view const& table) const;

  cudf::table_view const&
    _table;  ///< Non-owning reference to the table on which to evaluate the expression.
  expression const* _root_expression;        //< Root of the AST to be evaluated
  cudf::size_type _column_reference_offset;  ///< Global column offset.
  std::vector<std::variant<std::shared_ptr<cudf::ast::expression>,
                           std::shared_ptr<gqe::expression>>>
    _converted_expressions;  //< Converted expressions (either in `cudf::ast::expression` or
                             //`gqe::expression` format).
  std::vector<std::unique_ptr<cudf::scalar>> _converted_scalars;  ///< Scalars in cudf::ast format.
  std::vector<std::variant<cudf::ast::expression*, gqe::expression*>>
    _sub_tasks;  // Sub-expressions which need to be executed in order.
  cudf::size_type
    _next_intermediate;  // Counter for intermediate result columns needed during execution.

  static inline const std::unordered_map<cudf::binary_operator, cudf::ast::ast_operator>
    _operator_map = {
      {cudf::binary_operator::ADD, cudf::ast::ast_operator::ADD},
      {cudf::binary_operator::SUB, cudf::ast::ast_operator::SUB},
      {cudf::binary_operator::MUL, cudf::ast::ast_operator::MUL},
      {cudf::binary_operator::TRUE_DIV, cudf::ast::ast_operator::TRUE_DIV},
      {cudf::binary_operator::LOGICAL_AND, cudf::ast::ast_operator::LOGICAL_AND},
      {cudf::binary_operator::LOGICAL_OR, cudf::ast::ast_operator::NULL_LOGICAL_OR},
      {cudf::binary_operator::EQUAL, cudf::ast::ast_operator::EQUAL},
      {cudf::binary_operator::NOT_EQUAL, cudf::ast::ast_operator::NOT_EQUAL},
      {cudf::binary_operator::LESS, cudf::ast::ast_operator::LESS},
      {cudf::binary_operator::GREATER, cudf::ast::ast_operator::GREATER},
      {cudf::binary_operator::LESS_EQUAL, cudf::ast::ast_operator::LESS_EQUAL},
      {cudf::binary_operator::GREATER_EQUAL,
       cudf::ast::ast_operator::GREATER_EQUAL}  // TODO add more operators
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
 * @note This function uses the `cudf::ast` module to evaluate expressions efficiently.
 * Whenever an AST node is not supported by `cudf::ast`, we compute the result of
 * this sub-expression using a custom evaluation strategy.
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
