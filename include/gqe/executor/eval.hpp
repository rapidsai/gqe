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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {

/**
 * @brief Maps a `cudf::{unary|binary}_operator` to its `cudf::ast::ast_operator` equivalent.
 *
 * @param[in] op Input `cudf::{unary|binary}_operator`.
 *
 * @returns The equivalent `cudf::ast::ast_operator`.
 *
 * @throws std::logic_error if there is no equivalent `cudf::ast::ast_operator`.
 */
[[nodiscard]] cudf::ast::ast_operator cudf_to_ast_operator(
  std::variant<cudf::binary_operator, cudf::unary_operator> op);

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
  void visit(literal_expression<numeric::decimal32> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<numeric::decimal64> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<numeric::decimal128> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<std::string> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(literal_expression<T> const*)
   */
  void visit(literal_expression<cudf::timestamp_D> const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(binary_op_expression const*)
   */
  void visit(binary_op_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(if_then_else_expression const*)
   */
  void visit(if_then_else_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(cast_expression const*)
   */
  void visit(cast_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(unary_op_expression const*)
   */
  void visit(unary_op_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(is_null_expression const*)
   */
  void visit(is_null_expression const* expression) override;

  /**
   * @copydoc gqe::expression_visitor::visit(scalar_function_expression const*)
   */
  void visit(scalar_function_expression const* expression) override;

  /**
   * @brief Context infomation required to evaluate an expression.
   */
  struct evaluation_context {
    /**
     * @brief Construct an evaluation context object.
     *
     * @param[in] expr Expression for which the context is to be created.
     */
    evaluation_context(std::unique_ptr<gqe::expression> expr) : gqe_expression{std::move(expr)} {}

    /**
     * @brief Return the string representation of the evaluation context.
     */
    [[nodiscard]] std::string to_string() const noexcept;

    std::shared_ptr<gqe::expression> gqe_expression;  ///< Expression this context is for.
    std::vector<evaluation_context*>
      child_contexts;  ///< References the context objects of each child expression.
    std::optional<std::unique_ptr<cudf::ast::expression>>
      cudf_ast_expression;  ///< The equivalent `cudf::ast` expression, if available.
    std::vector<std::unique_ptr<cudf::ast::expression>>
      cudf_ast_dependencies;  ///< Any expression the `cudf::ast`expression depends on.
    std::optional<std::unique_ptr<cudf::scalar>>
      cudf_scalar;  ///< Stores the equivalent `cudf::scalar`, in case the expression is a
                    ///< `literal_expression`.
    std::optional<cudf::size_type>
      column_idx;  ///< Holds the index of the target column if the expression is either a
                   ///< `column_reference_expression` or is evaluated as a sub-task.
  };

 private:
  /**
   * @brief Flag an expression context to be evaluated as a (sub-)task. If the expression is already
   * a task, return the column index of the result column.
   *
   * @param[in] context Evaluation context of the expression
   *
   * @returns Column offset of the materialized evaluation result.
   */
  cudf::size_type dispatch_task(evaluation_context& context) noexcept;

  /**
   * @brief Flag an expression context to be evaluated as a (sub-)task.
   *
   * @tparam Args Types of forwarded ctor parameters for `evaluation_context`.
   *
   * @param[in] expression Pointer to the input expression (key) to be stored in the context map.
   * @param[in] args Ctor parameters of the evaluation_context (payload) to be emplaced in the
   * context map.
   *
   * @returns A pair consisting of a ref to the stored context object along with a bool indicating
   * if the key has been newly inserted into the map.
   */
  template <typename... Args>
  [[nodiscard]] std::pair<evaluation_context&, bool> emplace_context(expression const* expression,
                                                                     Args&&... args) noexcept;

  /**
   * @brief Fetches the associated `evaluation_context` for a given key expression.
   *
   * @param[in] expression Pointer to the expression (key) to search for.
   *
   * @returns A reference to the stored `evaluation_context` object.
   *
   * @throws If the key expression cannot located inside the map.
   */
  [[nodiscard]] evaluation_context& find_context(expression const* expression);

  /**
   * @brief Helper function to create and store the initial `evaluation_context` for a
   * `literal_expression`.
   *
   * @tparam T Underlying C++ type of the literal.
   *
   * @param[in] expression Input `literal_expression`.
   *
   * @returns A pair consisting of a ref to the stored context object along with a bool indicating
   * if the key has been newly inserted into the map.
   */
  template <typename T>
  void create_literal_context(literal_expression<T> const* expression) noexcept;

  // @TODO remove once we have ast support for fixed_point.
  template <typename Rep, numeric::Radix Rad>
  void create_decimal_literal_context(
    literal_expression<numeric::fixed_point<Rep, Rad>> const* expression) noexcept;

  cudf::table_view const&
    _table;  ///< Non-owning reference to the table on which to evaluate the expression.
  std::vector<cudf::data_type> _column_types;  ///< Column data types of the input table.
  expression const* _root_expression;          ///< Root of the AST to be evaluated
  cudf::size_type _column_reference_offset;    ///< Global column offset.
  std::vector<evaluation_context const*>
    _tasks;                    ///< Sub-expressions which need to be executed in order.
  cudf::size_type _next_task;  ///< Index of the next available output column.

  std::unordered_map<expression const*, evaluation_context>
    _evaluation_contexts;  ///< Storage for the evaluation context of each sub-expression.
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
