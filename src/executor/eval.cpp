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

#include <gqe/executor/eval.hpp>

#include <cudf/transform.hpp>

#include <stdexcept>

namespace gqe {

expression_evaluator::expression_evaluator(expression const* root_expression,
                                           cudf::size_type column_reference_offset)
  : _column_reference_offset(column_reference_offset)
{
  // translate the gqe::expression AST into an cudf::ast::expression AST
  // by traversing all the nodes

  // start traversal at root node
  root_expression->accept(*this);
}

[[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>>
expression_evaluator::evaluate(cudf::table_view const& table) const
{
  // evaluate the translate AST using `cudf::compute_column`

  // the last element in `_converted_expressions' is always the root node
  auto result = cudf::compute_column(table, *(_converted_expressions.back().get()));

  return std::make_pair(result->view(), std::move(result));
}
void expression_evaluator::visit(column_reference_expression const* expression)
{
  _converted_expressions.push_back(std::make_unique<cudf::ast::column_reference>(
    expression->column_idx() - _column_reference_offset));
}

void expression_evaluator::visit(literal_expression<int32_t> const* expression)
{
  // `expression` and `scalar` objects must be kept alive for the lifetime of
  // the evaluator so we store them in member vectors
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<int32_t>>(expression->value()));
  auto const value = dynamic_cast<cudf::numeric_scalar<int32_t>*>(_converted_scalars.back().get());
  _converted_expressions.push_back(std::make_unique<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<int64_t> const* expression)
{
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<int64_t>>(expression->value()));
  auto const value = dynamic_cast<cudf::numeric_scalar<int64_t>*>(_converted_scalars.back().get());
  _converted_expressions.push_back(std::make_unique<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<float> const* expression)
{
  _converted_scalars.push_back(std::make_unique<cudf::numeric_scalar<float>>(expression->value()));
  auto const value = dynamic_cast<cudf::numeric_scalar<float>*>(_converted_scalars.back().get());
  _converted_expressions.push_back(std::make_unique<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<double> const* expression)
{
  _converted_scalars.push_back(std::make_unique<cudf::numeric_scalar<double>>(expression->value()));
  auto const value = dynamic_cast<cudf::numeric_scalar<double>*>(_converted_scalars.back().get());
  _converted_expressions.push_back(std::make_unique<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(binary_op_expression const* expression)
{
  auto const op = expression->binary_operator();

  // find the `cudf::ast` equivalent of the given operation type
  auto converted_op = _operator_map.find(op);
  if (converted_op == _operator_map.end()) {
    throw std::logic_error("Cannot convert " + expression->to_string());
  }

  auto const children = expression->children();
  assert(children.size() == 2);

  // compute AST for the lhs child
  children[0]->accept(*this);
  // child has been created at the back of `_converted_expressions`
  auto const lhs_child = _converted_expressions.back().get();

  // same for rhs child
  children[1]->accept(*this);
  auto const rhs_child = _converted_expressions.back().get();

  // create binary op expression
  _converted_expressions.push_back(
    std::make_unique<cudf::ast::operation>(converted_op->second, *lhs_child, *rhs_child));
}

std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset)
{
  // FIXME: Extend this function to support scalar results.
  // The current implementation always evaluates to a cudf::column_view. This is okay for
  // expressions like `col_ref(1) + 10`. However, other expressions like `3 + 5` should be evaluated
  // to a cudf::scalar instead.

  std::vector<cudf::column_view> evaluated_results;
  evaluated_results.reserve(exprs.size());
  std::vector<std::unique_ptr<cudf::column>> column_cache;
  column_cache.reserve(exprs.size());

  for (auto expr : exprs) {
    // FIXME enable output column with variable width
    if (expr->type() == expression::expression_type::column_reference) {
      auto const column_idx =
        dynamic_cast<gqe::column_reference_expression const*>(expr)->column_idx();

      if (column_idx < column_reference_offset)
        throw std::out_of_range("Invalid column index and offset combination in expression: " +
                                expr->to_string());

      evaluated_results.push_back(table.column(column_idx - column_reference_offset));
      continue;
    }
    auto evaluator       = expression_evaluator(expr, column_reference_offset);
    auto [result, cache] = evaluator.evaluate(table);

    evaluated_results.push_back(std::move(result));
    column_cache.push_back(std::move(cache));
  }

  return std::make_pair(std::move(evaluated_results), std::move(column_cache));
}

}  // namespace gqe
