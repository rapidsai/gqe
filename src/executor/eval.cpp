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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace gqe {

expression_evaluator::expression_evaluator(cudf::table_view const& table,
                                           expression const* root_expression,
                                           cudf::size_type column_reference_offset)
  : _table{table},
    _root_expression{root_expression},
    _column_reference_offset{column_reference_offset},
    _next_intermediate{table.num_columns()}
{
  // traverse AST recursively starting at the root node
  root_expression->accept(*this);

  // mark the root expression as to be executed as well
  _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions.back()));
}

[[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>>
expression_evaluator::evaluate() const
{
  std::vector<std::unique_ptr<cudf::column>> intermediate_results;
  intermediate_results.reserve(_sub_tasks.size());

  auto cudf_scalar_cast = [](gqe::expression* expr) -> std::unique_ptr<cudf::scalar> {
    if (auto lit = dynamic_cast<gqe::literal_expression<std::string>*>(expr)) {
      return std::make_unique<cudf::string_scalar>(lit->value(), !lit->is_null());
    } else if (auto lit = dynamic_cast<gqe::literal_expression<int32_t>*>(expr)) {
      return std::make_unique<cudf::numeric_scalar<int32_t>>(lit->value(), !lit->is_null());
    } else if (auto lit = dynamic_cast<gqe::literal_expression<int64_t>*>(expr)) {
      return std::make_unique<cudf::numeric_scalar<int64_t>>(lit->value(), !lit->is_null());
    } else if (auto lit = dynamic_cast<gqe::literal_expression<float>*>(expr)) {
      return std::make_unique<cudf::numeric_scalar<float>>(lit->value(), !lit->is_null());
    } else if (auto lit = dynamic_cast<gqe::literal_expression<double>*>(expr)) {
      return std::make_unique<cudf::numeric_scalar<double>>(lit->value(), !lit->is_null());
    } else {
      throw std::logic_error("Invalid `gqe::literal_expression`");
    }
  };

  for (auto task : _sub_tasks) {
    // input table and associated column types the expression should be evaluated on
    auto table = concat_input_table(intermediate_results);
    auto types = column_types(table);

    if (std::holds_alternative<cudf::ast::expression*>(task)) {
      // evaluate the the `cudf::ast` (sub-)expression using `cudf::compute_column`
      auto result = cudf::compute_column(table, *std::get<cudf::ast::expression*>(task));
      intermediate_results.push_back(std::move(result));
    } else if (std::holds_alternative<gqe::expression*>(task)) {
      auto task_ptr = std::get<gqe::expression*>(task);
      auto children = task_ptr->children();

      switch (task_ptr->type()) {
        case gqe::expression::expression_type::binary_op: {
          auto binop   = dynamic_cast<gqe::binary_op_expression*>(task_ptr);
          auto lhs_ref = dynamic_cast<gqe::column_reference_expression*>(children[0]);
          auto rhs_ref = dynamic_cast<gqe::column_reference_expression*>(children[1]);

          // if the expression is a binary op we can evaluate it using `cudf::binary_operation`
          // children can only be of type column_reference or literal -> 4 combinations
          if (lhs_ref && rhs_ref) {
            auto result = cudf::binary_operation(table.column(lhs_ref->column_idx()),
                                                 table.column(rhs_ref->column_idx()),
                                                 binop->binary_operator(),
                                                 binop->data_type(types));
            intermediate_results.push_back(std::move(result));
          }

          if (lhs_ref && !rhs_ref) {
            auto rhs_scalar = cudf_scalar_cast(children[1]);
            auto result     = cudf::binary_operation(table.column(lhs_ref->column_idx()),
                                                 *(rhs_scalar.get()),
                                                 binop->binary_operator(),
                                                 binop->data_type(types));
            intermediate_results.push_back(std::move(result));
          }

          if (!lhs_ref && rhs_ref) {
            auto lhs_scalar = cudf_scalar_cast(children[0]);
            auto result     = cudf::binary_operation(*(lhs_scalar.get()),
                                                 table.column(rhs_ref->column_idx()),
                                                 binop->binary_operator(),
                                                 binop->data_type(types));
            intermediate_results.push_back(std::move(result));
          }

          if (!lhs_ref && !rhs_ref) { throw std::logic_error("Not implemented"); }
          break;
        }
        case gqe::expression::expression_type::if_then_else: {
          auto if_ref   = dynamic_cast<gqe::column_reference_expression*>(children[0]);
          auto then_ref = dynamic_cast<gqe::column_reference_expression*>(children[1]);
          auto else_ref = dynamic_cast<gqe::column_reference_expression*>(children[2]);

          auto result = cudf::copy_if_else(table.column(then_ref->column_idx()),
                                           table.column(else_ref->column_idx()),
                                           table.column(if_ref->column_idx()));
          intermediate_results.push_back(std::move(result));
          break;
        }
        case gqe::expression::expression_type::literal:
          intermediate_results.push_back(
            cudf::make_column_from_scalar(*cudf_scalar_cast(task_ptr), table.num_rows()));
          break;
        case gqe::expression::expression_type::cast: {
          auto const child_ref =
            dynamic_cast<gqe::column_reference_expression*>(children[0])->column_idx();
          auto const out_type = dynamic_cast<gqe::cast_expression*>(task_ptr)->out_type();

          intermediate_results.push_back(cudf::cast(table.column(child_ref), out_type));
          break;
        }
        default:
          throw std::logic_error("Cannot evaluate expression in the fallback path: " +
                                 task_ptr->to_string());
      }
    } else {
      throw std::bad_variant_access();
    }
  }

  // cuDF's AST module might evaluate expression to a different type than GQE expression's data
  // type. For example, the AST module might evaluate an expression to int32_t, while GQE promotes
  // the result to int64_t. In such case, we cast the result to GQE expression's data type.
  auto const expected_type = _root_expression->data_type(column_types(_table));
  if (intermediate_results.back()->type() != expected_type) {
    intermediate_results.push_back(cudf::cast(intermediate_results.back()->view(), expected_type));
  }

  return std::make_pair(intermediate_results.back()->view(),
                        std::move(intermediate_results.back()));
}

void expression_evaluator::visit(column_reference_expression const* expression)
{
  _converted_expressions.emplace_back(std::make_shared<cudf::ast::column_reference>(
    expression->column_idx() - _column_reference_offset));
}

void expression_evaluator::visit(literal_expression<int32_t> const* expression)
{
  // `expression` and `scalar` objects must be kept alive for the lifetime of
  // the evaluator so we store them in member vectors
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<int32_t>>(expression->value(), !expression->is_null()));
  auto const value = dynamic_cast<cudf::numeric_scalar<int32_t>*>(_converted_scalars.back().get());
  _converted_expressions.emplace_back(std::make_shared<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<int64_t> const* expression)
{
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<int64_t>>(expression->value(), !expression->is_null()));
  auto const value = dynamic_cast<cudf::numeric_scalar<int64_t>*>(_converted_scalars.back().get());
  _converted_expressions.emplace_back(std::make_shared<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<float> const* expression)
{
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<float>>(expression->value(), !expression->is_null()));
  auto const value = dynamic_cast<cudf::numeric_scalar<float>*>(_converted_scalars.back().get());
  _converted_expressions.emplace_back(std::make_shared<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<double> const* expression)
{
  _converted_scalars.push_back(
    std::make_unique<cudf::numeric_scalar<double>>(expression->value(), !expression->is_null()));
  auto const value = dynamic_cast<cudf::numeric_scalar<double>*>(_converted_scalars.back().get());
  _converted_expressions.emplace_back(std::make_shared<cudf::ast::literal>(*(value)));
}

void expression_evaluator::visit(literal_expression<std::string> const* expression)
{
  _converted_expressions.emplace_back(expression->clone());
}

void expression_evaluator::visit(binary_op_expression const* expression)
{
  // compute AST for the lhs child
  expression->_children[0]->accept(*this);
  // child has been created at the back of `_converted_expressions`
  auto lhs_child_index = _converted_expressions.size() - 1;

  // same for rhs child
  expression->_children[1]->accept(*this);
  auto rhs_child_index = _converted_expressions.size() - 1;

  // does lhs child use fallback eval strategy?
  bool const lhs_needs_fallback = std::holds_alternative<std::shared_ptr<gqe::expression>>(
    _converted_expressions[lhs_child_index]);
  // does rhs child use fallback eval strategy?
  bool const rhs_needs_fallback = std::holds_alternative<std::shared_ptr<gqe::expression>>(
    _converted_expressions[rhs_child_index]);

  // lhs fallback and rhs fallback -> emit gqe expr
  if (lhs_needs_fallback || rhs_needs_fallback ||
      // `cudf::ast` cannot handle non-fixed width output columns
      !cudf::is_fixed_width(expression->data_type(column_types(_table))) ||
      // `cudf::ast` cannot handle expressions with different types
      expression->_children[0]->data_type(column_types(_table)) !=
        expression->_children[1]->data_type(column_types(_table))) {
    if (!lhs_needs_fallback) {
      auto lhs_child =
        std::get<cudf::ast::expression*>(to_raw_expr_ptr(_converted_expressions[lhs_child_index]));

      // if the child expression is a `cudf::ast:literal` or `cudf::ast::column_reference`
      // replace it with their `gqe::expression` equivalent
      if (dynamic_cast<cudf::ast::literal*>(lhs_child) ||
          dynamic_cast<cudf::ast::column_reference*>(lhs_child)) {
        _converted_expressions[lhs_child_index] = expression->_children[0];
      } else {  // lhs is a `cudf::ast` sub-expression (needs to be evaluated separately)
        // mark the `cudf::ast` sub-expression as to be evaluated
        _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions[lhs_child_index]));

        // lhs is executed as a separate task; the current expression needs to pick up its result
        _converted_expressions.emplace_back(
          std::make_shared<gqe::column_reference_expression>(_next_intermediate++));

        // index of new child (reference to intermediate result)
        lhs_child_index = _converted_expressions.size() - 1;
      }
    }

    // same as above for rhs child
    if (!rhs_needs_fallback) {
      auto rhs_child =
        std::get<cudf::ast::expression*>(to_raw_expr_ptr(_converted_expressions[rhs_child_index]));

      if (dynamic_cast<cudf::ast::literal*>(rhs_child) or
          dynamic_cast<cudf::ast::column_reference*>(rhs_child)) {
        _converted_expressions[rhs_child_index] = expression->_children[1];
      } else {
        _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions[rhs_child_index]));

        _converted_expressions.emplace_back(
          std::make_shared<gqe::column_reference_expression>(_next_intermediate++));

        rhs_child_index = _converted_expressions.size() - 1;
      }
    }

    // make a copy of the current expression but with the new child expressions
    auto new_expr       = expression->clone();
    new_expr->_children = {
      std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[lhs_child_index]),
      std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[rhs_child_index])};

    _converted_expressions.emplace_back(std::move(new_expr));

    // mark the new expression as to be evaluated
    if (expression != _root_expression) {
      _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions.back()));
      // parent expression picks up the reference to the intermediate result for its child
      _converted_expressions.emplace_back(
        std::make_shared<cudf::ast::column_reference>(_next_intermediate++));
    }
  } else {  // no fallback strategy needed; continue building a `cudf::ast`
    // find the `cudf::ast` equivalent of the given operation type
    auto converted_op = _operator_map.find(expression->binary_operator());
    if (converted_op == _operator_map.end()) {
      throw std::logic_error("Cannot convert " + expression->to_string());
    }

    _converted_expressions.emplace_back(std::make_shared<cudf::ast::operation>(
      converted_op->second,
      *std::get<cudf::ast::expression*>(to_raw_expr_ptr(_converted_expressions[lhs_child_index])),
      *std::get<cudf::ast::expression*>(to_raw_expr_ptr(_converted_expressions[rhs_child_index]))));
  }
}

void expression_evaluator::visit(if_then_else_expression const* expression)
{
  if (!cudf::is_boolean(expression->_children[0]->data_type(column_types(_table)))) {
    throw std::logic_error("Cannot convert " + expression->_children[0]->to_string() +
                           " to boolean");
  }

  if (expression->_children[1]->data_type(column_types(_table)) !=
      expression->_children[2]->data_type(column_types(_table))) {
    throw std::logic_error("Column types of " + expression->_children[1]->to_string() + " and " +
                           expression->_children[2]->to_string() + " must be equal");
  }

  // create task for IF child
  auto if_child_index = create_column_reference(expression->_children[0].get());
  // create task for THEN child
  auto then_child_index = create_column_reference(expression->_children[1].get());
  // create task for ELSE child
  auto else_child_index = create_column_reference(expression->_children[2].get());

  // make a copy of the current expression but with the new child expressions
  auto new_expr       = expression->clone();
  new_expr->_children = {
    std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[if_child_index]),
    std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[then_child_index]),
    std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[else_child_index])};

  _converted_expressions.emplace_back(std::move(new_expr));

  // mark the new expression as to be evaluated
  if (expression != _root_expression) {
    _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions.back()));
    // parent expression picks up the reference to the intermediate result for its child
    _converted_expressions.emplace_back(
      std::make_shared<cudf::ast::column_reference>(_next_intermediate++));
  }
}

void expression_evaluator::visit(cast_expression const* expression)
{
  // FIXME: The current implementation always uses the fallback path to evaluate cast expressions,
  // but cuDF's AST module has cast support for limited types: int64, uint64 and double:
  // https://github.com/rapidsai/cudf/blob/branch-22.12/cpp/include/cudf/ast/expressions.hpp#L137-L139
  // It might be worthwhile to take advantage of that.

  assert(expression->_children.size() == 1);
  auto const child_index = create_column_reference(expression->_children[0].get());

  // make a copy of the current expression but with the new child expressions
  auto new_expr       = expression->clone();
  new_expr->_children = {
    std::get<std::shared_ptr<gqe::expression>>(_converted_expressions[child_index])};

  _converted_expressions.emplace_back(std::move(new_expr));

  // mark the new expression as to be evaluated
  if (expression != _root_expression) {
    _sub_tasks.emplace_back(to_raw_expr_ptr(_converted_expressions.back()));
    // parent expression picks up the reference to the intermediate result for its child
    _converted_expressions.emplace_back(
      std::make_shared<cudf::ast::column_reference>(_next_intermediate++));
  }
}

std::variant<cudf::ast::expression*, gqe::expression*> expression_evaluator::to_raw_expr_ptr(
  std::variant<std::shared_ptr<cudf::ast::expression>, std::shared_ptr<gqe::expression>> const&
    expr) const
{
  if (std::holds_alternative<std::shared_ptr<cudf::ast::expression>>(expr)) {
    return {std::get<std::shared_ptr<cudf::ast::expression>>(expr).get()};
  } else {
    return {std::get<std::shared_ptr<gqe::expression>>(expr).get()};
  }

  throw std::bad_variant_access();
}

cudf::table_view expression_evaluator::concat_input_table(
  std::vector<std::unique_ptr<cudf::column>> const& intermediate_results) const
{
  if (intermediate_results.empty()) {
    return _table;
  } else {
    std::vector<cudf::column_view> intermediate_results_cols;
    std::transform(intermediate_results.begin(),
                   intermediate_results.end(),
                   std::back_inserter(intermediate_results_cols),
                   [](auto& c) { return c->view(); });
    cudf::table_view intermediate_result_table{intermediate_results_cols};
    cudf::table_view combined_table({this->_table, intermediate_result_table});
    return combined_table;
  }
}

std::vector<cudf::data_type> expression_evaluator::column_types(cudf::table_view const& table) const
{
  std::vector<cudf::data_type> column_types;
  column_types.reserve(table.num_columns());
  for (auto const& col : table) {
    column_types.push_back(col.type());
  }
  return column_types;
}

cudf::size_type expression_evaluator::create_column_reference(gqe::expression* expr)
{
  // compute AST for `expr`
  expr->accept(*this);
  // the converted expression has been created at the back of _converted_expressions
  auto converted_expr = to_raw_expr_ptr(_converted_expressions.back());

  // if the current expression is already a `cudf::ast::column_reference`
  // we just convert it to a `gqe::column_reference_expression` and pass it on
  if (std::holds_alternative<cudf::ast::expression*>(converted_expr) &&
      dynamic_cast<cudf::ast::column_reference*>(
        std::get<cudf::ast::expression*>(converted_expr))) {
    _converted_expressions.emplace_back(std::make_shared<gqe::column_reference_expression>(
      dynamic_cast<cudf::ast::column_reference*>(std::get<cudf::ast::expression*>(converted_expr))
        ->get_column_index()));
  } else {
    // push the expression into task queue for evaluation
    _sub_tasks.emplace_back(converted_expr);
    // fetch result of the expression
    _converted_expressions.emplace_back(
      std::make_shared<gqe::column_reference_expression>(_next_intermediate++));
  }
  // The result of `eval` has been created at the back of `_converted_expressions`
  return _converted_expressions.size() - 1;
}

std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset)
{
  std::vector<cudf::column_view> evaluated_results;
  evaluated_results.reserve(exprs.size());
  std::vector<std::unique_ptr<cudf::column>> column_cache;
  column_cache.reserve(exprs.size());

  for (auto expr : exprs) {
    if (expr->type() == expression::expression_type::column_reference) {
      auto const column_idx =
        dynamic_cast<gqe::column_reference_expression const*>(expr)->column_idx();

      if (column_idx < column_reference_offset)
        throw std::out_of_range("Invalid column index and offset combination in expression: " +
                                expr->to_string());

      evaluated_results.push_back(table.column(column_idx - column_reference_offset));
      continue;
    }
    auto evaluator       = expression_evaluator(table, expr, column_reference_offset);
    auto [result, cache] = evaluator.evaluate();

    evaluated_results.push_back(std::move(result));
    column_cache.push_back(std::move(cache));
  }

  return std::make_pair(std::move(evaluated_results), std::move(column_cache));
}

}  // namespace gqe
