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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/like.hpp>
#include <gqe/utility/cuda.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace gqe {

[[nodiscard]] cudf::ast::ast_operator cudf_to_ast_operator(
  std::variant<cudf::binary_operator, cudf::unary_operator> op)
{
  static const std::unordered_map<std::variant<cudf::binary_operator, cudf::unary_operator>,
                                  cudf::ast::ast_operator>
    _operator_map = {
      {cudf::binary_operator::ADD, cudf::ast::ast_operator::ADD},
      {cudf::binary_operator::SUB, cudf::ast::ast_operator::SUB},
      {cudf::binary_operator::MUL, cudf::ast::ast_operator::MUL},
      {cudf::binary_operator::TRUE_DIV, cudf::ast::ast_operator::TRUE_DIV},
      {cudf::binary_operator::NULL_LOGICAL_AND, cudf::ast::ast_operator::NULL_LOGICAL_AND},
      {cudf::binary_operator::NULL_LOGICAL_OR, cudf::ast::ast_operator::NULL_LOGICAL_OR},
      {cudf::binary_operator::EQUAL, cudf::ast::ast_operator::EQUAL},
      {cudf::binary_operator::NOT_EQUAL, cudf::ast::ast_operator::NOT_EQUAL},
      {cudf::binary_operator::LESS, cudf::ast::ast_operator::LESS},
      {cudf::binary_operator::GREATER, cudf::ast::ast_operator::GREATER},
      {cudf::binary_operator::LESS_EQUAL, cudf::ast::ast_operator::LESS_EQUAL},
      {cudf::binary_operator::GREATER_EQUAL, cudf::ast::ast_operator::GREATER_EQUAL},
      {cudf::binary_operator::NULL_EQUALS, cudf::ast::ast_operator::NULL_EQUAL},
      {cudf::unary_operator::NOT,
       cudf::ast::ast_operator::NOT}};  ///> Emum mapper between cudf::{unary|binary}_op and
                                        /// cudf::ast::ast_operator.

  auto search = _operator_map.find(op);
  if (search == _operator_map.end()) {
    throw std::logic_error(
      "Unable to convert cudf::{unary|binary}_operator to cudf::ast::ast_operator");
  }

  return search->second;
}

std::string expression_evaluator::evaluation_context::to_string() const noexcept
{
  std::string s = "evaluation_context(" + this->gqe_expression->to_string() + ", ";
  s += (this->cudf_ast_expression.has_value()) ? "has" : "no";
  s += " cudf::ast equivalent, ";
  s += (this->column_idx.has_value())
         ? "column_offset = " + std::to_string(this->column_idx.value()) + ")"
         : "no task)";
  return s;
}

expression_evaluator::expression_evaluator(cudf::table_view const& table,
                                           expression const* root_expression,
                                           cudf::size_type column_reference_offset)
  : _table{table},
    _root_expression{root_expression},
    _column_reference_offset{column_reference_offset},
    _next_task{table.num_columns()}
{
  // this->column_types is an empty member vector
  this->_column_types.reserve(table.num_columns());
  for (auto const& col : table) {
    this->_column_types.push_back(col.type());
  }

  // traverse AST recursively starting at the root node
  root_expression->accept(*this);

  // mark the root expression as to be evaluated as well
  this->dispatch_task(this->find_context(root_expression));
}

[[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>>
expression_evaluator::evaluate(bool use_like_shift_and) const
{
  auto table        = this->_table;
  auto column_types = this->_column_types;

  // storage for intermediate result columns
  std::vector<std::unique_ptr<cudf::column>> intermediate_results;
  intermediate_results.reserve(_tasks.size());

  // helper function to append any newly generated column to the table
  auto append_result =
    [&table, &column_types, &intermediate_results](std::unique_ptr<cudf::column> result) {
      table = cudf::table_view({table, cudf::table_view{{result->view()}}});
      column_types.push_back(result->type());
      intermediate_results.push_back(std::move(result));
    };

  for (auto context : _tasks) {
    // cudf::compute_column currently only supports fixed-width output columns
    bool const can_use_cudf_ast =
      context->cudf_ast_expression.has_value() &&
      cudf::is_fixed_width(context->gqe_expression->data_type(this->_column_types));

    if (can_use_cudf_ast) {
      // evaluate the the `cudf::ast` (sub-)expression using `cudf::compute_column`
      append_result(cudf::compute_column(table, *context->cudf_ast_expression.value()));
    } else {
      // fallback evaluation method
      auto const expression = context->gqe_expression.get();
      auto const children   = expression->children();

      switch (expression->type()) {
        case gqe::expression::expression_type::column_reference: {
          throw std::logic_error("Evaluating " + expression->to_string() + " is not supported");
          // TODO either implement a deep copy of the input column to the output table
          // or share a reference-counted view.
          // With this, we can remove the WAR in the evaluate_expressions function
          // for when the input expressions is just a column reference
          break;
        }
        case gqe::expression::expression_type::literal: {
          // expand literal to a column
          append_result(
            cudf::make_column_from_scalar(*context->cudf_scalar.value().get(), table.num_rows()));
          break;
        }
        case gqe::expression::expression_type::binary_op: {
          auto const binop = dynamic_cast<gqe::binary_op_expression const*>(expression);

          auto const lhs_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);
          auto const rhs_ref = dynamic_cast<gqe::column_reference_expression const*>(children[1]);

          auto const& lhs_context = context->child_contexts[0];
          auto const& rhs_context = context->child_contexts[1];

          // children must either be a column reference or literal expression
          if (!lhs_ref && children[0]->type() != expression::expression_type::literal) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " in the fallback path requires an LHS child which is either a "
                                   "column reference or literal expression but is " +
                                   children[0]->to_string() + ".");
          }

          if (!rhs_ref && children[1]->type() != expression::expression_type::literal) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " in the fallback path requires an LHS child which is either a "
                                   "column reference or literal expression but is " +
                                   children[1]->to_string() + ".");
          }

          // if the expression is a binary op we can evaluate it using `cudf::binary_operation`.
          // Children can only be of type column_reference or literal -> 4 combinations.
          if (lhs_ref && rhs_ref) {
            append_result(cudf::binary_operation(table.column(lhs_ref->column_idx()),
                                                 table.column(rhs_ref->column_idx()),
                                                 binop->binary_operator(),
                                                 binop->data_type(column_types)));
          }

          if (lhs_ref && !rhs_ref) {
            append_result(cudf::binary_operation(table.column(lhs_ref->column_idx()),
                                                 *(rhs_context->cudf_scalar.value().get()),
                                                 binop->binary_operator(),
                                                 binop->data_type(column_types)));
          }

          if (!lhs_ref && rhs_ref) {
            append_result(cudf::binary_operation(*(lhs_context->cudf_scalar.value().get()),
                                                 table.column(rhs_ref->column_idx()),
                                                 binop->binary_operator(),
                                                 binop->data_type(column_types)));
          }

          if (!lhs_ref && !rhs_ref) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " on two scalar inputs in the fallback path is not supported");
          }
          break;
        }
        case gqe::expression::expression_type::unary_op: {
          auto const op      = dynamic_cast<gqe::unary_op_expression const*>(expression);
          auto const col_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          if (!col_ref) {
            throw std::logic_error(
              "Input of `cudf::unary_operation` must be a column reference but is " +
              children[0]->to_string());
          }

          append_result(
            cudf::unary_operation(table.column(col_ref->column_idx()), op->unary_operator()));
          break;
        }
        case gqe::expression::expression_type::if_then_else: {
          auto const if_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          if (!if_ref) {
            throw std::logic_error(
              "Input predicate of `cudf::copy_if_else` must be a column reference but is " +
              children[0]->to_string());
          }

          auto const then_ref = dynamic_cast<gqe::column_reference_expression const*>(children[1]);
          auto const else_ref = dynamic_cast<gqe::column_reference_expression const*>(children[2]);

          auto const& then_context = context->child_contexts[1];
          auto const& else_context = context->child_contexts[2];

          // `cudf::copy_if_else` supports THEN and ELSE children to be either
          // of type column_reference or literal -> 4 combinations.
          if (then_ref && else_ref) {
            append_result(cudf::copy_if_else(table.column(then_ref->column_idx()),
                                             table.column(else_ref->column_idx()),
                                             table.column(if_ref->column_idx())));
          }

          if (!then_ref && else_ref) {
            append_result(cudf::copy_if_else(*(then_context->cudf_scalar.value().get()),
                                             table.column(else_ref->column_idx()),
                                             table.column(if_ref->column_idx())));
          }

          if (then_ref && !else_ref) {
            append_result(cudf::copy_if_else(table.column(then_ref->column_idx()),
                                             *(else_context->cudf_scalar.value().get()),
                                             table.column(if_ref->column_idx())));
          }

          if (!then_ref && !else_ref) {
            append_result(cudf::copy_if_else(*(then_context->cudf_scalar.value().get()),
                                             *(else_context->cudf_scalar.value().get()),
                                             table.column(if_ref->column_idx())));
          }
          break;
        }
        case gqe::expression::expression_type::cast: {
          auto const child_ref =
            dynamic_cast<gqe::column_reference_expression*>(children[0])->column_idx();
          auto const out_type = dynamic_cast<gqe::cast_expression*>(expression)->out_type();

          append_result(cudf::cast(table.column(child_ref), out_type));
          break;
        }
        case gqe::expression::expression_type::scalar_function: {
          auto const scalar_func = dynamic_cast<gqe::scalar_function_expression*>(expression);
          auto const col_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          // Scalar function expressions are always evaluated in the fallback path.
          // Thus, the child expression must always be a column reference.
          if (!col_ref) {
            throw std::logic_error(
              "Input of `gqe::scalar_function_expression` must be a column reference but is " +
              children[0]->to_string());
          }

          switch (scalar_func->fn_kind()) {
            case gqe::scalar_function_expression::function_kind::like: {
              auto const like_expr = dynamic_cast<gqe::like_expression*>(scalar_func);

              if (like_expr->ignore_case()) {
                throw std::logic_error("Evaluating ILIKE expression not implemented");
              }

              cudf::string_scalar const escape_char{like_expr->escape_character()};

              if (use_like_shift_and) {
                // if GQE_FILTER_USE_LIKE_SHIFT_AND or filter_use_like_shift_and is true,
                // we use the like_shift_and kernel for better performance
                append_result(gqe::like(
                  table.column(col_ref->column_idx()), like_expr->pattern(), escape_char));
              } else {
                cudf::string_scalar const pattern{like_expr->pattern()};
                append_result(
                  cudf::strings::like(table.column(col_ref->column_idx()), pattern, escape_char));
              }
              break;
            }
            case gqe::scalar_function_expression::function_kind::substr: {
              auto const substr_expr = dynamic_cast<gqe::substr_expression*>(scalar_func);

              cudf::numeric_scalar const start{substr_expr->start()};
              cudf::numeric_scalar const end{substr_expr->start() + substr_expr->length()};

              append_result(
                cudf::strings::slice_strings(table.column(col_ref->column_idx()), start, end));
              break;
            }
            case gqe::scalar_function_expression::function_kind::datepart: {
              auto const dp_expr = dynamic_cast<gqe::datepart_expression*>(scalar_func);

              append_result(cudf::datetime::extract_datetime_component(
                table.column(col_ref->column_idx()), dp_expr->component()));
              break;
            }
            default:
              throw std::logic_error("Unable to evaluate `scalar_function_expression` " +
                                     expression->to_string());
          }
          break;
        }
        default:
          throw std::logic_error("Cannot evaluate expression in the fallback path: " +
                                 expression->to_string());
      }
    }
  }

  // cuDF's AST module might evaluate expression to a different type than GQE expression's data
  // type. For example, the AST module might evaluate an expression to int32_t, while GQE promotes
  // the result to int64_t. In such case, we cast the result to GQE expression's data type.
  auto const expected_type = _root_expression->data_type(this->_column_types);
  if (intermediate_results.back()->type() != expected_type) {
    append_result(cudf::cast(intermediate_results.back()->view(), expected_type));
  }

  return std::make_pair(intermediate_results.back()->view(),
                        std::move(intermediate_results.back()));
}

void expression_evaluator::visit(column_reference_expression const* expression)
{
  auto const column_id = expression->column_idx() - this->_column_reference_offset;
  auto [context, is_new] =
    this->emplace_context(expression, std::make_unique<column_reference_expression>(column_id));
  if (is_new) {
    context.cudf_ast_expression = std::make_unique<cudf::ast::column_reference>(column_id);
    context.column_idx          = column_id;
  }
}

void expression_evaluator::visit(literal_expression<int8_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<int32_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<int64_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<float> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<double> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<std::string> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<cudf::timestamp_D> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(unary_op_expression const* expression)
{
  // Emplace context in the context map
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const child = expression->_children[0];
    // compute AST for the child expression
    child->accept(*this);

    // Store a reference to the child context in the parent context
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);

    bool const child_is_column  = child_context.column_idx.has_value();
    bool const can_use_cudf_ast = cudf::is_fixed_width(expression->data_type(this->_column_types));

    if (can_use_cudf_ast) {
      auto const op = cudf_to_ast_operator(expression->unary_operator());

      if (child_is_column) {
        auto const child_column_idx = child_context.column_idx.value();
        context.gqe_expression->_children[0] =
          std::make_shared<column_reference_expression>(child_column_idx);
        auto new_child = std::make_unique<cudf::ast::column_reference>(child_column_idx);
        context.cudf_ast_expression = std::make_unique<cudf::ast::operation>(op, *new_child);
        context.cudf_ast_dependencies.emplace_back(std::move(new_child));
      } else {
        context.gqe_expression->_children[0] = child_context.gqe_expression;
        context.cudf_ast_expression =
          std::make_unique<cudf::ast::operation>(op, *child_context.cudf_ast_expression.value());
      }
    } else {
      auto const child_column_idx = this->dispatch_task(child_context);
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      this->dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(is_null_expression const* expression)
{
  // Emplace context in the context map
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const child = expression->_children[0];
    // compute AST for the child expression
    child->accept(*this);

    // Store a reference to the child context in the parent context
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);

    bool const child_is_column = child_context.column_idx.has_value();
    auto const op              = cudf::ast::ast_operator::IS_NULL;

    // IS NULL expression always returns a BOOL8, which is fixed width so we can use cudf ast
    // same logic as in visit(unary_op_expression const*)
    if (child_is_column) {
      auto const child_column_idx = child_context.column_idx.value();
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      auto new_child              = std::make_unique<cudf::ast::column_reference>(child_column_idx);
      context.cudf_ast_expression = std::make_unique<cudf::ast::operation>(op, *new_child);
      context.cudf_ast_dependencies.emplace_back(std::move(new_child));
    } else {
      context.gqe_expression->_children[0] = child_context.gqe_expression;
      context.cudf_ast_expression =
        std::make_unique<cudf::ast::operation>(op, *child_context.cudf_ast_expression.value());
    }
  }
}

void expression_evaluator::visit(binary_op_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const lhs = expression->_children[0];
    auto const rhs = expression->_children[1];
    // compute AST for the child expressions
    lhs->accept(*this);
    rhs->accept(*this);

    auto& lhs_context = this->find_context(lhs.get());
    auto& rhs_context = this->find_context(rhs.get());

    context.child_contexts.emplace_back(&lhs_context);
    context.child_contexts.emplace_back(&rhs_context);

    bool const can_use_cudf_ast =
      cudf::is_fixed_width(expression->data_type(this->_column_types)) &&
      (expression->_children[0]->data_type(this->_column_types) ==
       expression->_children[1]->data_type(this->_column_types));

    if (can_use_cudf_ast) {
      auto const op = cudf_to_ast_operator(expression->binary_operator());

      auto process_child = [&](auto& context,
                               auto const child_idx) -> cudf::ast::expression const& {
        auto const& child_context = context.child_contexts[child_idx];
        bool const is_task        = child_context->column_idx.has_value();
        if (is_task) {
          auto const column_idx = child_context->column_idx.value();
          context.gqe_expression->_children[child_idx] =
            std::make_shared<column_reference_expression>(column_idx);
          auto new_child = std::make_unique<cudf::ast::column_reference>(column_idx);
          return *context.cudf_ast_dependencies.emplace_back(std::move(new_child));
        } else {
          context.gqe_expression->_children[child_idx] = child_context->gqe_expression;
          return *child_context->cudf_ast_expression.value();
        }
      };

      auto const& cudf_ast_lhs = process_child(context, 0);
      auto const& cudf_ast_rhs = process_child(context, 1);

      context.cudf_ast_expression =
        std::make_unique<cudf::ast::operation>(op, cudf_ast_lhs, cudf_ast_rhs);
    } else {
      auto process_child = [&](auto& context, int child_idx) {
        auto const& child_context = context.child_contexts[child_idx];
        bool const is_scalar      = child_context->cudf_scalar.has_value();

        if (is_scalar) {
          context.gqe_expression->_children[child_idx] = child_context->gqe_expression;
        } else {
          context.gqe_expression->_children[child_idx] =
            std::make_shared<column_reference_expression>(dispatch_task(*child_context));
        }
      };

      bool const lhs_is_scalar = lhs_context.cudf_scalar.has_value();
      bool const rhs_is_scalar = rhs_context.cudf_scalar.has_value();

      if (lhs_is_scalar && rhs_is_scalar) {
        throw std::logic_error(
          "Unable to evaluate cudf::binary_operation on two cudf::scalars in " +
          expression->to_string());
      }

      process_child(context, 0);
      process_child(context, 1);

      dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(if_then_else_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    if (!cudf::is_boolean(expression->_children[0]->data_type(this->_column_types))) {
      throw std::logic_error("Cannot convert " + expression->_children[0]->to_string() +
                             " to boolean");
    }

    if (expression->_children[1]->data_type(this->_column_types) !=
        expression->_children[2]->data_type(this->_column_types)) {
      throw std::logic_error("Column types of " + expression->_children[1]->to_string() + " and " +
                             expression->_children[2]->to_string() + " must be equal");
    }

    for (std::size_t child_id = 0; child_id < expression->_children.size(); ++child_id) {
      auto const& child = expression->_children[child_id];
      child->accept(*this);
      auto& child_context = this->find_context(child.get());
      context.child_contexts.emplace_back(&child_context);

      // IF, THEN, and ELSE columns need to be evaluated.
      // However, cudf::copy_if_else supports scalar arguments for THEN and ELSE
      if (child_id == 0 || !child_context.cudf_scalar.has_value()) {
        auto const child_column_idx = this->dispatch_task(child_context);
        context.gqe_expression->_children[child_id] =
          std::make_shared<column_reference_expression>(child_column_idx);
      }
    }

    this->dispatch_task(context);
  }
}

void expression_evaluator::visit(cast_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const& child = expression->_children[0];
    child->accept(*this);

    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);
    bool const child_is_column = child_context.column_idx.has_value();

    // cudf::ast only supports cast operations to INT64, UIN64, and FLOAT64
    std::optional<cudf::ast::ast_operator> cudf_ast_operator;
    switch (expression->out_type().id()) {
      case cudf::type_id::INT64: cudf_ast_operator = cudf::ast::ast_operator::CAST_TO_INT64; break;
      case cudf::type_id::UINT64:
        cudf_ast_operator = cudf::ast::ast_operator::CAST_TO_UINT64;
        break;
      case cudf::type_id::FLOAT64:
        cudf_ast_operator = cudf::ast::ast_operator::CAST_TO_FLOAT64;
        break;
      default: break;
    }

    if (cudf_ast_operator.has_value()) {
      // same logic as in visit(unary_op_expression const*)
      if (child_is_column) {
        auto const child_column_idx = child_context.column_idx.value();
        context.gqe_expression->_children[0] =
          std::make_shared<column_reference_expression>(child_column_idx);
        auto new_child = std::make_unique<cudf::ast::column_reference>(child_column_idx);
        context.cudf_ast_expression =
          std::make_unique<cudf::ast::operation>(*cudf_ast_operator, *new_child);
        context.cudf_ast_dependencies.emplace_back(std::move(new_child));
      } else {
        context.gqe_expression->_children[0] = child_context.gqe_expression;
        context.cudf_ast_expression          = std::make_unique<cudf::ast::operation>(
          *cudf_ast_operator, *child_context.cudf_ast_expression.value());
      }
    } else {
      auto const child_column_idx = dispatch_task(child_context);
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(scalar_function_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const& child = expression->_children[0];
    child->accept(*this);
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);
    context.gqe_expression->_children[0] =
      std::make_shared<column_reference_expression>(dispatch_task(child_context));

    // scalar function expressions are always evaluated in the fallback path
    dispatch_task(context);
  }
}

void expression_evaluator::visit(subquery_expression const* expression)
{
  expression_visitor::visit(expression);
}

cudf::size_type expression_evaluator::dispatch_task(evaluation_context& context) noexcept
{
  // is already a task
  if (context.column_idx.has_value()) {
    return context.column_idx.value();
  } else {
    // make it a new task and emit column index of the result
    auto const column_idx = _next_task++;
    context.column_idx    = column_idx;

    this->_tasks.emplace_back(&context);
    return column_idx;
  }
}

template <typename... Args>
std::pair<expression_evaluator::evaluation_context&, bool> expression_evaluator::emplace_context(
  expression const* expression, Args&&... args) noexcept
{
  auto [iter, is_new] =
    this->_evaluation_contexts.try_emplace(expression, std::forward<Args>(args)...);
  return {iter->second, is_new};
}

[[nodiscard]] expression_evaluator::evaluation_context& expression_evaluator::find_context(
  expression const* expression)
{
  auto search = this->_evaluation_contexts.find(expression);
  if (search == this->_evaluation_contexts.end()) {
    throw std::logic_error("Unable to find evaluation context for " + expression->to_string());
  }

  return search->second;
}

template <typename T>
void expression_evaluator::create_literal_context(literal_expression<T> const* expression) noexcept
{
  using cudf_scalar_type =
    std::conditional_t<std::is_same_v<T, std::string>,
                       cudf::string_scalar,
                       std::conditional_t<std::is_same_v<T, cudf::timestamp_D>,
                                          cudf::timestamp_scalar<cudf::timestamp_D>,
                                          cudf::numeric_scalar<T>>>;
  auto [context, is_new] = this->emplace_context(expression, expression->clone());
  if (is_new) {
    auto scalar = std::make_unique<cudf_scalar_type>(expression->value(), !expression->is_null());
    context.cudf_ast_expression = std::make_unique<cudf::ast::literal>(*scalar);
    context.cudf_scalar         = std::move(scalar);
  }
}

std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset,
                     bool use_like_shift_and)
{
  utility::nvtx_scoped_range eval_expr_range("evaluate_expressions");

  std::vector<cudf::column_view> evaluated_results;
  evaluated_results.reserve(exprs.size());
  std::vector<std::unique_ptr<cudf::column>> column_cache;
  column_cache.reserve(exprs.size());

  for (auto expr : exprs) {
    if (expr->type() == expression::expression_type::column_reference) {
      auto const column_idx =
        dynamic_cast<gqe::column_reference_expression const*>(expr)->column_idx();

      if (column_idx < column_reference_offset) {
        throw std::out_of_range("Invalid column index and offset combination in expression: " +
                                expr->to_string());
      }
      evaluated_results.push_back(table.column(column_idx - column_reference_offset));
      continue;
    } else if (table.num_rows() == 0) {
      std::vector<cudf::data_type> column_types;
      column_types.reserve(table.num_columns());
      for (auto const& col : table) {
        column_types.push_back(col.type());
      }
      auto empty_column = cudf::make_empty_column(expr->data_type(column_types));
      evaluated_results.push_back(empty_column->view());
      column_cache.push_back(std::move(empty_column));
    } else {
      auto evaluator       = expression_evaluator(table, expr, column_reference_offset);
      auto [result, cache] = evaluator.evaluate(use_like_shift_and);

      evaluated_results.push_back(std::move(result));
      column_cache.push_back(std::move(cache));
    }
  }

  return std::make_pair(std::move(evaluated_results), std::move(column_cache));
}

}  // namespace gqe
