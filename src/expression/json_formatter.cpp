/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/json_formatter.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>

#include <sstream>

gqe::expression_json_formatter::expression_json_formatter(std::ostream& out)
  : _out(out), _isFirst(true)
{
}

std::string gqe::expression_json_formatter::to_json(const gqe::expression& expression)
{
  std::stringstream ss;
  expression_json_formatter formatter(ss);
  expression.accept(formatter);
  return ss.str();
}

void gqe::expression_json_formatter::open()
{
  _out << "{";
  _isFirst = true;
}

void gqe::expression_json_formatter::close() const { _out << "}"; }

void gqe::expression_json_formatter::add_key_value_pair(
  const std::string_view key, const std::function<void()>& value_serialization_functor)
{
  if (!_isFirst) {
    _out << ",";
  } else {
    _isFirst = false;
  }
  _out << "\"" << key << "\":";
  value_serialization_functor();
}

void gqe::expression_json_formatter::add_key_value_pair(const std::string_view key,
                                                        const std::string_view value)
{
  add_key_value_pair(key, [&] { _out << "\"" << value << "\""; });
}

void gqe::expression_json_formatter::add_expression_type_key_value_pair(
  const std::string_view value)
{
  add_key_value_pair("expression_type", value);
}

void gqe::expression_json_formatter::add_literal_key_value_pair(const expression& expression,
                                                                const std::string_view value)
{
  open();
  add_expression_type_key_value_pair("literal(" + cudf::type_to_name(expression.data_type({})) +
                                     ")");
  add_key_value_pair("value", value);
  close();
}

void gqe::expression_json_formatter::visit(const gqe::binary_op_expression* expression)
{
  open();
  const std::function<const std::string()> type_to_string{[&expression]() -> std::string {
    switch (auto op = expression->binary_operator()) {
      case cudf::binary_operator::NULL_LOGICAL_AND: return "and_expression";
      case cudf::binary_operator::NULL_LOGICAL_OR: return "or_expression";
      case cudf::binary_operator::EQUAL: return "equal_expression";
      case cudf::binary_operator::NOT_EQUAL: return "not_equal_expression";
      case cudf::binary_operator::LESS: return "less_expression";
      case cudf::binary_operator::LESS_EQUAL: return "less_equal_expression";
      case cudf::binary_operator::GREATER: return "greater_expression";
      case cudf::binary_operator::GREATER_EQUAL: return "greater_equal_expression";
      default: return "unknown binary operator type: " + std::to_string(static_cast<int32_t>(op));
    }
  }};
  const auto children = expression->children();
  const auto lhs      = children[0];
  const auto rhs      = children[1];
  add_expression_type_key_value_pair(type_to_string());
  add_key_value_pair("lhs", [&] { lhs->accept(*this); });
  add_key_value_pair("rhs", [&] { rhs->accept(*this); });
  close();
}

void gqe::expression_json_formatter::visit(const column_reference_expression* expression)
{
  open();
  add_expression_type_key_value_pair("column_reference");
  add_key_value_pair("column_index", std::to_string(expression->column_idx()));
  close();
}

void gqe::expression_json_formatter::visit(const literal_expression<std::string>* expression)
{
  add_literal_key_value_pair(*expression, expression->value());
}

void gqe::expression_json_formatter::visit(const literal_expression<int32_t>* expression)
{
  add_literal_key_value_pair(*expression, std::to_string(expression->value()));
}

void gqe::expression_json_formatter::visit(const literal_expression<int64_t>* expression)
{
  add_literal_key_value_pair(*expression, std::to_string(expression->value()));
}

void gqe::expression_json_formatter::visit(const literal_expression<float>* expression)
{
  add_literal_key_value_pair(*expression, std::to_string(expression->value()));
}

void gqe::expression_json_formatter::visit(const literal_expression<int8_t>* expression)
{
  add_literal_key_value_pair(*expression, std::to_string(expression->value()));
}

void gqe::expression_json_formatter::visit(const literal_expression<cudf::timestamp_D>* expression)
{
  const std::time_t time_since_epoch =
    cuda::std::chrono::system_clock::to_time_t(expression->value());
  std::tm tm{};
  localtime_r(&time_since_epoch, &tm);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d");
  add_literal_key_value_pair(*expression, ss.str());
}

void gqe::expression_json_formatter::visit(const literal_expression<double>* expression)
{
  add_literal_key_value_pair(*expression, std::to_string(expression->value()));
}

void gqe::expression_json_formatter::visit(const scalar_function_expression* expression)
{
  open();
  switch (expression->fn_kind()) {
    case gqe::scalar_function_expression::function_kind::like: {
      const auto like_expression = dynamic_cast<const gqe::like_expression*>(expression);
      add_expression_type_key_value_pair("like_expression");
      add_key_value_pair("pattern", like_expression->pattern());
      add_key_value_pair("escape_character", like_expression->escape_character());
      add_key_value_pair("ignore_case", std::to_string(like_expression->ignore_case()));
    } break;
    case gqe::scalar_function_expression::function_kind::datepart: {
      const auto datepart_expression = dynamic_cast<const gqe::datepart_expression*>(expression);
      add_expression_type_key_value_pair("datepart_expression");
      add_key_value_pair("component",
                         std::to_string(static_cast<uint8_t>(datepart_expression->component())));
    } break;
    case gqe::scalar_function_expression::function_kind::round: {
      const auto round_expression = dynamic_cast<const gqe::round_expression*>(expression);
      add_expression_type_key_value_pair("round_expression");
      add_key_value_pair("decimal_places", std::to_string(round_expression->decimal_places()));
    } break;
    case gqe::scalar_function_expression::function_kind::substr: {
      const auto substr_expression = dynamic_cast<const gqe::substr_expression*>(expression);
      add_expression_type_key_value_pair("substr_expression");
      add_key_value_pair("start", std::to_string(substr_expression->start()));
      add_key_value_pair("length", std::to_string(substr_expression->length()));
    } break;
    default: expression_visitor::visit(expression);
  }
  const auto child = expression->children()[0];
  add_key_value_pair("child", [&] { child->accept(*this); });
  close();
}

void gqe::expression_json_formatter::visit(const unary_op_expression* expression)
{
  open();
  const std::function<const std::string()> type_to_string{[&expression]() -> std::string {
    switch (const auto op = expression->unary_operator()) {
      case cudf::unary_operator::NOT: return "not_expression";
      default: return "unknown unary operator type: " + std::to_string(static_cast<int32_t>(op));
    }
  }};
  const auto child = expression->children()[0];
  add_expression_type_key_value_pair(type_to_string());
  add_key_value_pair("child", [&] { child->accept(*this); });
  close();
}

void gqe::expression_json_formatter::visit(const subquery_expression* expression)
{
  open();
  const std::function<const std::string()> type_to_string{[&expression]() -> std::string {
    switch (const auto type = expression->subquery_type()) {
      case gqe::subquery_expression::subquery_type_type::in_predicate: return "in_predicate";
      default: return "unknown subquery type: " + std::to_string(static_cast<int32_t>(type));
    }
  }};
  add_expression_type_key_value_pair(type_to_string());
  add_key_value_pair("relation_index", std::to_string(expression->relation_index()));
  add_key_value_pair("children", [&] {
    _out << "[";
    const auto& children = expression->children();
    std::for_each(children.begin(), children.end() - 1, [&](const auto& child) {
      child->accept(*this);
      _out << ",";
    });
    children.back()->accept(*this);
    _out << "]";
  });
  close();
}

void gqe::expression_json_formatter::visit(const if_then_else_expression* expression)
{
  open();
  const auto& children = expression->children();
  const auto if_expr   = children[0];
  const auto then_expr = children[1];
  const auto else_expr = children[2];
  add_expression_type_key_value_pair("if_then_else_expression");
  add_key_value_pair("if_expr", [&]() { if_expr->accept(*this); });
  add_key_value_pair("then_expr", [&]() { then_expr->accept(*this); });
  add_key_value_pair("else_expr", [&]() { else_expr->accept(*this); });
  close();
}

void gqe::expression_json_formatter::visit(const cast_expression* expression)
{
  open();
  add_expression_type_key_value_pair("cast_expression");
  add_key_value_pair("out_type", cudf::type_to_name(expression->out_type()));
  add_key_value_pair("child", [&] { expression->children().front()->accept(*this); });
  close();
}

void gqe::expression_json_formatter::visit(const is_null_expression* expression)
{
  open();
  add_expression_type_key_value_pair("is_null_expression");
  add_key_value_pair("child", [&] { expression->children().front()->accept(*this); });
  close();
}
