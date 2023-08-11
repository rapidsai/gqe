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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/utility/logger.hpp>

#include <substrait/algebra.pb.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

void gqe::substrait_parser::add_function_reference(uint32_t reference, std::string function_name)
{
  auto search = function_reference_to_name.find(reference);
  if (search != function_reference_to_name.end())
    throw std::runtime_error("Cannot add function reference: key already exists");

  function_reference_to_name[reference] = function_name;
}

std::string gqe::substrait_parser::get_function_name(uint32_t reference) const
{
  auto search = function_reference_to_name.find(reference);
  if (search == function_reference_to_name.end())
    throw std::runtime_error("Cannot get function name: key does not exist");

  return search->second;
}

std::vector<std::shared_ptr<gqe::logical::relation>> gqe::substrait_parser::from_file(
  std::string substrait_file)
{
  const std::string file_ext = ".bin";
  assert(mismatch(file_ext.rbegin(), file_ext.rend(), substrait_file.rbegin()).first ==
         file_ext.rend());
  std::ifstream query_plan_stream(substrait_file, std::ios::binary);
  substrait::Plan query_plan;
  query_plan.ParseFromIstream(&query_plan_stream);

  // Register functions
  for (auto& extension : query_plan.extensions()) {
    assert(extension.mapping_type_case() ==
           substrait::extensions::SimpleExtensionDeclaration::MappingTypeCase::kExtensionFunction);
    auto function_extension = extension.extension_function();

    add_function_reference(function_extension.function_anchor(), function_extension.name());
  }

  // Parse relation trees
  std::vector<std::shared_ptr<gqe::logical::relation>> relation_trees;

  for (auto relation : query_plan.relations()) {
    if (!relation.has_root()) {
      if (!relation.has_rel()) throw std::runtime_error("Top level PlanRel has no root or rel");
      relation_trees.push_back(parse_relation(relation.rel()));
    } else {
      relation_trees.push_back(parse_relation(relation.root().input()));
    }
  }

  return relation_trees;
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_expression(
  substrait::Expression const& expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subqueries) const
{
  if (expression.has_selection())
    return parse_selection_expression(expression.selection());
  else if (expression.has_scalar_function())
    return parse_scalar_function_expression(expression.scalar_function(), subqueries);
  else if (expression.has_literal())
    return parse_literal_expression(expression.literal());
  else if (expression.has_subquery())
    return parse_subquery_expression(expression.subquery(), subqueries);
  else if (expression.has_if_then())
    return parse_if_then_expression(expression.if_then(), subqueries);
  else if (expression.has_singular_or_list())
    return parse_in_list_expression(expression.singular_or_list(), subqueries);
  else if (expression.has_cast())
    return parse_cast_expression(expression.cast(), subqueries);
  else
    throw std::runtime_error("SubstraitParser cannot parse expression with type " +
                             std::to_string(expression.rex_type_case()));
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_if_then_expression(
  substrait::Expression_IfThen const& if_then_expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  if (if_then_expression.ifs_size() != 1)
    throw std::runtime_error(
      "Attempting to parse IfThen expression of size " +
      std::to_string(if_then_expression.ifs_size()) +
      ". SubstraitParser only support IfThen expression with `ifs` size of 1.");

  auto if_expr   = parse_expression(if_then_expression.ifs().at(0).if_(), subquery_relations);
  auto then_expr = parse_expression(if_then_expression.ifs().at(0).then(), subquery_relations);
  auto else_expr = parse_expression(if_then_expression.else_(), subquery_relations);

  return std::make_unique<gqe::if_then_else_expression>(
    std::move(if_expr), std::move(then_expr), std::move(else_expr));
}

namespace {
// Helper function for constructing an expression tree that OR() together EQUAL(value_expr,
// options[i]) expressions
std::unique_ptr<gqe::expression> construct_or_eq_list_expression(
  std::unique_ptr<gqe::expression> value_expr,
  std::vector<std::unique_ptr<gqe::expression>>& options)
{
  auto equal_expr =
    std::make_unique<gqe::equal_expression>(value_expr->clone(), std::move(options.back()));
  options.pop_back();
  // No more element to compare
  if (options.size() == 0) { return equal_expr; }
  // More element(s) to construct comparison(s) against
  return std::make_unique<gqe::logical_or_expression>(
    std::move(equal_expr), construct_or_eq_list_expression(std::move(value_expr), options));
}

// Helper function for translating Substrait decimal to C++ decimal
// Substrait encodes decimal value as 16-byte little-endian array
// source: https://substrait.io/types/type_classes/#compound-types
numeric::decimal128 from_substrait_decimal(substrait::Expression_Literal_Decimal decimal_expr)
{
  auto v_str = decimal_expr.value();
  assert(v_str.length() == 16);
  // Note: This implementation only works on little-endian machines. The bytes need to be reversed
  // on big-endian machines.
  auto const decimal_value = *reinterpret_cast<numeric::decimal128::rep const*>(v_str.c_str());
  auto const decimal_scale = static_cast<numeric::scale_type>(-decimal_expr.scale());
  numeric::scaled_integer<numeric::decimal128::rep> scaled_integer_value(decimal_value,
                                                                         decimal_scale);
  numeric::decimal128 fixed_point_value(scaled_integer_value);
  return fixed_point_value;
}
}  // namespace

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_in_list_expression(
  substrait::Expression_SingularOrList const& singular_or_list,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto value = parse_expression(singular_or_list.value(), subquery_relations);
  std::vector<std::unique_ptr<expression>> options;
  options.reserve(singular_or_list.options_size());
  std::transform(singular_or_list.options().begin(),
                 singular_or_list.options().end(),
                 std::back_inserter(options),
                 [this, &subquery_relations](substrait::Expression option) {
                   return parse_expression(option, subquery_relations);
                 });
  // Make OR(value = options[0], value = options[1], ..., value = options[n-1])
  return construct_or_eq_list_expression(std::move(value), options);
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_literal_expression(
  substrait::Expression_Literal const& literal_expression) const
{
  // TODO:
  // - Add unit test for all cases. For now, only i32, decimal, and NULL has been validated with
  // real plan
  switch (literal_expression.literal_type_case()) {
    case substrait::Expression_Literal::LiteralTypeCase::kBoolean:
      return std::make_unique<gqe::literal_expression<bool>>(literal_expression.boolean());
    case substrait::Expression_Literal::LiteralTypeCase::kI8:
      return std::make_unique<gqe::literal_expression<int32_t>>(literal_expression.i8());
    case substrait::Expression_Literal::LiteralTypeCase::kI16:
      return std::make_unique<gqe::literal_expression<int32_t>>(literal_expression.i16());
    case substrait::Expression_Literal::LiteralTypeCase::kI32:
      return std::make_unique<gqe::literal_expression<int32_t>>(literal_expression.i32());
    case substrait::Expression_Literal::LiteralTypeCase::kI64:
      return std::make_unique<gqe::literal_expression<int64_t>>(literal_expression.i64());
    case substrait::Expression_Literal::LiteralTypeCase::kFp32:
      return std::make_unique<gqe::literal_expression<float>>(literal_expression.fp32());
    case substrait::Expression_Literal::LiteralTypeCase::kFp64:
      return std::make_unique<gqe::literal_expression<double>>(literal_expression.fp64());
    case substrait::Expression_Literal::LiteralTypeCase::kString:
      return std::make_unique<gqe::literal_expression<std::string>>(literal_expression.string());
    case substrait::Expression_Literal::LiteralTypeCase::kFixedChar:
      return std::make_unique<gqe::literal_expression<std::string>>(
        literal_expression.fixed_char());
    case substrait::Expression_Literal::LiteralTypeCase::kDate: {
      cudf::duration_D duration(literal_expression.date());
      cudf::timestamp_D date(duration);
      return std::make_unique<gqe::literal_expression<cudf::timestamp_D>>(date);
    }
    case substrait::Expression_Literal::LiteralTypeCase::kDecimal: {
      // For now, always parse a decimal into a floating point number
      auto fixed_point_value = from_substrait_decimal(literal_expression.decimal());
      return std::make_unique<gqe::literal_expression<double>>(
        static_cast<double>(fixed_point_value));
    }
    case substrait::Expression_Literal::LiteralTypeCase::kNull: {
      std::unique_ptr<gqe::expression> null_literal;
      // TODO: Support more NULL types
      // We currently treat all integers as int64_t
      if (literal_expression.null().has_i16())
        null_literal = std::make_unique<gqe::literal_expression<int64_t>>(0, true);
      else if (literal_expression.null().has_i32())
        null_literal = std::make_unique<gqe::literal_expression<int64_t>>(0, true);
      else if (literal_expression.null().has_i32())
        null_literal = std::make_unique<gqe::literal_expression<int64_t>>(0, true);
      // We currently treat all floating point and decimals as double
      else if (literal_expression.null().has_decimal())
        null_literal = std::make_unique<gqe::literal_expression<double>>(0, true);
      else if (literal_expression.null().has_fp32())
        null_literal = std::make_unique<gqe::literal_expression<double>>(0, true);
      else if (literal_expression.null().has_fp64())
        null_literal = std::make_unique<gqe::literal_expression<double>>(0, true);
      else if (literal_expression.null().has_string())
        null_literal = std::make_unique<gqe::literal_expression<std::string>>("", true);
      else
        throw std::runtime_error("SubstraitParser cannot parse null literal expression with type " +
                                 std::to_string(literal_expression.null().kind_case()));

      return null_literal;
    }
    default:
      throw std::runtime_error("SubstraitParser cannot parse literal expression with type " +
                               std::to_string(literal_expression.literal_type_case()));
  }
}

namespace {
// Helper function for translating Substrait data type to cudf data type
cudf::data_type substrait_to_cudf_type(substrait::Type const& substrait_type)
{
  switch (substrait_type.kind_case()) {
    case substrait::Type::kBool: return cudf::data_type(cudf::type_id::BOOL8);
    case substrait::Type::kI8: return cudf::data_type(cudf::type_id::INT8);
    case substrait::Type::kI16: return cudf::data_type(cudf::type_id::INT16);
    case substrait::Type::kI32: return cudf::data_type(cudf::type_id::INT32);
    case substrait::Type::kI64: return cudf::data_type(cudf::type_id::INT64);
    case substrait::Type::kFp32: return cudf::data_type(cudf::type_id::FLOAT32);
    case substrait::Type::kFp64: return cudf::data_type(cudf::type_id::FLOAT64);
    case substrait::Type::kString:
    case substrait::Type::kVarchar: return cudf::data_type(cudf::type_id::STRING);
    case substrait::Type::kDate: return cudf::data_type(cudf::type_id::DURATION_DAYS);
    case substrait::Type::kDecimal: {
      auto const precision = substrait_type.decimal().precision();
      auto const scale     = substrait_type.decimal().scale();

      // Precision in radix-10 that can be represented by b bits = floor((b-1)*log_10(2))
      constexpr decltype(precision) decimal32_precision_threshold  = 9;
      constexpr decltype(precision) decimal64_precision_threshold  = 18;
      constexpr decltype(precision) decimal128_precision_threshold = 38;

      if (precision < 1 || precision > decimal128_precision_threshold) {
        throw std::logic_error("Invalid decimal precision in the substrait plan");
      }

      // Let's call the integer stored in the decimal `rep`, and the value it represented `value`.
      // In cuDF, `rep = value / 10^scale` but in Substrait `value = rep / 10^scale`. So, we flip
      // the sign bit of `scale` here to make them compatible.
      if (precision <= decimal32_precision_threshold) {
        return cudf::data_type(cudf::type_id::DECIMAL32, -scale);
      } else if (precision <= decimal64_precision_threshold) {
        return cudf::data_type(cudf::type_id::DECIMAL64, -scale);
      } else {
        return cudf::data_type(cudf::type_id::DECIMAL128, -scale);
      }
    }
    default:
      throw std::runtime_error("SubstraitParser cannot convert substrait type " +
                               std::to_string(substrait_type.kind_case()) + " to cuDF type");
  }
}

// Helper function to translate datetime component string into
// gqe::date_part_expression::datetime_component
gqe::datepart_expression::datetime_component datetime_component_from_str(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s == "year") return gqe::datepart_expression::datetime_component::year;
  if (s == "month") return gqe::datepart_expression::datetime_component::month;
  if (s == "day") return gqe::datepart_expression::datetime_component::day;
  if (s == "weekday") return gqe::datepart_expression::datetime_component::weekday;
  if (s == "hour") return gqe::datepart_expression::datetime_component::hour;
  if (s == "minute") return gqe::datepart_expression::datetime_component::minute;
  if (s == "second") return gqe::datepart_expression::datetime_component::second;
  if (s == "millisecond") return gqe::datepart_expression::datetime_component::millisecond;
  if (s == "nanosecond")
    return gqe::datepart_expression::datetime_component::nanosecond;
  else
    throw std::runtime_error("Unsupported datetime component string: " + s);
}

// Helper function for extracting literal value
template <typename T>
T try_get_literal_value(std::shared_ptr<gqe::expression> expr)
{
  auto expr_type = expr->type();
  auto data_type = expr->data_type({});

  if (expr_type == gqe::expression::expression_type::literal) {
    if ((std::is_integral_v<T> && cudf::is_integral(data_type)) ||
        (std::is_convertible_v<T, std::string> && (data_type.id() == cudf::type_id::STRING))) {
      auto lit = dynamic_cast<gqe::literal_expression<T>*>(expr.get());
      return lit->value();
    } else {
      throw std::invalid_argument(
        "Either the expression output data type (" + cudf::type_to_name(data_type) +
        ") is not suported by try_get_literal_value(), or the type " +
        cudf::type_to_name(data_type) + " is incompatible with the template type");
    }
  } else {
    throw std::invalid_argument(
      "try_get_integral_value() expects literal expresion, but got expression type " +
      std::to_string(static_cast<int32_t>(expr_type)));
  }
}

// Helper function to maintain the mapping between function names in Substrait (DataFusion) and GQE
// scalar function kinds
std::unordered_map<std::string, gqe::scalar_function_expression::function_kind>&
name_to_fkind_map() noexcept
{
  static std::unordered_map<std::string, gqe::scalar_function_expression::function_kind> map = {
    {"date_part", gqe::scalar_function_expression::function_kind::datepart},
    {"ilike", gqe::scalar_function_expression::function_kind::like},
    {"like", gqe::scalar_function_expression::function_kind::like},
    {"round", gqe::scalar_function_expression::function_kind::round},
    {"substr", gqe::scalar_function_expression::function_kind::substr}};
  return map;
}

// Return whether `function_name` has a corresponding `function_kind` supported by GQE
bool is_scalar_function(std::string function_name) noexcept
{
  auto map = name_to_fkind_map();
  return map.find(function_name) != map.end();
}

// Return scalar function kind enum from function name string
gqe::scalar_function_expression::function_kind fn_kind_from_str(std::string s)
{
  if (is_scalar_function(s)) return name_to_fkind_map()[s];
  throw std::runtime_error(
    "Cannot find a corresponding supported scalar function for function name \"" + s + "\"");
}

}  // namespace

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_cast_expression(
  substrait::Expression_Cast const& cast_expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto input_expr       = parse_expression(cast_expression.input(), subquery_relations);
  auto output_data_type = substrait_to_cudf_type(cast_expression.type());
  return std::make_unique<gqe::cast_expression>(std::move(input_expr), output_data_type);
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_selection_expression(
  substrait::Expression_FieldReference const& selection_expression) const
{
  if (!selection_expression.has_direct_reference())
    throw std::runtime_error("Only direct reference is supported for selection expressions");

  if (!selection_expression.direct_reference().has_struct_field())
    throw std::runtime_error("Only struct field is supported for selection expressions");

  auto struct_field = selection_expression.direct_reference().struct_field();

  if (struct_field.has_child())
    throw std::runtime_error("Does not support struct field with child");

  return std::make_unique<gqe::column_reference_expression>(struct_field.field());
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::_parse_scalar_function_expression(
  std::string function_name,
  std::vector<substrait::Expression> const& arg_expressions,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  int nargs = arg_expressions.size();
  if (is_scalar_function(function_name)) {
    // Get function kind and parse arguments
    gqe::scalar_function_expression::function_kind fn_kind = fn_kind_from_str(function_name);
    std::vector<std::shared_ptr<expression>> parsed_arguments;
    parsed_arguments.reserve(nargs);
    for (auto arg_expr : arg_expressions) {
      parsed_arguments.push_back(parse_expression(arg_expr, subquery_relations));
    }
    // Parse function
    switch (fn_kind) {
      case gqe::scalar_function_expression::function_kind::datepart: {
        assert(nargs == 2);
        auto component_str = try_get_literal_value<std::string>(parsed_arguments[0]);
        auto input         = parsed_arguments[1];
        return std::make_unique<gqe::datepart_expression>(
          std::move(input), datetime_component_from_str(component_str));
      }
      case gqe::scalar_function_expression::function_kind::like: {
        if (nargs < 2 || nargs > 3)
          throw std::runtime_error("like/ilike() expects 2 or 3 arguments. Got " +
                                   std::to_string(nargs) + " arguments");
        auto input       = parsed_arguments[0];
        auto pattern     = try_get_literal_value<std::string>(parsed_arguments[1]);
        auto escape_char = try_get_literal_value<std::string>(parsed_arguments[2]);
        return std::make_unique<gqe::like_expression>(
          std::move(input), pattern, escape_char, function_name == "ilike" ? true : false);
      }
      case gqe::scalar_function_expression::function_kind::round: {
        if (nargs != 2)
          throw std::runtime_error("round() currently only supports 2 arguments. Got " +
                                   std::to_string(nargs) + " arguments");
        auto input          = parsed_arguments[0];
        auto decimal_places = try_get_literal_value<std::int64_t>(parsed_arguments[1]);
        return std::make_unique<gqe::round_expression>(std::move(input), decimal_places);
      }
      case gqe::scalar_function_expression::function_kind::substr: {
        assert(nargs == 3);
        auto input  = parsed_arguments[0];
        auto start  = try_get_literal_value<std::int64_t>(parsed_arguments[1]);
        auto length = try_get_literal_value<std::int64_t>(parsed_arguments[2]);
        return std::make_unique<gqe::substr_expression>(std::move(input), start, length);
      }
      default: throw std::runtime_error("ScalarFunction " + function_name + "() is not supported");
    }
  } else {
    // If the function is not part of the supported scalar_function::function_kind types,
    // attempt to parse as a cudf supported operation
    if (nargs == 1) {
      auto input = parse_expression(arg_expressions[0], subquery_relations);
      if (function_name == "not")
        return std::make_unique<gqe::not_expression>(std::move(input));
      else
        throw std::runtime_error("SubstraitParser cannot parse unary scalar function \"" +
                                 function_name + "\"");
    } else if (nargs == 2) {  // Binary function base case
      auto lhs = parse_expression(arg_expressions[0], subquery_relations);
      auto rhs = parse_expression(arg_expressions[1], subquery_relations);

      if (function_name == "equal" || function_name == "equal:any_any")
        return std::make_unique<gqe::equal_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "not_equal")
        return std::make_unique<gqe::not_equal_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "and" || function_name == "and:bool")
        return std::make_unique<gqe::logical_and_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "or" || function_name == "or:bool")
        return std::make_unique<gqe::logical_or_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "gt" || function_name == "gt:any_any")
        return std::make_unique<gqe::greater_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "lt" || function_name == "lt:any_any")
        return std::make_unique<gqe::less_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "gte" || function_name == "gte:any_any")
        return std::make_unique<gqe::greater_equal_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "lte" || function_name == "lte:any_any")
        return std::make_unique<gqe::less_equal_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "is_not_distinct_from")
        return std::make_unique<gqe::nulls_equal_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "add")
        return std::make_unique<gqe::add_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "substract")
        return std::make_unique<gqe::subtract_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "multiply")
        return std::make_unique<gqe::multiply_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "divide")
        return std::make_unique<gqe::divide_expression>(std::move(lhs), std::move(rhs));
      else
        throw std::runtime_error("SubstraitParser cannot parse binary scalar function \"" +
                                 function_name + "\"");
    } else if (nargs > 2) {  // Multi-argument function recursive case
      // Only associative functions are implemented here
      auto output_expr = parse_expression(arg_expressions[0], subquery_relations);
      std::for_each(
        arg_expressions.begin() + 1,
        arg_expressions.end(),
        [&](substrait::Expression const& input_expr) {
          auto rhs = parse_expression(input_expr, subquery_relations);
          if (function_name == "and" || function_name == "and:bool")
            output_expr =
              std::make_unique<gqe::logical_and_expression>(std::move(output_expr), std::move(rhs));
          else if (function_name == "or" || function_name == "or:bool")
            output_expr =
              std::make_unique<gqe::logical_or_expression>(std::move(output_expr), std::move(rhs));
          else
            throw std::runtime_error("Cannot find matching multi-argument scalar function \"" +
                                     function_name + "\"" + " with " + std::to_string(nargs) +
                                     " arguments");
        });
      return output_expr;
    } else {
      throw std::runtime_error("Cannot find matching ScalarFunction with less than 1 arguments: " +
                               function_name);
    }
  }
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_scalar_function_expression(
  substrait::Expression_ScalarFunction const& scalar_function_expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto function_name = get_function_name(scalar_function_expression.function_reference());
  int nargs          = scalar_function_expression.arguments_size();
  // Get arguments
  std::vector<substrait::Expression> arg_expressions;
  arg_expressions.reserve(nargs);
  for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
    arg_expressions.push_back(scalar_function_expression.arguments().Get(arg_idx).value());
  }
  // Parse scalar function with extracted function name and arguments
  return _parse_scalar_function_expression(function_name, arg_expressions, subquery_relations);
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_subquery_expression(
  substrait::Expression_Subquery const& subquery_expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subqueries) const
{
  throw std::logic_error("Subquery expression parser has not been implemented");
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_relation(
  substrait::Rel const& relation) const
{
  if (relation.has_read())
    return parse_read_relation(relation.read());
  else if (relation.has_project())
    return parse_project_relation(relation.project());
  else if (relation.has_join())
    return parse_join_relation(relation.join());
  else if (relation.has_fetch())
    return parse_fetch_relation(relation.fetch());
  else if (relation.has_filter())
    return parse_filter_relation(relation.filter());
  else if (relation.has_sort())
    return parse_sort_relation(relation.sort());
  else if (relation.has_aggregate())
    return parse_aggregate_relation(relation.aggregate());
  else
    throw std::runtime_error("Unsupported relation type");
}

namespace {
gqe::window_frame_bound::type parse_window_bound(
  substrait::Expression_WindowFunction_Bound const& bound)
{
  if (bound.has_unbounded()) {
    return gqe::window_frame_bound::unbounded{};
  } else if (bound.has_preceding()) {
    auto offset = bound.preceding().offset();
    return gqe::window_frame_bound::bounded{-1 * offset};
  } else if (bound.has_following()) {
    auto offset = bound.following().offset();
    return gqe::window_frame_bound::bounded{offset};
  } else if (bound.has_current_row()) {
    return gqe::window_frame_bound::bounded{0};
  } else {
    throw std::runtime_error("Unsupported bound type " + std::to_string(bound.kind_case()));
  }
}
}  // namespace

void gqe::substrait_parser::parse_sorts(
  google::protobuf::RepeatedPtrField<substrait::SortField> const& sorts,
  std::vector<std::unique_ptr<gqe::expression>>& expressions,
  std::vector<cudf::order>& column_orders,
  std::vector<cudf::null_order>& null_precedences,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  for (auto const& sort_order : sorts) {
    if (!sort_order.has_direction())
      throw std::runtime_error("Does not support sort with comparison function reference");

    expressions.push_back(parse_expression(sort_order.expr(), subquery_relations));

    switch (sort_order.direction()) {
      case substrait::SortField_SortDirection::
        SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_FIRST:
        column_orders.push_back(cudf::order::ASCENDING);
        null_precedences.push_back(cudf::null_order::BEFORE);
        break;
      case substrait::SortField_SortDirection::
        SortField_SortDirection_SORT_DIRECTION_ASC_NULLS_LAST:
        column_orders.push_back(cudf::order::ASCENDING);
        null_precedences.push_back(cudf::null_order::AFTER);
        break;
      case substrait::SortField_SortDirection::
        SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_FIRST:
        column_orders.push_back(cudf::order::DESCENDING);
        null_precedences.push_back(cudf::null_order::BEFORE);
        break;
      case substrait::SortField_SortDirection::
        SortField_SortDirection_SORT_DIRECTION_DESC_NULLS_LAST:
        column_orders.push_back(cudf::order::DESCENDING);
        null_precedences.push_back(cudf::null_order::AFTER);
        break;
      default: throw std::runtime_error("Unsupported SortField_SortDirection");
    }
  }
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_window_function_expression(
  substrait::Expression::WindowFunction const& window_function_expression,
  std::unique_ptr<gqe::logical::relation> input_relation,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto function_name         = get_function_name(window_function_expression.function_reference());
  [[maybe_unused]] int nargs = window_function_expression.arguments_size();
  cudf::aggregation::Kind aggr_func;
  std::vector<std::unique_ptr<expression>> arguments;
  std::vector<std::unique_ptr<expression>> order_by;
  std::vector<std::unique_ptr<expression>> partition_by;
  std::vector<cudf::order> order_dirs;
  // NULL precedences are currently set but not used
  // TODO: Add to window relation
  [[maybe_unused]] std::vector<cudf::null_order> null_precedences;

  window_frame_bound::type window_lower_bound =
    parse_window_bound(window_function_expression.lower_bound());
  window_frame_bound::type window_upper_bound =
    parse_window_bound(window_function_expression.upper_bound());

  if (function_name == "rank") {
    assert(nargs == 0);
    aggr_func = cudf::aggregation::Kind::RANK;
  } else if (function_name == "sum") {
    assert(nargs == 1);
    aggr_func = cudf::aggregation::Kind::SUM;
    arguments.push_back(
      parse_expression(window_function_expression.arguments().Get(0).value(), subquery_relations));
  } else {
    throw std::runtime_error("SubstraitParser cannot parse aggr/window function \"" +
                             function_name + "\"");
  }

  for (auto partition_expr : window_function_expression.partitions()) {
    partition_by.push_back(parse_expression(partition_expr, subquery_relations));
  }

  parse_sorts(
    window_function_expression.sorts(), order_by, order_dirs, null_precedences, subquery_relations);

  return std::make_unique<gqe::logical::window_relation>(std::move(input_relation),
                                                         std::move(subquery_relations),
                                                         aggr_func,
                                                         std::move(arguments),
                                                         std::move(order_by),
                                                         std::move(partition_by),
                                                         std::move(order_dirs),
                                                         window_lower_bound,
                                                         window_upper_bound);
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_filter_relation(
  substrait::FilterRel const& filter_relation) const
{
  auto input_relation = parse_relation(filter_relation.input());
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  auto condition = parse_expression(filter_relation.condition(), subquery_relations);

  return std::make_unique<gqe::logical::filter_relation>(
    std::move(input_relation), std::move(subquery_relations), std::move(condition));
}

std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>
gqe::substrait_parser::parse_aggregate_function(
  substrait::AggregateFunction const& aggregate_function,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto function_name = get_function_name(aggregate_function.function_reference());
  int nargs          = aggregate_function.arguments_size();
  // We use DataFusion aggregate function naming standard for producer/consumer compatibility.
  // The producer uses the lowercase version of `Expr::AggregateFunction::to_string` in the file
  // https://github.com/apache/arrow-datafusion/blob/main/datafusion/expr/src/aggregate_function.rs
  // If in the future, other producer is used, we can add more naming options.
  if (function_name == "count") {
    // Different Substrait producer encode `count(*)` differently. Cases encountered so far:
    // - DataFusion encodes `count(*)` as `count(1)` in Substrait plan
    // - Isthmus encodes `count(*)` as `count()` in Substrait plan
    // According to https://www.postgresql.org/docs/current/functions-aggregate.html, `count("any")`
    // "computes the number of input rows which the input value is not null."
    //
    // We can think of `SELECT COUNT(<expression>) FROM table_name` as
    // ```
    // SELECT COUNT(*)
    // FROM (
    //         SELECT <expression>
    //         FROM table
    //         WHERE <expression> IS NOT NULL
    //      )
    // ```
    // Since `count(<literal>)` will return the total number of rows regardless of NULL values in
    // other columns (as they are irrelevant), we can use `COUNT_ALL` for this case. However, we
    // have opted to use `COUNT_VALID` to handle cases like `count(NULL)` without having to
    // explicitly check for the literal type.
    //
    // Please note that `COUNT_ALL` may allow for better performance depending on the optimization
    // further down the pipeline. This can be implemented by checking Substrait's literal type.

    std::unique_ptr<gqe::expression> arg_expression;
    if (nargs == 0) {
      // Zero argument implies COUNT_ALL
      return std::make_pair(cudf::aggregation::COUNT_ALL,
                            std::make_unique<gqe::literal_expression<uint32_t>>(1));
    } else if (nargs == 1) {
      return std::make_pair(
        cudf::aggregation::COUNT_VALID,
        parse_expression(aggregate_function.arguments().Get(0).value(), subquery_relations));
    } else {
      throw std::runtime_error("SubstraitParser cannot parse aggregate function \"count\" with " +
                               std::to_string(nargs) + " arguments. Must have 0 or 1 argument.");
    }
  } else if (nargs == 1) {  // Aggregate functions with strictly 1 argument
    cudf::aggregation::Kind agg_kind;
    if (function_name == "avg") {
      agg_kind = cudf::aggregation::MEAN;
    } else if (function_name == "max") {
      agg_kind = cudf::aggregation::MAX;
    } else if (function_name == "median") {
      agg_kind = cudf::aggregation::MEDIAN;
    } else if (function_name == "min") {
      agg_kind = cudf::aggregation::MIN;
    } else if (function_name == "stddev") {
      agg_kind = cudf::aggregation::STD;
    } else if (function_name == "sum") {
      agg_kind = cudf::aggregation::SUM;
    } else if (function_name == "variance") {
      agg_kind = cudf::aggregation::VARIANCE;
    } else {
      throw std::runtime_error("SubstraitParser cannot parse aggregate function \"" +
                               function_name + "\"");
    }
    return std::make_pair(
      agg_kind,
      parse_expression(aggregate_function.arguments().Get(0).value(), subquery_relations));
  } else {  // Aggregate functions with more than 1 arguments
    throw std::runtime_error(
      "Aggregated function with more than 1 arguments is not yet supported. " +
      std::to_string(nargs) + " arguments were passed.");
  }
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_aggregate_relation(
  substrait::AggregateRel const& aggregate_relation) const
{
  // Parse input relation
  auto input_relation = parse_relation(aggregate_relation.input());

  // Parse aggregation key(s)
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  std::vector<std::unique_ptr<expression>> keys;
  if (aggregate_relation.groupings_size() > 1)
    throw std::runtime_error("Does not support groupby with multiple keys");

  auto grouping = aggregate_relation.groupings(0);
  for (auto& key_expression : grouping.grouping_expressions()) {
    keys.push_back(parse_expression(key_expression, subquery_relations));
  }

  // Parse value expressions
  // Translate measures to pairs of function name and value expression
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures;
  for (auto& measure : aggregate_relation.measures()) {
    if (measure.has_filter())
      throw std::runtime_error("Aggregate measure with filter not supported");
    measures.push_back(parse_aggregate_function(measure.measure(), subquery_relations));
  }

  return std::make_unique<gqe::logical::aggregate_relation>(
    std::move(input_relation), std::move(subquery_relations), std::move(keys), std::move(measures));
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_fetch_relation(
  substrait::FetchRel const& fetch_relation) const
{
  auto input_relation = parse_relation(fetch_relation.input());

  return std::make_unique<gqe::logical::fetch_relation>(
    std::move(input_relation), fetch_relation.offset(), fetch_relation.count());
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_sort_relation(
  substrait::SortRel const& sort_relation) const
{
  auto input_relation = parse_relation(sort_relation.input());

  std::vector<cudf::order> column_orders;
  std::vector<cudf::null_order> null_precedences;
  std::vector<std::unique_ptr<expression>> expressions;
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;

  const size_t num_sorts = sort_relation.sorts_size();
  column_orders.reserve(num_sorts);
  null_precedences.reserve(num_sorts);
  expressions.reserve(num_sorts);

  parse_sorts(
    sort_relation.sorts(), expressions, column_orders, null_precedences, subquery_relations);

  return std::make_unique<gqe::logical::sort_relation>(std::move(input_relation),
                                                       std::move(subquery_relations),
                                                       std::move(column_orders),
                                                       std::move(null_precedences),
                                                       std::move(expressions));
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_project_relation(
  substrait::ProjectRel const& project_relation) const
{
  // Extract `input` field from Substrait ProjectRel
  // Recursively parse input relation into logical relations
  auto input_relation = parse_relation(project_relation.input());

  std::vector<std::unique_ptr<gqe::expression>> output_expressions;
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  for (auto& expression : project_relation.expressions()) {
    // If the expression is WindowFunction, turn into a Window relation before returning and replace
    // the window expression with a column reference in this Projection's expression list.
    // Otherwise, add expression to the Projection's expression list.
    // TODO: Handle window functions nested in other expressions.
    //       For example, if the Substrait plan contains an expression in the form of `f(window0,
    //       window1)`. To handle relation nested in expression, we can most likely use the same
    //       logic as subquery relations.
    if (expression.has_window_function()) {
      input_relation = parse_window_function_expression(
        expression.window_function(), std::move(input_relation), subquery_relations);
      // Replace window function expression with column reference
      output_expressions.push_back(std::make_unique<gqe::column_reference_expression>(
        input_relation->data_types().size() - 1));
    } else {
      output_expressions.push_back(parse_expression(expression, subquery_relations));
    }
  }

  return std::make_unique<gqe::logical::project_relation>(
    std::move(input_relation), std::move(subquery_relations), std::move(output_expressions));
}

namespace {

// Helper function for calculating the number of columns in the join result
cudf::size_type num_columns_join_result(cudf::size_type num_columns_left,
                                        cudf::size_type num_columns_right,
                                        gqe::join_type_type join_type)
{
  if (join_type == gqe::join_type_type::inner || join_type == gqe::join_type_type::full ||
      join_type == gqe::join_type_type::left || join_type == gqe::join_type_type::single) {
    return num_columns_left + num_columns_right;
  } else if (join_type == gqe::join_type_type::left_semi ||
             join_type == gqe::join_type_type::left_anti) {
    return num_columns_left;
  } else {
    throw std::runtime_error("substrait_parser: Unsupported join type");
  }
}

}  // namespace

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_join_relation(
  substrait::JoinRel const& join_relation) const
{
  // Parse children relations
  auto left_relation  = parse_relation(join_relation.left());
  auto right_relation = parse_relation(join_relation.right());

  // Parse the join relation for the join type
  join_type_type join_type;
  switch (join_relation.type()) {
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_INNER:
      join_type = join_type_type::inner;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_OUTER:
      join_type = join_type_type::full;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT:
      join_type = join_type_type::left;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_SINGLE:
      join_type = join_type_type::single;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_SEMI:
      join_type = join_type_type::left_semi;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_ANTI:
      join_type = join_type_type::left_anti;
      break;
    default:
      throw std::runtime_error("SubstraitParser: unsupported join type " +
                               std::to_string(join_relation.type()));
  }

  // Check that the join relation has either join expression or join filter or both
  // Note that this check does not apply to cross join, however cross joins are not yet supported in
  // datafusion-substrait
  assert(join_relation.has_expression() || join_relation.has_post_join_filter());
  // Parse equi-join expression if exsits
  std::unique_ptr<gqe::expression> join_condition;
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  if (join_relation.has_expression()) {
    join_condition = parse_expression(join_relation.expression(), subquery_relations);
  } else {
    join_condition = std::make_unique<gqe::literal_expression<bool>>(true);
  }
  GQE_LOG_DEBUG("substrait parser: join condition: " + join_condition->to_string());
  // Construct projection indices
  // TODO: Configure projection indices from parent projection relation. For now,
  //       we'll return all columns and handle projection in a separate relation.
  auto const num_output_columns =
    num_columns_join_result(left_relation->num_columns(), right_relation->num_columns(), join_type);
  std::vector<cudf::size_type> projection_indices(num_output_columns);
  std::iota(projection_indices.begin(), projection_indices.end(), 0);
  // Construct join relation
  auto join = std::make_unique<gqe::logical::join_relation>(std::move(left_relation),
                                                            std::move(right_relation),
                                                            std::move(subquery_relations),
                                                            std::move(join_condition),
                                                            join_type,
                                                            std::move(projection_indices));
  // Wrap `join` in a filter relation if `post_join_filter` is not empty
  // As of Datafusion version 27.0.0 and Substrait version 0.31.0, there is no projection pushdown
  // field in join relations. Thus, the input column indices of the join relation and
  // `post_join_filter` correspond to the same list of fields. Note: Once projection pushdown (from
  // parent relation) is implemented, the field indices used in `post_join_filter`
  //       will need to reflect the changes
  if (join_relation.has_post_join_filter()) {
    auto join_filter = parse_expression(join_relation.post_join_filter(), subquery_relations);
    GQE_LOG_DEBUG("substrait parser: join filter: " + join_filter->to_string());
    return std::make_unique<gqe::logical::filter_relation>(
      std::move(join), std::move(subquery_relations), std::move(join_filter));
  }

  return join;
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_read_relation(
  substrait::ReadRel const& read_relation) const
{
  if (read_relation.read_type_case() != substrait::ReadRel::ReadTypeCase::kNamedTable)
    throw std::runtime_error("Only named table is supported for read relation");

  std::vector<size_t> projection_indices;

  if (read_relation.has_filter())
    throw std::runtime_error("Read relation does not support hard filter");

  if (read_relation.has_projection()) {
    if (!read_relation.projection().has_select())
      throw std::runtime_error("Read relation projection requires select field");
    auto select = read_relation.projection().select();
    for (substrait::Expression_MaskExpression_StructItem struct_item : select.struct_items()) {
      if (struct_item.has_child())
        throw std::runtime_error("Mask expression struct item with child is not supported");
      projection_indices.push_back(struct_item.field());
    }
  } else {
    projection_indices.resize(read_relation.base_schema().names().size());
    std::iota(projection_indices.begin(), projection_indices.end(), 0);
  }

  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  std::unique_ptr<expression> partial_filter = nullptr;
  // TODO: parse `best_effort_filter` when released by Substrait
  // (PR: https://github.com/substrait-io/substrait/pull/271)

  auto named_table = read_relation.named_table();
  assert(named_table.names_size() == 1);
  const std::string table_name = named_table.names(0);

  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;

  for (auto col_idx : projection_indices) {
    auto column_name = read_relation.base_schema().names()[col_idx];
    column_names.push_back(column_name);
    column_types.push_back(_catalog->column_type(table_name, column_name));
  }

  return std::make_unique<gqe::logical::read_relation>(std::move(subquery_relations),
                                                       std::move(column_names),
                                                       std::move(column_types),
                                                       std::move(table_name),
                                                       std::move(partial_filter));
}
