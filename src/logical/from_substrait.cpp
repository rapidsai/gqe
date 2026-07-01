/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/logical/from_substrait.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/fixed_point/conv.hpp>

#include <ddl_extension.pb.h>
#include <google/protobuf/any.pb.h>
#include <substrait/algebra.pb.h>

#include <atomic>
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

std::vector<std::shared_ptr<gqe::logical::relation>> gqe::substrait_parser::from_plan(
  substrait::Plan& query_plan)
{
  // Register functions
  for (auto& extension : query_plan.extensions()) {
    assert(extension.mapping_type_case() ==
           substrait::extensions::SimpleExtensionDeclaration::MappingTypeCase::kExtensionFunction);
    auto function_extension = extension.extension_function();

    add_function_reference(function_extension.function_anchor(), function_extension.name());
  }

  // Atomic counter for unique temporary table names
  static std::atomic<uint64_t> temp_table_counter{0};

  // Parse relation trees
  std::vector<std::shared_ptr<gqe::logical::relation>> relation_trees;

  for (auto relation : query_plan.relations()) {
    // Get the top-level Rel
    substrait::Rel const* top_rel = nullptr;
    if (!relation.has_root()) {
      if (!relation.has_rel()) throw std::invalid_argument("Top level PlanRel has no root or rel");
      top_rel = &relation.rel();
    } else {
      top_rel = &relation.root().input();
    }

    if (top_rel->has_ddl()) {
      // Handle DDL: parse and execute directly
      auto cmd = parse_ddl_command(top_rel->ddl());
      switch (cmd.op) {
        case ddl_command::operation::create_or_replace_table:
        case ddl_command::operation::create_table: {
          if (cmd.op == ddl_command::operation::create_or_replace_table &&
              _catalog->has_table(cmd.table_name)) {
            _catalog->unregister_table(cmd.table_name);
          }
          std::vector<column_traits> columns;
          columns.reserve(cmd.column_names.size());
          for (std::size_t i = 0; i < cmd.column_names.size(); ++i) {
            columns.emplace_back(cmd.column_names[i], cmd.column_types[i]);
          }
          // Translate key index-lists → column-name lists
          std::vector<std::vector<std::string>> unique_keys;
          unique_keys.reserve(cmd.unique_keys.size());
          for (auto const& key_indices : cmd.unique_keys) {
            std::vector<std::string> key_names;
            key_names.reserve(key_indices.size());
            for (auto idx : key_indices) {
              key_names.push_back(cmd.column_names[idx]);
            }
            unique_keys.push_back(std::move(key_names));
          }
          // File-backed storage supports partitioning inference; in-memory storage doesn't.
          // Add new file-backed variants here as they're introduced.
          auto is_file_backed = [](storage_kind::type const& s) {
            return std::holds_alternative<storage_kind::parquet_file>(s);
          };
          partitioning_schema_kind::type partitioning =
            is_file_backed(cmd.storage)
              ? partitioning_schema_kind::type{partitioning_schema_kind::automatic{}}
              : partitioning_schema_kind::type{partitioning_schema_kind::none{}};
          _catalog->register_table(cmd.table_name, columns, cmd.storage, partitioning, unique_keys);
          GQE_LOG_INFO("DDL: Created table '{}'", cmd.table_name);
          break;
        }
        case ddl_command::operation::drop_table:
          _catalog->unregister_table(cmd.table_name);
          GQE_LOG_INFO("DDL: Dropped table '{}'", cmd.table_name);
          break;
        case ddl_command::operation::drop_table_if_exists:
          if (_catalog->has_table(cmd.table_name)) {
            _catalog->unregister_table(cmd.table_name);
            GQE_LOG_INFO("DDL: Dropped table '{}'", cmd.table_name);
          } else {
            GQE_LOG_INFO("DDL: Table '{}' does not exist, skipping DROP", cmd.table_name);
          }
          break;
      }
      // DDL produces no relation tree
    } else if (top_rel->has_write()) {
      // Handle Write: register temp parquet source, build read->write plan
      auto cmd = parse_write_command(top_rel->write());

      // Register temporary parquet source table
      auto temp_name  = std::format("__pq_{}_{}", cmd.table_name, temp_table_counter.fetch_add(1));
      auto file_paths = gqe::utility::get_parquet_files(cmd.file_path);
      std::vector<column_traits> pq_columns;
      pq_columns.reserve(cmd.column_names.size());
      for (std::size_t i = 0; i < cmd.column_names.size(); ++i) {
        pq_columns.emplace_back(cmd.column_names[i], cmd.column_types[i]);
      }
      _catalog->register_table(temp_name,
                               pq_columns,
                               storage_kind::parquet_file{file_paths},
                               partitioning_schema_kind::automatic{});

      // Build read_relation from temporary parquet source
      auto read = std::make_unique<gqe::logical::read_relation>(
        std::vector<std::shared_ptr<gqe::logical::relation>>{},
        cmd.column_names,
        cmd.column_types,
        temp_name,
        nullptr);

      // Build write_relation to destination table
      auto write = std::make_unique<gqe::logical::write_relation>(
        std::move(read), cmd.column_names, cmd.column_types, cmd.table_name);

      relation_trees.push_back(std::move(write));
      GQE_LOG_INFO("Write: COPY into '{}' from '{}'", cmd.table_name, cmd.file_path);
    } else {
      // Normal query relation
      relation_trees.push_back(parse_relation(*top_rel));
    }
  }

  for (auto const& relation : relation_trees) {
    GQE_LOG_TRACE("Imported a query plan: \n {}", relation->to_string());
  }

  return relation_trees;
}

std::vector<std::shared_ptr<gqe::logical::relation>> gqe::substrait_parser::from_file(
  std::string substrait_file)
{
  const std::string file_ext = ".bin";
  assert(mismatch(file_ext.rbegin(), file_ext.rend(), substrait_file.rbegin()).first ==
         file_ext.rend());
  if (!std::filesystem::exists(substrait_file))
    throw std::runtime_error(std::format("Substrait file {} does not exist", substrait_file));
  std::ifstream query_plan_stream(substrait_file, std::ios::binary);
  substrait::Plan query_plan;
  query_plan.ParseFromIstream(&query_plan_stream);

  return from_plan(query_plan);
}

std::vector<std::shared_ptr<gqe::logical::relation>> gqe::substrait_parser::from_binary(
  const void* data, std::size_t size)
{
  substrait::Plan query_plan;
  if (!query_plan.ParseFromArray(data, static_cast<int>(size))) {
    throw std::invalid_argument("Failed to parse Substrait plan from binary data");
  }

  return from_plan(query_plan);
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
    throw std::invalid_argument("SubstraitParser cannot parse expression with type " +
                                std::to_string(expression.rex_type_case()));
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_if_then_expression(
  substrait::Expression_IfThen const& if_then_expression,
  std::vector<std::shared_ptr<gqe::logical::relation>>& subquery_relations) const
{
  auto const num_ifs = if_then_expression.ifs_size();

  if (num_ifs == 0) {
    throw std::invalid_argument("IfThen expression must have at least one if clause");
  }

  // Parse the ELSE expression (default value)
  auto else_expr = parse_expression(if_then_expression.else_(), subquery_relations);

  // Build nested if_then_else expressions from right to left (innermost to outermost)
  // For CASE WHEN c1 THEN v1 WHEN c2 THEN v2 WHEN c3 THEN v3 ELSE default:
  // Build: IF c1 THEN v1 ELSE (IF c2 THEN v2 ELSE (IF c3 THEN v3 ELSE default))
  std::unique_ptr<gqe::expression> current_else = std::move(else_expr);

  // Process if clauses in reverse order to build nested structure
  for (int i = num_ifs - 1; i >= 0; --i) {
    auto if_expr   = parse_expression(if_then_expression.ifs(i).if_(), subquery_relations);
    auto then_expr = parse_expression(if_then_expression.ifs(i).then(), subquery_relations);

    // Create nested if_then_else: IF condition THEN value ELSE (previous_else)
    current_else = std::make_unique<gqe::if_then_else_expression>(
      std::move(if_expr), std::move(then_expr), std::move(current_else));
  }

  return current_else;
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
      GQE_LOG_WARN("Use FLOAT64 to represent a decimal literal");
      return std::make_unique<gqe::literal_expression<double>>(
        cudf::convert_fixed_to_floating<double>(fixed_point_value));
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
        throw std::invalid_argument(
          "SubstraitParser cannot parse null literal expression with type " +
          std::to_string(literal_expression.null().kind_case()));

      return null_literal;
    }
    default:
      throw std::invalid_argument("SubstraitParser cannot parse literal expression with type " +
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
    case substrait::Type::kDate: return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case substrait::Type::kDecimal: {
      // Currently GQE uses floating point types to represent decimal types.
      // TODO: Properly support decimal types.
      /*
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
      */
      GQE_LOG_WARN("Use FLOAT64 to represent decimal type");
      return cudf::data_type(cudf::type_id::FLOAT64);
    }
    default:
      throw std::invalid_argument("SubstraitParser cannot convert substrait type " +
                                  std::to_string(substrait_type.kind_case()) + " to cuDF type");
  }
}

// Helper function to translate datetime component string into
// gqe::date_part_expression::datetime_component
cudf::datetime::datetime_component datetime_component_from_str(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s == "year") return cudf::datetime::datetime_component::YEAR;
  if (s == "month") return cudf::datetime::datetime_component::MONTH;
  if (s == "day") return cudf::datetime::datetime_component::DAY;
  if (s == "weekday") return cudf::datetime::datetime_component::WEEKDAY;
  if (s == "hour") return cudf::datetime::datetime_component::HOUR;
  if (s == "minute") return cudf::datetime::datetime_component::MINUTE;
  if (s == "second") return cudf::datetime::datetime_component::SECOND;
  if (s == "millisecond") return cudf::datetime::datetime_component::MILLISECOND;
  if (s == "nanosecond")
    return cudf::datetime::datetime_component::NANOSECOND;
  else
    throw std::invalid_argument("Unsupported datetime component string: " + s);
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
    {"substring", gqe::scalar_function_expression::function_kind::substr}};
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
  throw std::invalid_argument(
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
    throw std::invalid_argument("Only direct reference is supported for selection expressions");

  if (!selection_expression.direct_reference().has_struct_field())
    throw std::invalid_argument("Only struct field is supported for selection expressions");

  auto struct_field = selection_expression.direct_reference().struct_field();

  if (struct_field.has_child())
    throw std::invalid_argument("Does not support struct field with child");

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
          throw std::invalid_argument("like/ilike() expects 2 or 3 arguments. Got " +
                                      std::to_string(nargs) + " arguments");
        auto input   = parsed_arguments[0];
        auto pattern = try_get_literal_value<std::string>(parsed_arguments[1]);
        auto escape_char =
          nargs == 3 ? try_get_literal_value<std::string>(parsed_arguments[2]) : std::string{};
        return std::make_unique<gqe::like_expression>(
          std::move(input), pattern, escape_char, function_name == "ilike" ? true : false);
      }
      case gqe::scalar_function_expression::function_kind::round: {
        if (nargs != 2)
          throw std::invalid_argument("round() currently only supports 2 arguments. Got " +
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
        return std::make_unique<gqe::substr_expression>(std::move(input), start - 1, length);
      }
      default:
        throw std::invalid_argument("ScalarFunction " + function_name + "() is not supported");
    }
  } else {
    // If the function is not part of the supported scalar_function::function_kind types,
    // attempt to parse as a cudf supported operation
    if (nargs == 1) {
      auto input = parse_expression(arg_expressions[0], subquery_relations);
      if (function_name == "not")
        return std::make_unique<gqe::not_expression>(std::move(input));
      else if (function_name == "is_null")
        return std::make_unique<gqe::is_null_expression>(std::move(input));
      else if (function_name == "is_not_null")
        return std::make_unique<gqe::not_expression>(
          std::move(std::make_unique<gqe::is_null_expression>(std::move(input))));
      else
        throw std::invalid_argument("SubstraitParser cannot parse unary scalar function \"" +
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
      else if (function_name == "subtract")
        return std::make_unique<gqe::subtract_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "multiply")
        return std::make_unique<gqe::multiply_expression>(std::move(lhs), std::move(rhs));
      else if (function_name == "divide")
        return std::make_unique<gqe::divide_expression>(std::move(lhs), std::move(rhs));
      else
        throw std::invalid_argument("SubstraitParser cannot parse binary scalar function \"" +
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
            throw std::invalid_argument("Cannot find matching multi-argument scalar function \"" +
                                        function_name + "\"" + " with " + std::to_string(nargs) +
                                        " arguments");
        });
      return output_expr;
    } else {
      throw std::invalid_argument(
        "Cannot find matching ScalarFunction with less than 1 arguments: " + function_name);
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
  else if (relation.has_set())
    return parse_set_relation(relation.set());
  else
    throw std::invalid_argument("Unsupported relation type");
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
    throw std::invalid_argument("Unsupported bound type " + std::to_string(bound.kind_case()));
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
      default: throw std::invalid_argument("Unsupported SortField_SortDirection");
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
    throw std::invalid_argument("SubstraitParser cannot parse aggr/window function \"" +
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

  auto const num_output_columns = input_relation->num_columns();
  std::vector<cudf::size_type> projection_indices(num_output_columns);
  std::iota(projection_indices.begin(), projection_indices.end(), 0);

  return std::make_unique<gqe::logical::filter_relation>(std::move(input_relation),
                                                         std::move(subquery_relations),
                                                         std::move(condition),
                                                         std::move(projection_indices));
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
    // `count(*)` is often `nargs == 1` (e.g. DataFusion encodes it as `count(1)`). Use `COUNT_ALL`
    // for that and for any other non-null literal: SQL `COUNT(<non-null constant>)` matches row
    // count, same as `COUNT(*)`. Only `count(null)` must use `COUNT_VALID` (Substrait encodes the
    // argument as a null literal).

    if (nargs == 0) {
      // Zero argument implies COUNT_ALL. Use int64_t: expression visitors/eval (eval.cpp, etc.)
      // only specialize a fixed set of literal types, not e.g. uint32_t.
      return std::make_pair(cudf::aggregation::COUNT_ALL,
                            std::make_unique<gqe::literal_expression<int64_t>>(1));
    } else if (nargs == 1) {
      substrait::Expression const& arg0 = aggregate_function.arguments().Get(0).value();
      if (arg0.has_literal() && arg0.literal().literal_type_case() !=
                                  substrait::Expression_Literal::LiteralTypeCase::kNull) {
        return std::make_pair(cudf::aggregation::COUNT_ALL,
                              std::make_unique<gqe::literal_expression<int64_t>>(1));
      }
      auto arg_expr = parse_expression(arg0, subquery_relations);
      return std::make_pair(cudf::aggregation::COUNT_VALID, std::move(arg_expr));
    } else {
      throw std::invalid_argument(
        "SubstraitParser cannot parse aggregate function \"count\" with " + std::to_string(nargs) +
        " arguments. Must have 0 or 1 argument.");
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
      throw std::invalid_argument("SubstraitParser cannot parse aggregate function \"" +
                                  function_name + "\"");
    }
    return std::make_pair(
      agg_kind,
      parse_expression(aggregate_function.arguments().Get(0).value(), subquery_relations));
  } else {  // Aggregate functions with more than 1 arguments
    throw std::invalid_argument(
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
    throw std::invalid_argument("Does not support groupby with multiple keys");

  auto grouping            = aggregate_relation.groupings(0);
  auto groupingExpressions = aggregate_relation.grouping_expressions();
  for (const auto expression_reference : grouping.expression_references()) {
    auto key_expression = groupingExpressions.at(expression_reference);
    keys.push_back(parse_expression(key_expression, subquery_relations));
  }

  // Parse value expressions
  // Translate measures to pairs of function name and value expression
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures;
  for (auto& measure : aggregate_relation.measures()) {
    if (measure.has_filter())
      throw std::invalid_argument("Aggregate measure with filter not supported");
    measures.push_back(parse_aggregate_function(measure.measure(), subquery_relations));
  }

  return std::make_unique<gqe::logical::aggregate_relation>(
    std::move(input_relation), std::move(subquery_relations), std::move(keys), std::move(measures));
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_fetch_relation(
  substrait::FetchRel const& fetch_relation) const
{
  auto input_relation = parse_relation(fetch_relation.input());

  // Helper to retrieve the integer value from an OFFSET or LIMIT expression
  auto fold_expression = [this](const substrait::Expression& expression) -> int64_t {
    auto subquery_relations      = std::vector<std::shared_ptr<gqe::logical::relation>>{};
    const auto parsed_expression = parse_expression(expression, subquery_relations);
    if (const auto integer_literal_expression =
          dynamic_cast<gqe::literal_expression<int64_t>*>(parsed_expression.get())) {
      return integer_literal_expression->value();
    }
    throw std::invalid_argument(
      "FetchRel.offset_expr/FetchRel.count_expr is only supported for literal expressions");
  };

  int64_t offset = 0;   // Unset is treated as 0
  int64_t count  = -1;  // Unset is treated as ALL
  switch (fetch_relation.offset_mode_case()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    case substrait::FetchRel::kOffset: offset = fetch_relation.offset(); break;
#pragma GCC diagnostic pop
    case substrait::FetchRel::kOffsetExpr:
      offset = fold_expression(fetch_relation.offset_expr());
      break;
    case substrait::FetchRel::OFFSET_MODE_NOT_SET: break;  // Use default
  }
  switch (fetch_relation.count_mode_case()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    case substrait::FetchRel::kCount: count = fetch_relation.count(); break;
#pragma GCC diagnostic pop
    case substrait::FetchRel::kCountExpr:
      count = fold_expression(fetch_relation.count_expr());
      break;
    case substrait::FetchRel::COUNT_MODE_NOT_SET: break;  // Use default
  }
  return std::make_unique<gqe::logical::fetch_relation>(std::move(input_relation), offset, count);
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
    throw std::invalid_argument("substrait_parser: Unsupported join type");
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
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT_SINGLE:
      join_type = join_type_type::single;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT_SEMI:
      join_type = join_type_type::left_semi;
      break;
    case substrait::JoinRel_JoinType::JoinRel_JoinType_JOIN_TYPE_LEFT_ANTI:
      join_type = join_type_type::left_anti;
      break;
    default:
      throw std::invalid_argument("SubstraitParser: unsupported join type " +
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
    std::vector<cudf::size_type> filter_projection_indices(num_output_columns);
    std::iota(filter_projection_indices.begin(), filter_projection_indices.end(), 0);
    return std::make_unique<gqe::logical::filter_relation>(std::move(join),
                                                           std::move(subquery_relations),
                                                           std::move(join_filter),
                                                           std::move(filter_projection_indices));
  }

  return join;
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_read_relation(
  substrait::ReadRel const& read_relation) const
{
  if (read_relation.read_type_case() != substrait::ReadRel::ReadTypeCase::kNamedTable)
    throw std::invalid_argument("Only named table is supported for read relation");

  std::vector<size_t> projection_indices;

  if (read_relation.has_projection()) {
    if (!read_relation.projection().has_select())
      throw std::invalid_argument("Read relation projection requires select field");
    auto select                  = read_relation.projection().select();
    auto const base_column_count = read_relation.base_schema().names_size();
    for (substrait::Expression_MaskExpression_StructItem struct_item : select.struct_items()) {
      if (struct_item.has_child())
        throw std::invalid_argument("Mask expression struct item with child is not supported");
      if (struct_item.field() < 0 || struct_item.field() >= base_column_count)
        throw std::invalid_argument(
          std::format("ReadRel: projection field {} out of range (base schema has {} columns)",
                      struct_item.field(),
                      base_column_count));
      projection_indices.push_back(struct_item.field());
    }
  } else {
    projection_indices.resize(read_relation.base_schema().names().size());
    std::iota(projection_indices.begin(), projection_indices.end(), 0);
  }

  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  std::unique_ptr<expression> partial_filter =
    read_relation.has_filter() ? parse_expression(read_relation.filter(), subquery_relations)
                               : nullptr;

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

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_set_relation(
  substrait::SetRel const& set_relation) const
{
  assert(set_relation.inputs_size() > 1);
  // Parse inputs
  std::vector<std::unique_ptr<gqe::logical::relation>> inputs;
  inputs.reserve(set_relation.inputs_size());
  std::transform(set_relation.inputs().begin(),
                 set_relation.inputs().end(),
                 std::back_inserter(inputs),
                 [this](substrait::Rel input) { return parse_relation(input); });
  // Parse set operation
  gqe::logical::set_relation::set_operator_type op;
  switch (set_relation.op()) {
    case substrait::SetRel::SetOp::SetRel_SetOp_SET_OP_INTERSECTION_PRIMARY:
      op = gqe::logical::set_relation::set_intersect;
      break;
    case substrait::SetRel::SetOp::SetRel_SetOp_SET_OP_UNION_ALL:
      op = gqe::logical::set_relation::set_union_all;
      break;
    case substrait::SetRel::SetOp::SetRel_SetOp_SET_OP_UNION_DISTINCT:
      op = gqe::logical::set_relation::set_union;
      break;
    case substrait::SetRel::SetOp::SetRel_SetOp_SET_OP_MINUS_PRIMARY:
      op = gqe::logical::set_relation::set_minus;
      break;
    default:
      throw std::invalid_argument("Set operation not supported: " +
                                  substrait::SetRel_SetOp_Name(set_relation.op()));
  }
  // Turn op and list of inputs into (nested) set relation
  auto set = std::move(inputs[0]);
  for (std::size_t input_idx = 1; input_idx < inputs.size(); input_idx++) {
    set = std::make_unique<gqe::logical::set_relation>(
      std::move(set), std::move(inputs[input_idx]), op);
  }
  return set;
}

namespace {

struct decoded_extension {
  std::vector<std::vector<std::size_t>> unique_keys;
  gqe::storage_kind::type storage = gqe::storage_kind::boost_shared_memory{};
};

// Parse a page_kind string into gqe::page_kind::type.
// Throws std::invalid_argument for unrecognised values.
gqe::page_kind::type parse_page_kind(std::string const& s)
{
  auto const lower = gqe::utility::to_lower(s);
  if (lower == "system_default") return gqe::page_kind::system_default;
  if (lower == "small") return gqe::page_kind::small;
  if (lower == "transparent_huge") return gqe::page_kind::transparent_huge;
  if (lower == "huge2mb") return gqe::page_kind::huge2mb;
  if (lower == "huge1gb") return gqe::page_kind::huge1gb;
  throw std::invalid_argument(
    std::format("WITH: unknown page_kind '{}'. Valid values: system_default, small, "
                "transparent_huge, huge2mb, huge1gb",
                s));
}

// Parse the typed storage_options map from a WITH (...) clause into a concrete
// storage_kind::type variant.  Validates that:
//   - "storage_kind" names a known variant
//   - no unrecognised keys appear alongside the chosen variant
//
// When an optional per-variant parameter is omitted, the variant's default constructor is used.
//
// Defaults to boost_shared_memory when the map is empty or has no "storage_kind" key.
gqe::storage_kind::type parse_storage_kind(
  google::protobuf::Map<std::string, gqe::proto::StorageOptionValue> const& opts)
{
  // Extract a string from a typed option value.
  auto get_string = [](gqe::proto::StorageOptionValue const& v,
                       std::string_view key) -> std::string {
    switch (v.value_case()) {
      case gqe::proto::StorageOptionValue::kStringVal: return v.string_val();
      case gqe::proto::StorageOptionValue::kIntVal: return std::to_string(v.int_val());
      default: throw std::invalid_argument(std::format("WITH: '{}' must be a string value", key));
    }
  };

  // Extract a non-negative integer that fits in int from a typed option value.
  // Only IntVal is accepted; quoted-integer SQL syntax (e.g. device_id = '3') is rejected
  // with a message pointing the user at the unquoted form.
  auto get_non_negative_int = [](gqe::proto::StorageOptionValue const& v,
                                 std::string_view key) -> int {
    switch (v.value_case()) {
      case gqe::proto::StorageOptionValue::kIntVal:
        return gqe::utility::checked_cast<int>(v.int_val(), std::format("WITH: '{}'", key), 0);
      case gqe::proto::StorageOptionValue::kStringVal:
        throw std::invalid_argument(std::format(
          "WITH: '{}' must be an unquoted integer (e.g. {} = 3, not {} = '3')", key, key, key));
      default: throw std::invalid_argument(std::format("WITH: '{}' must be an integer value", key));
    }
  };

  // Extract a cpu_set from a typed option value; only IntListVal (ARRAY[...]) is accepted.
  auto get_cpu_set = [](gqe::proto::StorageOptionValue const& v,
                        std::string_view key) -> gqe::cpu_set {
    if (v.value_case() != gqe::proto::StorageOptionValue::kIntListVal)
      throw std::invalid_argument(
        std::format("WITH: '{}' must be an integer list, e.g. ARRAY[0, 1]", key));
    gqe::cpu_set result;
    for (int64_t id : v.int_list_val().values()) {
      result.add(gqe::utility::checked_cast<int>(id, std::format("WITH: '{}'", key), 0));
    }
    return result;
  };

  auto const kind_it     = opts.find("storage_kind");
  std::string const kind = (kind_it != opts.end())
                             ? gqe::utility::to_lower(get_string(kind_it->second, "storage_kind"))
                             : "boost_shared_memory";

  // Validate that every key other than "storage_kind" belongs to `allowed`.
  auto check_extra_keys = [&](std::initializer_list<std::string_view> allowed) {
    for (auto const& [k, v] : opts) {
      if (k == "storage_kind") continue;
      bool found = false;
      for (auto a : allowed)
        if (k == a) {
          found = true;
          break;
        }
      if (!found)
        throw std::invalid_argument(
          std::format("WITH: option '{}' is not valid for storage_kind '{}'", k, kind));
    }
  };

  if (kind == "boost_shared_memory") {
    check_extra_keys({});
    return gqe::storage_kind::boost_shared_memory{};
  }
  if (kind == "system_memory") {
    check_extra_keys({});
    return gqe::storage_kind::system_memory{};
  }
  if (kind == "pinned_memory") {
    check_extra_keys({});
    return gqe::storage_kind::pinned_memory{};
  }
  if (kind == "managed_memory") {
    check_extra_keys({});
    return gqe::storage_kind::managed_memory{};
  }
  if (kind == "device_memory") {
    check_extra_keys({"device_id"});
    if (auto it = opts.find("device_id"); it != opts.end()) {
      return gqe::storage_kind::device_memory{
        rmm::cuda_device_id{get_non_negative_int(it->second, "device_id")}};
    }
    // device_id omitted: default-construct, which selects the current CUDA device.
    return gqe::storage_kind::device_memory{};
  }
  if (kind == "numa_pool_memory") {
    check_extra_keys({"numa_node_id"});
    if (auto it = opts.find("numa_node_id"); it != opts.end()) {
      return gqe::storage_kind::numa_pool_memory{get_non_negative_int(it->second, "numa_node_id")};
    }
    // numa_node_id omitted: use the constructor that auto-selects based on the current
    // CUDA device's memory affinity.
    return gqe::storage_kind::numa_pool_memory{};
  }
  if (kind == "shared_numa_pool_memory") {
    check_extra_keys({"numa_node_id"});
    if (auto it = opts.find("numa_node_id"); it != opts.end()) {
      return gqe::storage_kind::shared_numa_pool_memory{
        get_non_negative_int(it->second, "numa_node_id")};
    }
    // numa_node_id omitted: use the constructor that auto-selects based on the current
    // CUDA device's memory affinity.
    return gqe::storage_kind::shared_numa_pool_memory{};
  }
  if (kind == "numa_memory" || kind == "numa_pinned_memory") {
    check_extra_keys({"numa_node_set", "page_kind"});

    gqe::page_kind::type pk = gqe::page_kind::system_default;
    if (auto pk_it = opts.find("page_kind"); pk_it != opts.end())
      pk = parse_page_kind(get_string(pk_it->second, "page_kind"));

    if (auto ns_it = opts.find("numa_node_set"); ns_it != opts.end()) {
      gqe::cpu_set node_set = get_cpu_set(ns_it->second, "numa_node_set");
      if (kind == "numa_memory") return gqe::storage_kind::numa_memory{node_set, pk};
      return gqe::storage_kind::numa_pinned_memory{node_set, pk};
    }
    // numa_node_set omitted: use the constructor that auto-selects based on the current
    // CUDA device's memory affinity.
    if (kind == "numa_memory") return gqe::storage_kind::numa_memory{pk};
    return gqe::storage_kind::numa_pinned_memory{pk};
  }
  if (kind == "parquet_file") {
    check_extra_keys({"location"});
    auto it = opts.find("location");
    if (it == opts.end())
      throw std::invalid_argument(
        "storage_kind 'parquet_file' requires a 'location' "
        "(set via CREATE EXTERNAL TABLE ... LOCATION '...')");
    // A directory that the server recursively scans for the .parquet files under it.
    auto location = get_string(it->second, "location");
    return gqe::storage_kind::parquet_file{gqe::utility::get_parquet_files(std::move(location))};
  }

  throw std::invalid_argument(std::format(
    "WITH: unknown storage_kind '{}'. Valid values: boost_shared_memory, system_memory, "
    "pinned_memory, managed_memory, device_memory, numa_pool_memory, "
    "shared_numa_pool_memory, numa_memory, numa_pinned_memory, parquet_file",
    kind));
}

// Decode the gqe.proto.CreateTableExtension from a google.protobuf.Any enhancement
// field. Returns decoded unique-key column-index lists and the resolved storage_kind.
// Returns defaults when the type_url doesn't match or the payload can't be parsed.
//
// Throws std::invalid_argument when the extension IS a CreateTableExtension, parses
// successfully, but contains a malformed UNIQUE/PRIMARY KEY (empty column_indices
// or an out-of-range index).
decoded_extension decode_create_table_extension(google::protobuf::Any const& any,
                                                std::size_t num_columns)
{
  constexpr std::string_view type_url = "type.googleapis.com/gqe.proto.CreateTableExtension";
  if (any.type_url() != type_url) return {};

  gqe::proto::CreateTableExtension ext;
  if (!ext.ParseFromString(any.value())) return {};

  decoded_extension result;

  result.unique_keys.reserve(ext.unique_keys_size());
  for (auto const& key : ext.unique_keys()) {
    if (key.column_indices_size() == 0) {
      throw std::invalid_argument(
        "DdlRel: CreateTableExtension contains a unique key with no column indices");
    }
    std::vector<std::size_t> key_indices;
    key_indices.reserve(key.column_indices_size());
    for (auto idx : key.column_indices()) {
      if (idx >= num_columns) {
        throw std::invalid_argument(
          std::format("DdlRel: CreateTableExtension unique-key column index {} out of range "
                      "(table has {} columns)",
                      idx,
                      num_columns));
      }
      key_indices.push_back(static_cast<std::size_t>(idx));
    }
    result.unique_keys.push_back(std::move(key_indices));
  }

  result.storage = parse_storage_kind(ext.storage_options());

  return result;
}

}  // namespace

gqe::ddl_command gqe::substrait_parser::parse_ddl_command(substrait::DdlRel const& ddl) const
{
  ddl_command cmd;

  // Extract table name
  if (!ddl.has_named_object())
    throw std::invalid_argument("DdlRel: only named_object write_type is supported");
  auto const& names = ddl.named_object().names();
  if (names.empty()) throw std::invalid_argument("DdlRel: named_object has no names");
  cmd.table_name = names[0];

  // Map operation
  switch (ddl.op()) {
    case substrait::DdlRel::DDL_OP_CREATE: cmd.op = ddl_command::operation::create_table; break;
    case substrait::DdlRel::DDL_OP_CREATE_OR_REPLACE:
      cmd.op = ddl_command::operation::create_or_replace_table;
      break;
    case substrait::DdlRel::DDL_OP_DROP: cmd.op = ddl_command::operation::drop_table; break;
    case substrait::DdlRel::DDL_OP_DROP_IF_EXIST:
      cmd.op = ddl_command::operation::drop_table_if_exists;
      break;
    default:
      throw std::invalid_argument(
        std::format("DdlRel: unsupported DDL operation {}", std::to_string(ddl.op())));
  }

  // For CREATE / CREATE OR REPLACE, parse table schema
  if ((cmd.op == ddl_command::operation::create_table ||
       cmd.op == ddl_command::operation::create_or_replace_table) &&
      ddl.has_table_schema()) {
    auto const& named_struct = ddl.table_schema();
    auto const& struct_type  = named_struct.struct_();
    for (int i = 0; i < named_struct.names_size(); ++i) {
      cmd.column_names.push_back(named_struct.names(i));
      cmd.column_types.push_back(substrait_to_cudf_type(struct_type.types(i)));
    }

    // Decode GQE DDL extension carrying UNIQUE / PRIMARY KEY and storage_options (if present).
    if (ddl.has_advanced_extension() && ddl.advanced_extension().has_enhancement()) {
      auto ext        = decode_create_table_extension(ddl.advanced_extension().enhancement(),
                                               cmd.column_names.size());
      cmd.unique_keys = std::move(ext.unique_keys);
      cmd.storage     = ext.storage;
    }
  }

  return cmd;
}

gqe::write_command gqe::substrait_parser::parse_write_command(
  substrait::WriteRel const& write) const
{
  write_command cmd;

  // Extract table name
  if (!write.has_named_table())
    throw std::invalid_argument("WriteRel: only named_table write_type is supported");
  auto const& names = write.named_table().names();
  if (names.empty()) throw std::invalid_argument("WriteRel: named_table has no names");
  cmd.table_name = names[0];

  // Extract file path from the input ReadRel's LocalFiles
  if (!write.has_input() || !write.input().has_read())
    throw std::invalid_argument("WriteRel: expected input ReadRel");
  auto const& read_rel = write.input().read();
  if (!read_rel.has_local_files() || read_rel.local_files().items_size() == 0)
    throw std::invalid_argument("WriteRel: expected LocalFiles in input ReadRel");
  auto const& file_item = read_rel.local_files().items(0);
  cmd.file_path         = file_item.uri_path();

  // Validate file format is parquet
  if (!file_item.has_parquet())
    throw std::invalid_argument(
      std::format("WriteRel: only parquet format is supported, got file_format_case={}",
                  std::to_string(file_item.file_format_case())));

  // Parse table schema
  if (write.has_table_schema()) {
    auto const& named_struct = write.table_schema();
    auto const& struct_type  = named_struct.struct_();
    for (int i = 0; i < named_struct.names_size(); ++i) {
      cmd.column_names.push_back(named_struct.names(i));
      cmd.column_types.push_back(substrait_to_cudf_type(struct_type.types(i)));
    }
  }

  return cmd;
}
