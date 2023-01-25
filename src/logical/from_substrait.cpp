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

#include <cstddef>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>

#include <optional>
#include <string>
#include <substrait/algebra.pb.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
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

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_literal_expression(
  substrait::Expression_Literal const& literal_expression) const
{
  // TODO:
  // - Support
  //  - substrait::Expression_Literal::LiteralTypeCase::kDecimal
  //  - substrait::Expression_Literal::LiteralTypeCase::kNull
  //  - substrait::Expression_Literal::LiteralTypeCase::kDate
  // - Add unit test for all cases. For now, only i32 has been validated with real plan
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
    default:
      throw std::runtime_error("SubstraitParser cannot parse literal expression with type " +
                               std::to_string(literal_expression.literal_type_case()));
  }
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
  if (nargs == 2) {  // Binary function base case
    auto lhs = parse_expression(arg_expressions[0], subquery_relations);
    auto rhs = parse_expression(arg_expressions[1], subquery_relations);

    if (function_name == "equal" || function_name == "equal:any_any")
      return std::make_unique<gqe::equal_expression>(std::move(lhs), std::move(rhs));
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
    else
      throw std::runtime_error("SubstraitParser cannot parse binary scalar function \"" +
                               function_name + "\"");
  } else if (nargs > 2) {  // Multi-argument function recursive case
    // Only associative functions are implemented here
    auto output_expr = parse_expression(arg_expressions[0], subquery_relations);
    std::for_each(arg_expressions.begin() + 1,
                  arg_expressions.end(),
                  [&](substrait::Expression const& input_expr) {
                    auto rhs = parse_expression(input_expr, subquery_relations);
                    if (function_name == "and" || function_name == "and:bool")
                      output_expr = std::make_unique<gqe::logical_and_expression>(
                        std::move(output_expr), std::move(rhs));
                    else if (function_name == "or" || function_name == "or:bool")
                      output_expr = std::make_unique<gqe::logical_or_expression>(
                        std::move(output_expr), std::move(rhs));
                    else
                      throw std::runtime_error(
                        "SubstraitParser cannot parse multi-argument scalar function \"" +
                        function_name + "\"");
                  });
    return output_expr;
  } else {
    throw std::runtime_error("ScalarFunction with less than 2 arguments is not supported");
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

  if (function_name == "count" || function_name == "count:opt" ||
      function_name == "count:opt_any") {
    // Different Substrait producer encode `count(*)` differently. Cases encountered so far:
    // - DataFusion encodes `count(*)` as `count(1)` in Substrait plan
    // - Isthmus encodes `count(*)` as `count()` in Susbtrait plan
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
  } else {  // Aggregate functions with strictly 1 argument
    assert(nargs == 1);
    if (function_name == "sum" || function_name == "sum:opt_dec" ||
        function_name == "sum:opt_i32") {  // TODO: Look into substrait function naming standards
      return std::make_pair(
        cudf::aggregation::SUM,
        parse_expression(aggregate_function.arguments().Get(0).value(), subquery_relations));
    } else if (function_name == "avg") {
      return std::make_pair(
        cudf::aggregation::MEAN,
        parse_expression(aggregate_function.arguments().Get(0).value(), subquery_relations));
    } else {
      throw std::runtime_error("SubstraitParser cannot parse aggregate function \"" +
                               function_name + "\"");
    }
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

  for (auto const& sort_order : sort_relation.sorts()) {
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
    output_expressions.push_back(parse_expression(expression, subquery_relations));
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

  // Parse join condition
  std::vector<std::shared_ptr<gqe::logical::relation>> subquery_relations;
  auto condition = parse_expression(join_relation.expression(), subquery_relations);

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
    // TODO: Add left_semi, left_anti
    default: throw std::runtime_error("SubstraitParser: unsupported join type");
  }

  // Construct and return join relation
  // TODO: Configure projection indices from parent projection relation. For now,
  //       we'll return all columns and handle projection in a separate relation.
  auto const num_output_columns =
    num_columns_join_result(left_relation->num_columns(), right_relation->num_columns(), join_type);
  std::vector<cudf::size_type> projection_indices(num_output_columns);
  std::iota(projection_indices.begin(), projection_indices.end(), 0);

  return std::make_unique<gqe::logical::join_relation>(std::move(left_relation),
                                                       std::move(right_relation),
                                                       std::move(subquery_relations),
                                                       std::move(condition),
                                                       join_type,
                                                       std::move(projection_indices));
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
