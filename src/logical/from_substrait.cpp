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
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>

#include <substrait/algebra.pb.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
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
    if (!relation.has_root())
      throw std::runtime_error("Non-root top-level relation is not yet supported");

    relation_trees.push_back(parse_relation(relation.root().input()));
  }

  return relation_trees;
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_expression(
  substrait::Expression const& expression) const
{
  if (expression.has_selection())
    return parse_selection_expression(expression.selection());
  else if (expression.has_scalar_function())
    return parse_scalar_function_expression(expression.scalar_function());
  else if (expression.has_literal())
    return parse_literal_expression(expression.literal());
  else
    throw std::runtime_error("SubstraitParser cannot parse expression with type " +
                             std::to_string(expression.rex_type_case()));
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
    throw std::runtime_error("Only kDirectReference is supported for selection expressions");

  if (!selection_expression.direct_reference().has_struct_field())
    throw std::runtime_error("Only struct field is supported for selection expressions");

  auto struct_field = selection_expression.direct_reference().struct_field();

  if (struct_field.has_child())
    throw std::runtime_error("Does not support struct field with child");

  return std::make_unique<gqe::column_reference_expression>(struct_field.field());
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
  auto condition      = parse_expression(filter_relation.condition());

  return std::make_unique<gqe::logical::filter_relation>(std::move(input_relation),
                                                         std::move(condition));
}

std::pair<cudf::aggregation::Kind, std::unique_ptr<gqe::expression>>
gqe::substrait_parser::parse_aggregate_function(
  substrait::AggregateFunction const& aggregate_function) const
{
  auto function_name = get_function_name(aggregate_function.function_reference());

  if (function_name == "sum" || function_name == "sum:opt_dec" ||
      function_name == "sum:opt_i32") {  // TODO: Look into substrait function naming standards
    assert(aggregate_function.arguments_size() == 1);
    return std::make_pair(
      cudf::aggregation::SUM,
      parse_expression(std::move(aggregate_function.arguments().Get(0).value())));
  } else {
    throw std::runtime_error("SubstraitParser cannot parse aggregate function \"" + function_name +
                             "\"");
  }
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_aggregate_relation(
  substrait::AggregateRel const& aggregate_relation) const
{
  // Parse input relation
  auto input_relation = parse_relation(aggregate_relation.input());

  // Parse aggregation key(s)
  std::vector<std::unique_ptr<expression>> keys;
  if (aggregate_relation.groupings_size() > 1)
    throw std::runtime_error("Does not support groupby with multiple keys");

  auto grouping = aggregate_relation.groupings(0);
  for (auto& key_expression : grouping.grouping_expressions()) {
    keys.push_back(parse_expression(key_expression));
  }

  // Parse value expressions
  // Translate measures to pairs of function name and value expression
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures;
  for (auto& measure : aggregate_relation.measures()) {
    if (measure.has_filter())
      throw std::runtime_error("Aggregate measure with filter not supported");
    measures.push_back(parse_aggregate_function(measure.measure()));
  }

  return std::make_unique<gqe::logical::aggregate_relation>(
    std::move(input_relation), std::move(keys), std::move(measures));
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

  const size_t num_sorts = sort_relation.sorts_size();
  column_orders.reserve(num_sorts);
  null_precedences.reserve(num_sorts);
  expressions.reserve(num_sorts);

  for (auto const& sort_order : sort_relation.sorts()) {
    if (!sort_order.has_direction())
      throw std::runtime_error("Does not support sort with comparison function reference");

    expressions.push_back(parse_expression(sort_order.expr()));

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
  for (auto& expression : project_relation.expressions()) {
    output_expressions.push_back(parse_expression(expression));
  }

  return std::make_unique<gqe::logical::project_relation>(std::move(input_relation),
                                                          std::move(output_expressions));
}

std::unique_ptr<gqe::expression> gqe::substrait_parser::parse_scalar_function_expression(
  substrait::Expression_ScalarFunction const& scalar_function_expression) const
{
  if (scalar_function_expression.arguments_size() != 2)
    throw std::runtime_error("Non-binary functions are not yet supported");

  auto function_name = get_function_name(scalar_function_expression.function_reference());

  if (function_name == "equal" ||
      function_name == "equal:any_any") {  // TODO: Look into substrait function naming standards
    assert(scalar_function_expression.arguments_size() == 2);
    auto lhs = parse_expression(scalar_function_expression.arguments().Get(0).value());
    auto rhs = parse_expression(scalar_function_expression.arguments().Get(1).value());
    return std::make_unique<gqe::equal_expression>(std::move(lhs), std::move(rhs));
  } else if (function_name == "and" || function_name == "and:bool") {
    assert(scalar_function_expression.arguments_size() == 2);
    auto lhs = parse_expression(scalar_function_expression.arguments().Get(0).value());
    auto rhs = parse_expression(scalar_function_expression.arguments().Get(1).value());
    return std::make_unique<gqe::logical_and_expression>(std::move(lhs), std::move(rhs));
  } else {
    throw std::runtime_error("SubstraitParser cannot parse scalar function \"" + function_name +
                             "\"");
  }
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_join_relation(
  substrait::JoinRel const& join_relation) const
{
  // Parse children relations
  auto left_relation  = parse_relation(join_relation.left());
  auto right_relation = parse_relation(join_relation.right());

  // Parse join condition
  auto condition = parse_expression(join_relation.expression());

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
  return std::make_unique<gqe::logical::join_relation>(
    std::move(left_relation), std::move(right_relation), std::move(condition), join_type);
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_read_relation(
  substrait::ReadRel const& read_relation) const
{
  if (read_relation.read_type_case() != substrait::ReadRel::ReadTypeCase::kNamedTable)
    throw std::runtime_error("Only named table is supported for read relation");

  if (read_relation.has_filter())
    throw std::runtime_error("Filter is not supported for read relation");

  if (read_relation.has_projection())
    throw std::runtime_error("Projection is not supported for read relation");

  auto named_table = read_relation.named_table();
  assert(named_table.names_size() == 1);
  const std::string table_name = named_table.names(0);

  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  for (auto const& column_name : read_relation.base_schema().names()) {
    // Store column name in `column_names`
    column_names.push_back(column_name);

    // Store column type in `column_types`
    column_types.push_back(_catalog->column_type(table_name, column_name));
  }

  return std::make_unique<gqe::logical::read_relation>(
    std::move(column_names), std::move(column_types), std::move(table_name));
}
