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

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void gqe::substrait_parser::register_input_table(
  std::string table_name,
  std::vector<std::string> const& column_names,
  std::vector<cudf::data_type> const& column_types,
  std::vector<std::string> const& file_paths)  // TODO: build table/filepath map
{
  if (column_names.size() != column_types.size())
    throw std::runtime_error(R"("column_names" and "column_types" must have the same length)");
  if (input_column_types.find(table_name) != input_column_types.end())
    throw std::runtime_error("Table " + table_name + " is already registered");

  for (std::size_t column_idx = 0; column_idx < column_names.size(); column_idx++) {
    input_column_types[table_name][column_names[column_idx]] = column_types[column_idx];
  }
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
  else
    throw std::runtime_error("SubstraitParser cannot parse expression with type " +
                             std::to_string(expression.rex_type_case()));
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
  else
    throw std::runtime_error("Unsupported relation type");
}

std::unique_ptr<gqe::logical::relation> gqe::substrait_parser::parse_project_relation(
  substrait::ProjectRel const& project_relation) const
{
  // Extract `input` field from Substrait ProjectRel
  // Recursively parse input relation into logical relations
  auto input_relation = parse_relation(project_relation.input());

  std::vector<std::shared_ptr<gqe::expression>> output_expressions;
  for (auto& expression : project_relation.expressions()) {
    output_expressions.push_back(parse_expression(expression));
  }

  return std::make_unique<gqe::logical::project_relation>(std::move(input_relation),
                                                          std::move(output_expressions));
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
  for (auto const& name : read_relation.base_schema().names()) {
    // Store column name in `column_names`
    column_names.push_back(name);

    // Check that the associated table has been registered
    auto table_iter = input_column_types.find(table_name);
    if (table_iter == input_column_types.end())
      throw std::runtime_error("Cannot find " + table_name + " in registered tables");
    auto const& input_column_types_this_table = table_iter->second;

    // Check that this column is in the matched table
    auto column_type_iter = input_column_types_this_table.find(name);
    if (column_type_iter == input_column_types_this_table.end())
      throw std::runtime_error("Cannot find " + name + " column in registered tables");
    auto const& column_type = column_type_iter->second;

    // Store column type in `column_types`
    column_types.push_back(column_type);
  }

  return std::make_unique<gqe::logical::read_relation>(
    std::move(column_names), std::move(column_types), std::move(table_name));
}
