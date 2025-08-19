/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <cudf/aggregation.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/optimizer/rules/column_name_assignment.hpp>
#include <regex>
#include <sstream>

std::shared_ptr<gqe::logical::relation> gqe::optimizer::column_name_assignment::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  // Process this relation based on its type
  std::vector<std::string> output_column_names;

  switch (logical_relation->type()) {
    case relation_t::read: {
      auto read_relation  = dynamic_cast<gqe::logical::read_relation*>(logical_relation.get());
      output_column_names = read_relation->column_names();

      // For partial filters and other expressions, use the full table schema from catalog
      // since they reference the original table columns before projection
      auto full_schema_names = get_catalog()->column_names(read_relation->table_name());
      update_column_references(logical_relation.get(), full_schema_names);
      break;
    }

    case relation_t::project: {
      auto project_relation = dynamic_cast<gqe::logical::project_relation*>(logical_relation.get());
      auto children         = logical_relation->children_unsafe();
      assert(children.size() == 1);
      auto input_column_names = extract_column_names(children[0]);

      // Update project expressions with input column names (not output names!)
      update_column_references(project_relation, input_column_names);

      output_column_names = assign_project_column_names(project_relation, input_column_names);
      break;
    }

    case relation_t::aggregate: {
      auto aggregate_relation =
        dynamic_cast<gqe::logical::aggregate_relation*>(logical_relation.get());
      auto children = logical_relation->children_unsafe();
      assert(children.size() == 1);

      // Special handling: If child is a filter relation, get projected column names
      std::vector<std::string> input_column_names;
      if (children[0]->type() == relation_t::filter) {
        auto filter_child       = dynamic_cast<const gqe::logical::filter_relation*>(children[0]);
        auto filter_input_names = extract_column_names(filter_child->children_unsafe()[0]);
        auto projection_indices = filter_child->projection_indices();

        input_column_names.reserve(projection_indices.size());
        for (auto idx : projection_indices) {
          if (idx >= 0 && static_cast<size_t>(idx) < filter_input_names.size()) {
            input_column_names.push_back(filter_input_names[idx]);
          } else {
            input_column_names.push_back("col_default_" + std::to_string(idx));
          }
        }
      } else {
        input_column_names = extract_column_names(children[0]);
      }

      // Update the column references within the aggregate relation itself
      update_column_references(aggregate_relation, input_column_names);

      output_column_names = assign_aggregate_column_names(aggregate_relation, input_column_names);
      break;
    }

    case relation_t::join: {
      auto join_relation = dynamic_cast<gqe::logical::join_relation*>(logical_relation.get());
      auto children      = logical_relation->children_unsafe();
      assert(children.size() == 2);
      auto left_column_names  = extract_column_names(children[0]);
      auto right_column_names = extract_column_names(children[1]);

      // Create full concatenated column names for join condition references
      std::vector<std::string> full_column_names;
      full_column_names.insert(
        full_column_names.end(), left_column_names.begin(), left_column_names.end());
      full_column_names.insert(
        full_column_names.end(), right_column_names.begin(), right_column_names.end());

      // Update join condition column references with full concatenated names
      update_column_references(join_relation, full_column_names);

      // Get projected output column names
      output_column_names =
        assign_join_column_names(join_relation, left_column_names, right_column_names);
      break;
    }

    case relation_t::filter:
    case relation_t::sort:
    case relation_t::window:
    case relation_t::fetch: {
      // These relations pass through their input column structure unchanged
      auto children = logical_relation->children_unsafe();
      assert(children.size() == 1);
      output_column_names = extract_column_names(children[0]);
      break;
    }

    case relation_t::set: {
      // Set operations preserve the input column structure (both inputs have same schema)
      auto children = logical_relation->children_unsafe();
      assert(children.size() == 2);
      output_column_names = extract_column_names(children[0]);
      break;
    }

    case relation_t::write:
    case relation_t::user_defined:
    default: {
      // For unsupported or special relations, generate generic names
      auto num_columns = logical_relation->num_columns();
      output_column_names.reserve(num_columns);
      for (cudf::size_type i = 0; i < num_columns; ++i) {
        output_column_names.push_back("col_default_" + std::to_string(i));
      }
      break;
    }
  }

  // Store the column names for this relation
  _relation_column_names[logical_relation.get()] = output_column_names;

  // Update column references in expressions to include names
  // Skip this for aggregate, read, join, and project relations since we handle them specifically
  if (logical_relation->type() != relation_t::aggregate &&
      logical_relation->type() != relation_t::read &&
      logical_relation->type() != relation_t::join &&
      logical_relation->type() != relation_t::project) {
    update_column_references(logical_relation.get(), output_column_names);
  }

  // This rule always does meaningful work (assigns/stores column names)
  rule_applied = true;

  return logical_relation;
}

std::vector<std::string> gqe::optimizer::column_name_assignment::assign_project_column_names(
  gqe::logical::project_relation* project_relation,
  const std::vector<std::string>& input_column_names) const
{
  std::vector<std::string> output_column_names;
  auto output_expressions = project_relation->const_output_expressions_unsafe();

  for (const auto* expr : output_expressions) {
    std::string column_name;

    if (expr->type() == gqe::expression::expression_type::column_reference) {
      // For column references, preserve the original name
      auto col_ref = dynamic_cast<const gqe::column_reference_expression*>(expr);
      auto col_idx = col_ref->column_idx();
      if (col_idx >= 0 && static_cast<size_t>(col_idx) < input_column_names.size()) {
        column_name = input_column_names[col_idx];
      } else {
        column_name = "col_default_" + std::to_string(col_idx);
      }
    } else {
      // For complex expressions, use a cleaned version of the expression string
      column_name = get_clean_expression_string(expr, input_column_names);
    }

    output_column_names.push_back(column_name);
  }

  return output_column_names;
}

std::vector<std::string> gqe::optimizer::column_name_assignment::assign_aggregate_column_names(
  gqe::logical::aggregate_relation* aggregate_relation,
  const std::vector<std::string>& input_column_names) const
{
  std::vector<std::string> output_column_names;

  // First, add names for grouping keys
  auto keys     = aggregate_relation->keys_unsafe();
  auto measures = aggregate_relation->measures_unsafe();
  for (const auto* key_expr : keys) {
    if (key_expr->type() == gqe::expression::expression_type::column_reference) {
      auto col_ref = dynamic_cast<const gqe::column_reference_expression*>(key_expr);
      // Use the column name from the column reference expression itself (same as parser)
      auto column_name = col_ref->column_name();
      if (!column_name.empty()) {
        output_column_names.push_back(column_name);
      } else {
        // Fallback to input column names if column reference doesn't have a name
        auto col_idx = col_ref->column_idx();
        if (col_idx >= 0 && static_cast<size_t>(col_idx) < input_column_names.size()) {
          output_column_names.push_back(input_column_names[col_idx]);
        } else {
          output_column_names.push_back("groupby_key_" + std::to_string(col_idx));
        }
      }
    } else {
      output_column_names.push_back(get_clean_expression_string(key_expr, input_column_names));
    }
  }

  // Then, add names for aggregation measures
  for (const auto& measure : measures) {
    auto agg_kind    = measure.first;
    auto* value_expr = measure.second;

    std::string agg_name = get_aggregation_name(agg_kind);
    std::string operand_name;

    if (value_expr->type() == gqe::expression::expression_type::column_reference) {
      auto col_ref = dynamic_cast<const gqe::column_reference_expression*>(value_expr);
      // Use the column name from the column reference expression itself (same as parser)
      auto column_name = col_ref->column_name();
      if (!column_name.empty()) {
        operand_name = column_name;
      } else {
        // Fallback to input column names if column reference doesn't have a name
        auto col_idx = col_ref->column_idx();
        if (col_idx >= 0 && static_cast<size_t>(col_idx) < input_column_names.size()) {
          operand_name = input_column_names[col_idx];
        } else {
          operand_name = "col_default_" + std::to_string(col_idx);
        }
      }
    } else {
      operand_name = get_clean_expression_string(value_expr, input_column_names);
    }

    output_column_names.push_back(agg_name + "(" + operand_name + ")");
  }

  return output_column_names;
}

std::vector<std::string> gqe::optimizer::column_name_assignment::assign_join_column_names(
  gqe::logical::join_relation* join_relation,
  const std::vector<std::string>& left_column_names,
  const std::vector<std::string>& right_column_names) const
{
  std::vector<std::string> output_column_names;

  // For joins, we typically concatenate left and right column names
  // The projection indices determine which columns are actually output
  auto projection_indices = join_relation->projection_indices();

  // Create full concatenated column names
  std::vector<std::string> full_column_names;
  full_column_names.insert(
    full_column_names.end(), left_column_names.begin(), left_column_names.end());
  full_column_names.insert(
    full_column_names.end(), right_column_names.begin(), right_column_names.end());

  // Extract only the projected columns
  for (auto idx : projection_indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < full_column_names.size()) {
      output_column_names.push_back(full_column_names[idx]);
    } else {
      output_column_names.push_back("join_col_" + std::to_string(idx));
    }
  }

  return output_column_names;
}

void gqe::optimizer::column_name_assignment::update_column_references(
  gqe::logical::relation* relation, const std::vector<std::string>& column_names) const
{
  // Create an expression modifier that updates column references to include names
  auto expr_modifier =
    [&](gqe::expression* expr,
        std::vector<cudf::data_type> const& column_types) -> std::unique_ptr<gqe::expression> {
    if (expr->type() == gqe::expression::expression_type::column_reference) {
      auto col_ref = dynamic_cast<gqe::column_reference_expression*>(expr);
      auto col_idx = col_ref->column_idx();

      // Get the current column name (might be empty)
      auto current_name = col_ref->column_name();

      // Determine the new column name
      std::string new_name;
      if (col_idx >= 0 && static_cast<size_t>(col_idx) < column_names.size()) {
        new_name = column_names[col_idx];
      } else {
        new_name = "col_default_" + std::to_string(col_idx);
      }

      // Only create a new expression if the name is different
      if (current_name != new_name) {
        return std::make_unique<gqe::column_reference_expression>(col_idx, new_name);
      }
    }

    return nullptr;  // No change needed
  };

  // Apply the modifier to all expressions in the relation
  rewrite_relation_expressions(relation, expr_modifier, transform_direction::DOWN);
}

std::vector<std::string> gqe::optimizer::column_name_assignment::extract_column_names(
  const gqe::logical::relation* relation) const
{
  // Check if we have already assigned names to this relation
  auto it = _relation_column_names.find(relation);
  if (it != _relation_column_names.end()) { return it->second; }

  // Try to extract names based on relation type
  if (relation->type() == relation_t::read) {
    auto read_relation = dynamic_cast<const gqe::logical::read_relation*>(relation);
    return read_relation->column_names();
  }

  // Fallback: generate generic column names
  std::vector<std::string> column_names;
  auto num_columns = relation->num_columns();
  column_names.reserve(num_columns);
  for (cudf::size_type i = 0; i < num_columns; ++i) {
    column_names.push_back("col_default_" + std::to_string(i));
  }

  return column_names;
}

std::string gqe::optimizer::column_name_assignment::get_clean_expression_string(
  const gqe::expression* expr, const std::vector<std::string>& input_column_names) const
{
  if (!expr) return "unknown_expr";

  // Get the original string representation
  std::string result = expr->to_string();

  // Simple regex replacements for cleaner output
  std::regex literal_regex(R"(literal\([^)]+\s+([^)]+)\))");
  result = std::regex_replace(result, literal_regex, "$1");

  std::regex column_ref_regex(R"(column_reference\((\d+)\))");
  result = std::regex_replace(result, column_ref_regex, "$1");
  // Limit length to keep names manageable
  if (result.length() > 70) { result = result.substr(0, 67) + "..."; }

  return result;
}

std::string gqe::optimizer::column_name_assignment::get_aggregation_name(
  cudf::aggregation::Kind agg_kind) const
{
  switch (agg_kind) {
    case cudf::aggregation::SUM: return "sum";
    case cudf::aggregation::MEAN: return "avg";
    case cudf::aggregation::COUNT_VALID: return "count";
    case cudf::aggregation::COUNT_ALL: return "count";
    case cudf::aggregation::MIN: return "min";
    case cudf::aggregation::MAX: return "max";
    case cudf::aggregation::VARIANCE: return "var";
    case cudf::aggregation::STD: return "std";
    case cudf::aggregation::MEDIAN: return "median";
    case cudf::aggregation::NUNIQUE: return "nunique";
    case cudf::aggregation::NTH_ELEMENT: return "nth";
    case cudf::aggregation::ARGMAX: return "argmax";
    case cudf::aggregation::ARGMIN: return "argmin";
    default: return "agg";
  }
}
