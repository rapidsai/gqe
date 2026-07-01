/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/column_name_assignment.hpp>

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
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>

#include <cudf/aggregation.hpp>

#include <algorithm>
#include <regex>
#include <sstream>

namespace gqe::optimizer {

class column_name_assignment::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(column_name_assignment const& rule) : _rule{rule} {}

  [[nodiscard]] std::vector<std::string> const& output_names() const noexcept
  {
    return _output_names;
  }

  void visit(gqe::logical::read_relation* read) override
  {
    visit_children(read);  // post-order (read has no children; explicit for consistency)
    _output_names = read->column_names();

    // For partial filters and other expressions, use the full table schema from catalog
    // since they reference the original table columns before projection
    auto full_schema_names = _rule.get_catalog()->column_names(read->table_name());
    _rule.update_column_references(read, full_schema_names);
    _rule._relation_column_names[read] = _output_names;
  }

  void visit(gqe::logical::project_relation* project) override
  {
    visit_children(project);  // post-order: children before parent
    auto children = project->children_unsafe();
    assert(children.size() == 1);
    auto input_column_names = _rule.extract_column_names(children[0]);

    // Update project expressions with input column names (not output names!)
    _rule.update_column_references(project, input_column_names);

    _output_names = _rule.assign_project_column_names(project, input_column_names);
    _rule._relation_column_names[project] = _output_names;
  }

  void visit(gqe::logical::aggregate_relation* aggregate) override
  {
    visit_children(aggregate);  // post-order: children before parent
    auto children = aggregate->children_unsafe();
    assert(children.size() == 1);

    // Special handling: If child is a filter relation, get projected column names
    std::vector<std::string> input_column_names;
    if (children[0]->type() == relation_t::filter) {
      auto filter_child       = static_cast<const gqe::logical::filter_relation*>(children[0]);
      auto filter_input_names = _rule.extract_column_names(filter_child->children_unsafe()[0]);
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
      input_column_names = _rule.extract_column_names(children[0]);
    }

    // Update the column references within the aggregate relation itself
    _rule.update_column_references(aggregate, input_column_names);

    _output_names = _rule.assign_aggregate_column_names(aggregate, input_column_names);
    _rule._relation_column_names[aggregate] = _output_names;
  }

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);  // post-order: children before parent
    auto children = join->children_unsafe();
    assert(children.size() == 2);
    auto left_column_names  = _rule.extract_column_names(children[0]);
    auto right_column_names = _rule.extract_column_names(children[1]);

    // Create full concatenated column names for join condition references
    std::vector<std::string> full_column_names;
    full_column_names.reserve(left_column_names.size() + right_column_names.size());
    full_column_names.insert(
      full_column_names.end(), left_column_names.begin(), left_column_names.end());
    full_column_names.insert(
      full_column_names.end(), right_column_names.begin(), right_column_names.end());

    // Update join condition column references with full concatenated names
    _rule.update_column_references(join, full_column_names);

    // Get projected output column names
    _output_names = _rule.assign_join_column_names(join, left_column_names, right_column_names);
    _rule._relation_column_names[join] = _output_names;
  }

  // filter / sort / window / fetch all pass through their input column structure unchanged.
  void visit(gqe::logical::filter_relation* filter) override { _passthrough_then_update(filter); }
  void visit(gqe::logical::sort_relation* sort) override { _passthrough_then_update(sort); }
  void visit(gqe::logical::window_relation* window) override { _passthrough_then_update(window); }
  void visit(gqe::logical::fetch_relation* fetch) override { _passthrough_then_update(fetch); }

  void visit(gqe::logical::set_relation* set_op) override
  {
    visit_children(set_op);  // post-order: children before parent
    // Set operations preserve the input column structure (both inputs have same schema)
    auto children = set_op->children_unsafe();
    assert(children.size() == 2);
    _output_names = _rule.extract_column_names(children[0]);
    _rule.update_column_references(set_op, _output_names);
    _rule._relation_column_names[set_op] = _output_names;
  }

  void visit(gqe::logical::write_relation* write) override { _generate_default_then_update(write); }
  void visit(gqe::logical::user_defined_relation* user_defined) override
  {
    _generate_default_then_update(user_defined);
  }

 private:
  void _passthrough_then_update(gqe::logical::relation* rel)
  {
    visit_children(rel);  // post-order: children before parent
    auto children = rel->children_unsafe();
    assert(children.size() == 1);
    _output_names = _rule.extract_column_names(children[0]);
    _rule.update_column_references(rel, _output_names);
    _rule._relation_column_names[rel] = _output_names;
  }

  void _generate_default_then_update(gqe::logical::relation* rel)
  {
    visit_children(rel);  // post-order: children before parent
    // For unsupported or special relations, generate generic names
    auto num_columns = rel->num_columns();
    _output_names.clear();
    _output_names.reserve(num_columns);
    for (cudf::size_type i = 0; i < num_columns; ++i) {
      _output_names.push_back("col_default_" + std::to_string(i));
    }
    _rule.update_column_references(rel, _output_names);
    _rule._relation_column_names[rel] = _output_names;
  }

  column_name_assignment const& _rule;
  std::vector<std::string> _output_names;
};

std::shared_ptr<gqe::logical::relation> column_name_assignment::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this};
  root->accept(visitor);
  rule_applied = true;
  return root;
}

std::vector<std::string> column_name_assignment::assign_project_column_names(
  gqe::logical::project_relation* project_relation,
  const std::vector<std::string>& input_column_names) const
{
  std::vector<std::string> output_column_names;
  auto output_expressions = project_relation->const_output_expressions_unsafe();
  output_column_names.reserve(output_expressions.size());

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

std::vector<std::string> column_name_assignment::assign_aggregate_column_names(
  gqe::logical::aggregate_relation* aggregate_relation,
  const std::vector<std::string>& input_column_names) const
{
  std::vector<std::string> output_column_names;

  // First, add names for grouping keys
  auto keys     = aggregate_relation->keys_unsafe();
  auto measures = aggregate_relation->measures_unsafe();
  output_column_names.reserve(keys.size() + measures.size());
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

std::vector<std::string> column_name_assignment::assign_join_column_names(
  gqe::logical::join_relation* join_relation,
  const std::vector<std::string>& left_column_names,
  const std::vector<std::string>& right_column_names) const
{
  std::vector<std::string> output_column_names;

  // For joins, we typically concatenate left and right column names
  // The projection indices determine which columns are actually output
  auto projection_indices = join_relation->projection_indices();
  output_column_names.reserve(projection_indices.size());

  // Create full concatenated column names
  std::vector<std::string> full_column_names;
  full_column_names.reserve(left_column_names.size() + right_column_names.size());
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

void column_name_assignment::update_column_references(
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

std::vector<std::string> column_name_assignment::extract_column_names(
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

std::string column_name_assignment::get_clean_expression_string(
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

std::string column_name_assignment::get_aggregation_name(cudf::aggregation::Kind agg_kind) const
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

}  // namespace gqe::optimizer
