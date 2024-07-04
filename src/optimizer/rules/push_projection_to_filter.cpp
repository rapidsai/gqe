/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/logical/filter.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/optimizer/rules/push_projection_to_filter.hpp>

#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>

namespace {

std::unordered_map<cudf::size_type, cudf::size_type> construct_map(
  std::vector<cudf::size_type> const& required_cr_indices)
{
  std::unordered_map<cudf::size_type, cudf::size_type> map;
  for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(required_cr_indices.size()); i++) {
    if (i != required_cr_indices[i]) { map.insert({required_cr_indices[i], i}); }
  }
  return map;
}

bool is_all_col_ref(std::vector<gqe::expression const*> const& expressions)
{
  for (auto& expr : expressions) {
    if (expr->type() != gqe::expression::expression_type::column_reference) { return false; }
  }
  return true;
}

void store_column_idx(gqe::expression const* expr, std::set<cudf::size_type>& required_columns)
{
  auto cr = dynamic_cast<gqe::column_reference_expression const*>(expr);
  required_columns.insert(cr->column_idx());
}

void extract_required_columns(gqe::expression const* expr,
                              std::set<cudf::size_type>& required_columns)
{
  auto children = expr->children();
  if (children.size() != 0) {
    for (auto child : children) {
      extract_required_columns(child, required_columns);
    }
  }

  else if (expr->type() == gqe::expression::expression_type::column_reference) {
    store_column_idx(expr, required_columns);
  }
}

/*
 * Since, project relation will be removed,
 * reordering must be done in filter relation
 * order of required indices needs to preserved
 *
 * This function expects all expressions to be column references
 */
std::vector<cudf::size_type> required_indices_from_cr_expressions(
  std::vector<gqe::expression const*> const& expressions)
{
  std::vector<cudf::size_type> required_cr_indices;
  for (auto& expr : expressions) {
    assert(expr->type() == gqe::expression::expression_type::column_reference);

    auto cr = dynamic_cast<gqe::column_reference_expression const*>(expr);
    required_cr_indices.push_back(cr->column_idx());
  }
  return required_cr_indices;
}

/*
 * Since, project relation will be retained
 * reordering can be done in project task
 * sorted list is maintained to reduce number of rewritings
 */
std::vector<cudf::size_type> required_indices_from_mixed_expressions(
  std::vector<gqe::expression const*> const& expressions)
{
  // By default, set stores elements in ascending order
  std::set<cudf::size_type> required_columns;
  for (auto& expr : expressions) {
    if (expr->type() != gqe::expression::expression_type::column_reference) {
      extract_required_columns(expr, required_columns);
    } else {
      store_column_idx(expr, required_columns);
    }
  }

  std::vector<cudf::size_type> required_cr_indices(required_columns.begin(),
                                                   required_columns.end());
  return required_cr_indices;
}

std::vector<cudf::size_type> inspect_output_expressions(
  std::vector<gqe::expression const*> const& expressions, bool& all_col_ref)
{
  all_col_ref = is_all_col_ref(expressions);

  if (all_col_ref) {
    return required_indices_from_cr_expressions(expressions);
  }

  else {
    return required_indices_from_mixed_expressions(expressions);
  }
}

std::vector<cudf::size_type> get_updated_projection_indices(
  std::vector<cudf::size_type> const& old_projection_indices,
  std::vector<cudf::size_type> const& required_cr_indices)
{
  std::vector<cudf::size_type> new_projection_indices;
  for (auto i : required_cr_indices) {
    new_projection_indices.push_back(old_projection_indices[i]);
  }
  return new_projection_indices;
}

}  // namespace

void gqe::optimizer::push_projection_to_filter::rewrite_filter_relation(
  std::shared_ptr<gqe::logical::filter_relation> filter,
  std::vector<cudf::size_type> const& required_cr_indices) const
{
  auto projection_indices =
    get_updated_projection_indices(filter->_projection_indices, required_cr_indices);
  filter->_projection_indices = projection_indices;
}

void gqe::optimizer::push_projection_to_filter::rewrite_project_relation(
  std::shared_ptr<gqe::logical::relation> project,
  std::vector<cudf::size_type> const& required_cr_indices) const
{
  auto map = construct_map(required_cr_indices);

  auto expr_modifier = [&](expression* expr) -> std::unique_ptr<gqe::expression> {
    if (expr->type() == gqe::expression::expression_type::column_reference) {
      auto cr = dynamic_cast<gqe::column_reference_expression*>(expr);

      auto search = map.find(cr->column_idx());
      if (search == map.end()) { return static_cast<std::unique_ptr<gqe::expression>>(nullptr); }
      return std::make_unique<gqe::column_reference_expression>(search->second);
    }

    else {
      return static_cast<std::unique_ptr<gqe::expression>>(nullptr);
    }
  };

  // Any transformation direction can be used
  rewrite_relation_expressions(project.get(), expr_modifier, transform_direction::DOWN);
}

std::shared_ptr<gqe::logical::relation> gqe::optimizer::push_projection_to_filter::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  // Check if it's a project relation
  if (logical_relation->type() != relation_t::project) return logical_relation;
  auto project = std::dynamic_pointer_cast<logical::project_relation>(logical_relation);

  // Check if project's child is filter
  auto children = project->children_safe();
  assert(children.size() == 1);
  auto child = children[0];
  if (child->type() != relation_t::filter) return logical_relation;

  // Identify which columns are used by project relation
  bool all_col_ref;
  auto required_cr_indices =
    inspect_output_expressions(project->const_output_expressions_unsafe(), all_col_ref);

  std::shared_ptr<logical::filter_relation> filter =
    std::dynamic_pointer_cast<logical::filter_relation>(child);

  if (all_col_ref) {
    // If the project relation has only column reference expressions, we can remove the project
    // relation from the query plan.
    rule_applied = true;
    rewrite_filter_relation(filter, required_cr_indices);
    return filter;
  } else {
    // Otherwise, we push the column indices needed for the project relation to the filter relation,
    // and rewrite both relations to update the column references.
    auto old_projection_indices = filter->_projection_indices;

    // If no columns to be dropped, no rewriting
    if (old_projection_indices.size() == required_cr_indices.size()) {
      return logical_relation;
    }

    // Update the projection indices of filter and column references of project
    else {
      rule_applied = true;
      rewrite_filter_relation(filter, required_cr_indices);
      rewrite_project_relation(logical_relation, required_cr_indices);
      return logical_relation;
    }
  }
}
