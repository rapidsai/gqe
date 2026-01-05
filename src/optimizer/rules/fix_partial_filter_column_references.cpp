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

#include <gqe/optimizer/rules/fix_partial_filter_column_references.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/relation.hpp>

std::shared_ptr<gqe::logical::relation>
gqe::optimizer::fix_partial_filter_column_references::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  if (logical_relation->type() == gqe::logical::relation::relation_type::filter) {
    const auto child = logical_relation->children_unsafe()[0];
    if (child->type() == gqe::logical::relation::relation_type::read) {
      const auto filter_relation =
        static_cast<gqe::logical::filter_relation*>(logical_relation.get());
      const auto read_relation       = static_cast<gqe::logical::read_relation*>(child);
      const auto table_name          = read_relation->table_name();
      const auto base_table_columns  = get_catalog()->column_names(table_name);
      const auto projected_columns   = read_relation->column_names();
      auto original_filter_condition = filter_relation->condition()->clone();
      auto updated_filter_condition =
        replace_column_references(base_table_columns, projected_columns, filter_relation);
      set_expression(read_relation, std::move(updated_filter_condition));
      set_expression(filter_relation, std::move(original_filter_condition));
    }
  }
  return logical_relation;
}

void gqe::optimizer::fix_partial_filter_column_references::set_expression(
  gqe::logical::relation* relation, std::unique_ptr<gqe::expression> new_expression) const
{
  auto expression_modifier = [&new_expression](
                               gqe::expression* expression,
                               [[maybe_unused]] const std::vector<cudf::data_type>& _) {
    return std::move(new_expression);
  };
  rewrite_relation_expressions(relation, expression_modifier, transform_direction::NONE);
}

std::unique_ptr<gqe::expression>
gqe::optimizer::fix_partial_filter_column_references::replace_column_references(
  const std::vector<std::string>& base_table_columns,
  const std::vector<std::string>& projected_columns,
  gqe::logical::filter_relation* filter_relation) const
{
  auto column_reference_modifier =
    [&projected_columns, &base_table_columns](
      expression* expression,
      [[maybe_unused]] const std::vector<cudf::data_type>& _) -> std::unique_ptr<gqe::expression> {
    if (expression->type() == expression::expression_type::column_reference) {
      auto column_reference = static_cast<column_reference_expression*>(expression);
      auto column           = projected_columns[column_reference->column_idx()];
      auto it = std::find(base_table_columns.begin(), base_table_columns.end(), column);
      if (it == base_table_columns.end()) {
        throw std::logic_error("Did not find column in base table schema: " + column);
      } else {
        return std::make_unique<gqe::column_reference_expression>(
          std::distance(base_table_columns.begin(), it));
      }
    } else {
      return expression->clone();
    }
  };

  rewrite_relation_expressions(
    filter_relation, column_reference_modifier, transform_direction::DOWN);
  return filter_relation->condition()->clone();
}
