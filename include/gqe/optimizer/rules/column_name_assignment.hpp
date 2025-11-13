/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <cudf/types.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gqe {
namespace logical {
// Forward declarations
class read_relation;
class project_relation;
class aggregate_relation;
class join_relation;
}  // namespace logical
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;

/**
 * @brief This rule assigns meaningful column names to all relations in the logical plan
 * and updates column_reference_expressions to include these names. It works independently
 * of the Substrait parser and can be applied to any logical plan.
 *
 * The rule traverses the logical plan in a bottom-up manner and:
 * 1. For read relations: Extracts column names from the table schema
 * 2. For project relations: Uses expression strings or preserves input names
 * 3. For aggregate relations: Generates meaningful aggregation names
 * 4. For join relations: Concatenates left and right column names
 * 5. For other relations: Preserves input column names
 *
 * This rule should be applied early in the optimization pipeline to ensure
 * that subsequent rules have access to meaningful column names.
 */
class column_name_assignment : public optimization_rule {
 public:
  column_name_assignment(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::UP)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::column_name_assignment;
  }

 private:
  /**
   * @brief Generate column names for a project relation based on expressions
   *
   * @param project_relation The project relation to assign names to
   * @param input_column_names Column names from the input relation
   * @return Generated column names for the project output
   */
  std::vector<std::string> assign_project_column_names(
    gqe::logical::project_relation* project_relation,
    const std::vector<std::string>& input_column_names) const;

  /**
   * @brief Generate column names for an aggregate relation
   *
   * @param aggregate_relation The aggregate relation to assign names to
   * @param input_column_names Column names from the input relation
   * @return Generated column names for the aggregate output
   */
  std::vector<std::string> assign_aggregate_column_names(
    gqe::logical::aggregate_relation* aggregate_relation,
    const std::vector<std::string>& input_column_names) const;

  /**
   * @brief Generate column names for a join relation
   *
   * @param join_relation The join relation to assign names to
   * @param left_column_names Column names from the left input
   * @param right_column_names Column names from the right input
   * @return Generated column names for the join output
   */
  std::vector<std::string> assign_join_column_names(
    gqe::logical::join_relation* join_relation,
    const std::vector<std::string>& left_column_names,
    const std::vector<std::string>& right_column_names) const;

  /**
   * @brief Update column reference expressions in the relation to include column names
   *
   * @param relation The relation to update
   * @param column_names The column names to assign
   */
  void update_column_references(gqe::logical::relation* relation,
                                const std::vector<std::string>& column_names) const;

  /**
   * @brief Extract column names from a relation (if already assigned)
   *
   * @param relation The relation to extract names from
   * @return Vector of column names, or generated placeholder names if not available
   */
  std::vector<std::string> extract_column_names(const gqe::logical::relation* relation) const;

  /**
   * @brief Generate a clean expression string for use in column names
   *
   * @param expr Expression to convert to string
   * @return Clean string representation without verbose wrappers
   */
  std::string get_clean_expression_string(const gqe::expression* expr,
                                          const std::vector<std::string>& input_column_names) const;

  /**
   * @brief Generate aggregation function name string
   *
   * @param agg_kind The aggregation kind
   * @return String representation of the aggregation function
   */
  std::string get_aggregation_name(cudf::aggregation::Kind agg_kind) const;

  // Storage for column names associated with each relation
  mutable std::unordered_map<const gqe::logical::relation*, std::vector<std::string>>
    _relation_column_names;
};

}  // namespace optimizer
}  // namespace gqe