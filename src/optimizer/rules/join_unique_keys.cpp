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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/optimizer/rules/join_unique_keys.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <cassert>
#include <cstddef>
#include <memory>

namespace {
using expression_t = gqe::expression::expression_type;

/* equijoin predicate:
 * - predicate is a conjunction of equality comparisons
 * - each equality comparison is between two column references, one from each child relation
 */
bool is_equijoin_predicate(gqe::expression const* condition,
                           cudf::size_type const num_columns_left_child_relation,
                           std::vector<cudf::size_type>& left_key_indices,
                           std::vector<cudf::size_type>& right_key_indices)
{
  if (condition->type() == expression_t::binary_op) {
    auto binary_op_condition = dynamic_cast<gqe::binary_op_expression const*>(condition);
    auto child_exprs         = condition->children();
    assert(child_exprs.size() == 2);

    int col_idx_0, col_idx_1;
    switch (binary_op_condition->binary_operator()) {
      case cudf::binary_operator::EQUAL:
      case cudf::binary_operator::NULL_EQUALS:
        if (child_exprs[0]->type() == expression_t::column_reference &&
            child_exprs[1]->type() == expression_t::column_reference) {
          col_idx_0 = static_cast<int>(
            dynamic_cast<gqe::column_reference_expression const*>(child_exprs[0])->column_idx());
          col_idx_1 = static_cast<int>(
            dynamic_cast<gqe::column_reference_expression const*>(child_exprs[1])->column_idx());

          // check that col_idx_0 and col_idx_1 do not both belong to the same child relation
          if ((col_idx_0 < num_columns_left_child_relation) ==
              (col_idx_1 < num_columns_left_child_relation))
            return false;

          left_key_indices.push_back(col_idx_0 < num_columns_left_child_relation ? col_idx_0
                                                                                 : col_idx_1);
          right_key_indices.push_back(
            (col_idx_0 < num_columns_left_child_relation ? col_idx_1 : col_idx_0) -
            num_columns_left_child_relation);
        } else
          return false;
        break;
      case cudf::binary_operator::LOGICAL_AND:
      case cudf::binary_operator::NULL_LOGICAL_AND:
        // If the top-level expression is AND, we recursively check the two children expressions
        if (!is_equijoin_predicate(
              child_exprs[0], num_columns_left_child_relation, left_key_indices, right_key_indices))
          return false;
        if (!is_equijoin_predicate(
              child_exprs[1], num_columns_left_child_relation, left_key_indices, right_key_indices))
          return false;
        break;
      default: return false;
    }
    return true;
  } else
    return false;
}

}  // namespace

std::shared_ptr<gqe::logical::relation> gqe::optimizer::join_unique_keys::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  // only apply rule for join relation
  if (logical_relation->type() != relation_t::join) return logical_relation;

  auto join = dynamic_cast<gqe::logical::join_relation*>(logical_relation.get());

  // only apply rule for inner join
  if (join->join_type() != join_type_type::inner) return logical_relation;

  auto children = join->children_unsafe();
  // only apply rule for equijoin predicate
  std::vector<cudf::size_type> left_key_indices, right_key_indices;
  if (!is_equijoin_predicate(
        join->condition(), children[0]->num_columns(), left_key_indices, right_key_indices))
    return logical_relation;

  bool left_key_unique(false), right_key_unique(false);
  const auto& left_col_props  = children[0]->relation_traits().properties();
  const auto& right_col_props = children[1]->relation_traits().properties();

  for (auto col_idx : left_key_indices) {
    if (left_col_props.check_column_property(col_idx,
                                             optimizer::column_property::property_id::unique)) {
      left_key_unique = true;
      break;
    }
  }

  for (auto col_idx : right_key_indices) {
    if (right_col_props.check_column_property(col_idx,
                                              optimizer::column_property::property_id::unique)) {
      right_key_unique = true;
      break;
    }
  }

  // apply rule by setting unique keys policy in join relation
  if (left_key_unique && right_key_unique) {
    join->set_unique_keys_policy(gqe::unique_keys_policy::either);
    rule_applied = true;
  } else if (left_key_unique) {
    join->set_unique_keys_policy(gqe::unique_keys_policy::left);
    rule_applied = true;
  } else if (right_key_unique) {
    join->set_unique_keys_policy(gqe::unique_keys_policy::right);
    rule_applied = true;
  }

  return logical_relation;
}
