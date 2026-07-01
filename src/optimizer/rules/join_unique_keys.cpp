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

#include <gqe/optimizer/rules/join_unique_keys.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/optimizer/relation_properties.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <unordered_set>

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
        // NULL_EQUALS (IS NOT DISTINCT FROM) is intentionally excluded: a nullable UNIQUE column
        // permits multiple NULL rows, so the unique-build-side assumption would be violated when
        // a probe-side NULL matches more than one build-side NULL.
        // TODO: Handle NULL_EQUALS (IS NOT DISTINCT FROM) if needed
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

namespace gqe::optimizer {

class join_unique_keys::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(bool& rule_applied) : _rule_applied{rule_applied} {}

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);  // post-order: recurse first

    // only apply rule for inner join
    if (join->join_type() != join_type_type::inner) return;

    auto children = join->children_unsafe();
    // only apply rule for equijoin predicate
    std::vector<cudf::size_type> left_key_indices, right_key_indices;
    if (!is_equijoin_predicate(
          join->condition(), children[0]->num_columns(), left_key_indices, right_key_indices))
      return;

    const auto& left_col_props  = children[0]->relation_traits().properties();
    const auto& right_col_props = children[1]->relation_traits().properties();

    // Returns true if any registered unique key-set is fully covered by the equi-join keys.
    // A singleton key {col} is covered iff col ∈ equi_set (single-column unique case).
    auto side_covered = [](optimizer::relation_properties const& props,
                           std::vector<cudf::size_type> const& equi_keys) {
      std::unordered_set<cudf::size_type> equi_set(equi_keys.begin(), equi_keys.end());
      for (auto const& key : props.unique_keys()) {
        if (std::all_of(key.begin(), key.end(), [&](auto c) { return equi_set.count(c) > 0; }))
          return true;
      }
      return false;
    };

    bool const left_key_unique  = side_covered(left_col_props, left_key_indices);
    bool const right_key_unique = side_covered(right_col_props, right_key_indices);

    // apply rule by setting unique keys policy in join relation
    if (left_key_unique && right_key_unique) {
      join->set_unique_keys_policy(gqe::unique_keys_policy::either);
      _rule_applied = true;
    } else if (left_key_unique) {
      join->set_unique_keys_policy(gqe::unique_keys_policy::left);
      _rule_applied = true;
    } else if (right_key_unique) {
      join->set_unique_keys_policy(gqe::unique_keys_policy::right);
      _rule_applied = true;
    }
  }

 private:
  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> join_unique_keys::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
