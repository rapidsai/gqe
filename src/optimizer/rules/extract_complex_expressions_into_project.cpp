/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/extract_complex_expressions_into_project.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>

#include <cudf/binaryop.hpp>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe::optimizer {

namespace {

using expr_type = gqe::expression::expression_type;

/// @brief `true` if `expr` is exactly `column_reference(idx)`.
[[nodiscard]] bool is_column_reference_at(gqe::expression const* expr, cudf::size_type idx)
{
  return expr->type() == expr_type::column_reference &&
         static_cast<gqe::column_reference_expression const*>(expr)->column_idx() == idx;
}

/// @brief `true` if a `subquery_expression` appears anywhere in `expr`.
[[nodiscard]] bool contains_subquery(gqe::expression const* expr)
{
  if (expr == nullptr) return false;
  if (expr->type() == expr_type::subquery) return true;
  for (auto const* child : expr->children()) {
    if (contains_subquery(child)) return true;
  }
  return false;
}

/// @brief Which join input an operand references, classified by its column-reference indices.
enum class operand_side {
  left,     ///< All column references address the left input (`idx < num_left`).
  right,    ///< All column references address the right input (`idx >= num_left`).
  invalid,  ///< References both inputs, or no column at all — cannot be assigned to one side.
};

/// @brief Collect every column-reference index reachable from `expr`.
void collect_column_indices(gqe::expression const* expr, std::vector<cudf::size_type>& indices)
{
  if (expr->type() == expr_type::column_reference) {
    indices.push_back(static_cast<gqe::column_reference_expression const*>(expr)->column_idx());
    return;
  }
  for (auto const* child : expr->children()) {
    collect_column_indices(child, indices);
  }
}

/// @brief Classify `expr` as referencing the left input, the right input, or neither/both.
[[nodiscard]] operand_side classify_operand(gqe::expression const* expr, cudf::size_type num_left)
{
  std::vector<cudf::size_type> indices;
  collect_column_indices(expr, indices);
  // This means it's a cross join with true condition - no need to rewrite
  if (indices.empty()) return operand_side::invalid;
  if (std::all_of(indices.begin(), indices.end(), [&](auto i) { return i < num_left; })) {
    return operand_side::left;
  }
  if (std::all_of(indices.begin(), indices.end(), [&](auto i) { return i >= num_left; })) {
    return operand_side::right;
  }
  return operand_side::invalid;
}

/**
 * @brief Recursively split a join condition into equi-key operand pairs and residual nodes.
 *
 * Mirrors the executor's `parse_join_condition`: AND-nested equalities are flattened, each
 * `EQUAL`/`NULL_EQUALS` contributes its two operands as a pair, and anything else is collected
 * as a residual (non-equality) condition that the rule leaves in place (only remapping its
 * column references).
 *
 * Note: for join conditions like
 * `left_table.a+1 = right_table.b+2 AND left_table.c > right_table.d`
 * It should be an equi-join since we can use hash join and have a post-filter to handle the
 * non-equi parts. However, the current executor doesn't support this and we have to fall
 * back to use cudf's join implementation. So we also skip such inner joins for this rewrite.
 */
void parse_join_condition(gqe::expression* condition,
                          std::vector<std::pair<gqe::expression*, gqe::expression*>>& eq_pairs,
                          std::vector<gqe::expression*>& non_equality)
{
  if (condition->type() == expr_type::binary_op) {
    auto* binary  = static_cast<gqe::binary_op_expression*>(condition);
    auto children = condition->children();
    switch (binary->binary_operator()) {
      case cudf::binary_operator::EQUAL:
      case cudf::binary_operator::NULL_EQUALS:
        eq_pairs.emplace_back(children[0], children[1]);
        return;
      case cudf::binary_operator::LOGICAL_AND:
      case cudf::binary_operator::NULL_LOGICAL_AND:
        parse_join_condition(children[0], eq_pairs, non_equality);
        parse_join_condition(children[1], eq_pairs, non_equality);
        return;
      default: break;
    }
  }
  non_equality.push_back(condition);
}

}  // namespace

class complex_expression_extraction_into_project::apply_visitor
  : public gqe::logical::relation_visitor {
 public:
  apply_visitor(complex_expression_extraction_into_project const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit(gqe::logical::aggregate_relation* agg) override
  {
    visit_children(agg);  // post-order: recurse first
    _rule.extract_aggregate(agg, _rule_applied);
  }

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);  // post-order: recurse first
    _rule.extract_join(join, _rule_applied);
  }

 private:
  complex_expression_extraction_into_project const& _rule;
  bool& _rule_applied;
};

void complex_expression_extraction_into_project::extract_aggregate(
  gqe::logical::aggregate_relation* agg, bool& rule_applied) const
{
  auto const keys     = agg->keys_unsafe();
  auto const measures = agg->measures_unsafe();
  auto const num_keys = static_cast<cudf::size_type>(keys.size());

  // Canonicalize: fire unless the keys/measures are already bare column references at the
  // canonical positions (keys at [0..k-1], measure args at [k..k+m-1]).
  bool is_canonical = true;
  for (cudf::size_type i = 0; is_canonical && i < num_keys; ++i) {
    if (!is_column_reference_at(keys[i], i)) is_canonical = false;
  }
  for (cudf::size_type j = 0; is_canonical && j < static_cast<cudf::size_type>(measures.size());
       ++j) {
    if (!is_column_reference_at(measures[j].second, num_keys + j)) is_canonical = false;
  }
  if (is_canonical) return;

  // Build the child project emitting [keys..., measure-args...] in that order.
  // In GQE, project operator does column reordering, column addition and column deletion,
  // so we have to add all the keys and measures to the project here. In some databases,
  // project operator only does column addition and it doesn't drop any columns.
  std::vector<std::unique_ptr<gqe::expression>> outputs;
  outputs.reserve(keys.size() + measures.size());
  for (auto const* k : keys) {
    outputs.push_back(k->clone());
  }
  for (auto const& [kind, arg] : measures) {
    outputs.push_back(arg->clone());
  }

  // If `child` is already a project, this produces a project-over-project. We intentionally do
  // not collapse the two here: the extra project is near-zero cost given masked-table operator
  // support, and a correct merge would have to re-index any `subquery_expression`s. Collapsing
  // consecutive projects is left as a follow-up (candidate: a generic `merge_consecutive_projects`
  // rule). (issue!387)
  auto child   = agg->children_safe()[0];
  auto project = std::make_shared<gqe::logical::project_relation>(
    std::move(child), agg->subqueries_safe(), std::move(outputs));
  // The aggregate's subqueries are moved to the project (its cloned keys/measures, now the
  // project's outputs, still reference them by index), so clear them from the aggregate.
  clear_subqueries(agg);
  replace_child_at(agg, 0, std::move(project));

  // Rewrite keys -> column_reference(0..num_keys-1) and measure args ->
  // column_reference(num_keys..num_keys+num_measures-1). This relies on
  // rewrite_expressions_visitor::visit(aggregate) traversing keys before measures.
  cudf::size_type counter = 0;
  auto to_column_reference =
    [counter_ptr = &counter](
      gqe::expression*, std::vector<cudf::data_type> const&) -> std::unique_ptr<gqe::expression> {
    return std::make_unique<gqe::column_reference_expression>((*counter_ptr)++);
  };
  rewrite_relation_expressions(agg, to_column_reference, transform_direction::NONE);
  rule_applied = true;
}

void complex_expression_extraction_into_project::extract_join(gqe::logical::join_relation* join,
                                                              bool& rule_applied) const
{
  // Conservative: only plain inner joins, no subqueries in the condition.
  if (join->join_type() != gqe::join_type_type::inner) return;
  auto* condition = join->condition();
  if (condition == nullptr || contains_subquery(condition)) return;

  auto children        = join->children_safe();
  auto const num_left  = children[0]->num_columns();
  auto const num_right = children[1]->num_columns();

  std::vector<std::pair<gqe::expression*, gqe::expression*>> eq_pairs;
  std::vector<gqe::expression*> non_equality;
  parse_join_condition(condition, eq_pairs, non_equality);

  // Skip the entire rewrite when there is any non-equality residual: the moment a non-equi part
  // is present we do not apply this rule (such inner joins fall back to the executor's general
  // path).
  if (!non_equality.empty()) return;
  // Not an equi-join (e.g. cross join / literal-true condition)
  if (eq_pairs.empty()) return;

  auto const num_keys = static_cast<cudf::size_type>(eq_pairs.size());

  // Classify each equality pair into its left and right operand by column-reference index range
  // (robust to operand order, e.g. `r.b == l.a`). Bail (leave the join untouched) if a pair is not
  // cleanly one-left + one-right (an operand referencing both inputs, or a constant).
  std::vector<gqe::expression*> left_keys(num_keys);
  std::vector<gqe::expression*> right_keys(num_keys);
  for (cudf::size_type i = 0; i < num_keys; ++i) {
    auto* op0        = eq_pairs[i].first;
    auto* op1        = eq_pairs[i].second;
    auto const side0 = classify_operand(op0, num_left);
    auto const side1 = classify_operand(op1, num_left);
    if (side0 == operand_side::left && side1 == operand_side::right) {
      left_keys[i]  = op0;
      right_keys[i] = op1;
    } else if (side0 == operand_side::right && side1 == operand_side::left) {
      left_keys[i]  = op1;
      right_keys[i] = op0;
    } else {
      return;  // not cleanly splittable into one left + one right operand
    }
  }

  // Per-side canonical check: a side is canonical when all of its keys are already bare column
  // references at the trailing key positions this rule would produce (the last `num_keys` columns
  // of that side, in order). A complex key fails this (not a bare col-ref), and so does a
  // bare-but-misordered key — either way that side needs a project. Trailing (not leading like the
  // aggregate) because the per-side project passes the original columns through first and then
  // appends the keys; that layout also makes a re-run a no-op.
  bool left_canonical = true;
  for (cudf::size_type i = 0; left_canonical && i < num_keys; ++i) {
    if (!is_column_reference_at(left_keys[i], num_left - num_keys + i)) left_canonical = false;
  }
  bool right_canonical = true;
  for (cudf::size_type i = 0; right_canonical && i < num_keys; ++i) {
    if (!is_column_reference_at(right_keys[i], num_left + num_right - num_keys + i)) {
      right_canonical = false;
    }
  }
  // Both sides already canonical -> nothing to do (idempotent no-op).
  if (left_canonical && right_canonical) return;

  // Number of key columns the left side appends (0 if the left side is left untouched). Only the
  // left-side insertion shifts the indices of the right-side original columns.
  auto const n_left_added = left_canonical ? cudf::size_type{0} : num_keys;

  // Build a per-side project [orig_cols..., keys...] that materializes ALL of that side's keys
  // (bare and complex alike) as trailing columns. If a join input is already a project this
  // produces a project-over-project that we intentionally do not collapse here (see the note in
  // extract_aggregate); collapsing is a follow-up.
  auto build_side_project = [&](std::shared_ptr<gqe::logical::relation> input,
                                cudf::size_type num_orig,
                                std::vector<gqe::expression*> const& keys,
                                cudf::size_type shift) {
    std::vector<std::unique_ptr<gqe::expression>> outputs;
    outputs.reserve(static_cast<std::size_t>(num_orig) + keys.size());
    for (cudf::size_type i = 0; i < num_orig; ++i) {
      outputs.push_back(std::make_unique<gqe::column_reference_expression>(i));
    }
    for (auto const* key : keys) {
      auto cloned = key->clone();
      if (shift != 0) {
        // Right-side operands address the combined schema; shift them into right-local indices.
        auto shift_refs =
          [shift](gqe::expression* e,
                  std::vector<cudf::data_type> const&) -> std::unique_ptr<gqe::expression> {
          if (e->type() == expr_type::column_reference) {
            auto idx = static_cast<gqe::column_reference_expression*>(e)->column_idx();
            return std::make_unique<gqe::column_reference_expression>(idx - shift);
          }
          return nullptr;
        };
        if (auto replaced =
              rewrite_expression(cloned.get(), shift_refs, transform_direction::DOWN)) {
          cloned = std::move(replaced);
        }
      }
      outputs.push_back(std::move(cloned));
    }
    return std::make_shared<gqe::logical::project_relation>(
      std::move(input), std::vector<std::shared_ptr<gqe::logical::relation>>{}, std::move(outputs));
  };

  // Project only the side(s) that are not already canonical.
  if (!left_canonical) {
    replace_child_at(join, 0, build_side_project(children[0], num_left, left_keys, /*shift=*/0));
  }
  if (!right_canonical) {
    replace_child_at(
      join, 1, build_side_project(children[1], num_right, right_keys, /*shift=*/num_left));
  }

  // Map every key operand (by pointer) to its target column in the rewritten combined schema. A
  // canonical side keeps its keys at their existing trailing positions; a projected side moves its
  // keys to the newly appended trailing columns. Right-side targets also account for the left-side
  // columns inserted before them.
  std::unordered_map<gqe::expression const*, cudf::size_type> operand_target;
  for (cudf::size_type i = 0; i < num_keys; ++i) {
    operand_target[left_keys[i]] = left_canonical ? (num_left - num_keys + i) : (num_left + i);
    operand_target[right_keys[i]] =
      (num_left + n_left_added) + (right_canonical ? (num_right - num_keys + i) : (num_right + i));
  }
  auto remap = [&operand_target](
                 gqe::expression* e,
                 std::vector<cudf::data_type> const&) -> std::unique_ptr<gqe::expression> {
    if (auto it = operand_target.find(e); it != operand_target.end()) {
      return std::make_unique<gqe::column_reference_expression>(it->second);
    }
    return nullptr;
  };
  rewrite_relation_expressions(join, remap, transform_direction::DOWN);

  // Remap projection indices past the left-side keys inserted before the right-original columns.
  // (Right-side appended keys sit at the very end and are never part of projection_indices.)
  std::vector<cudf::size_type> new_projection;
  new_projection.reserve(join->projection_indices().size());
  for (auto idx : join->projection_indices()) {
    new_projection.push_back(idx < num_left ? idx : idx + n_left_added);
  }
  set_join_projection_indices(join, std::move(new_projection));

  rule_applied = true;
}

std::shared_ptr<gqe::logical::relation> complex_expression_extraction_into_project::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  // The visitor processes each aggregate/join node itself (post-order), so the root is handled
  // by `accept` without special-casing.
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
