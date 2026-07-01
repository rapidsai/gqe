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

#include <gqe/optimizer/rules/uniqueness_propagation.hpp>

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
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
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {
using expression_t = gqe::expression::expression_type;
bool is_unique_expression(gqe::expression* expr,
                          std::unordered_set<cudf::size_type> unique_input_cols,
                          std::vector<cudf::data_type> const& column_types)
{
  switch (expr->type()) {
    case expression_t::column_reference: {
      auto col_ref = dynamic_cast<gqe::column_reference_expression*>(expr);
      auto col_idx = col_ref->column_idx();
      if (unique_input_cols.count(col_idx)) { return true; }
      return false;
    }
    case expression_t::binary_op: {
      auto binary_op = dynamic_cast<gqe::binary_op_expression*>(expr);
      auto op        = binary_op->binary_operator();
      if (op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB) {
        auto operands = binary_op->children();
        assert(operands.size() == 2);
        // non-integer addition/subtraction does not preserve uniqueness
        if (!cudf::is_integral(operands[0]->data_type(column_types)) ||
            !cudf::is_integral(operands[1]->data_type(column_types))) {
          return false;
        }
        if (operands[0]->type() == expression_t::literal) {
          return is_unique_expression(operands[1], unique_input_cols, column_types);
        } else if (operands[1]->type() == expression_t::literal) {
          return is_unique_expression(operands[0], unique_input_cols, column_types);
        }
      }
      return false;
    }
    case expression_t::cast: {
      // TODO: Determine uniqueness based on the conversion types
      return false;
    }
    case expression_t::gen_ident_col: return true;
    case expression_t::unary_op: {
      auto unary_op = dynamic_cast<gqe::unary_op_expression*>(expr);
      auto op       = unary_op->unary_operator();
      // Only unary operator which preserves uniqueness
      if (op == cudf::unary_operator::BIT_INVERT) return true;
      return false;
    }
    // No uniqueness guarantee can be derived from the following expression types
    case expression_t::if_then_else:
    case expression_t::is_null:
    case expression_t::literal:
    case expression_t::scalar_function:
    case expression_t::subquery: return false;
    default: throw std::runtime_error("Unsupported expression type");
  }
}

/**
 * @brief Collect the set of (left_global_idx, right_global_idx) equijoin column pairs implied by
 * the predicate.
 *
 * AND: union of pairs from both children (all equalities hold for every output row).
 * OR: intersection — only pairs present in both branches count (an equality is guaranteed for an
 * output row only if it holds under either disjunct).
 * EQUAL with both operands column references and one on each side: a single pair.
 * Other expressions contribute no pairs.
 */
std::set<std::pair<cudf::size_type, cudf::size_type>> collect_equijoin_pairs(
  gqe::expression* expr, cudf::size_type n_left_cols)
{
  std::set<std::pair<cudf::size_type, cudf::size_type>> pairs;
  if (expr->type() != gqe::expression::expression_type::binary_op) return pairs;
  auto bin_op   = static_cast<gqe::binary_op_expression*>(expr);
  auto children = bin_op->children();
  assert(children.size() == 2);
  auto op = bin_op->binary_operator();

  if (op == cudf::binary_operator::EQUAL) {
    if (children[0]->type() == gqe::expression::expression_type::column_reference &&
        children[1]->type() == gqe::expression::expression_type::column_reference) {
      auto c0 = static_cast<gqe::column_reference_expression*>(children[0])->column_idx();
      auto c1 = static_cast<gqe::column_reference_expression*>(children[1])->column_idx();
      // if c0 and c1 are from different child relations
      if ((c0 < n_left_cols) != (c1 < n_left_cols)) {
        pairs.emplace(std::min(c0, c1), std::max(c0, c1));
      }
    }
  } else if (op == cudf::binary_operator::LOGICAL_AND ||
             op == cudf::binary_operator::NULL_LOGICAL_AND) {
    auto l_pairs = collect_equijoin_pairs(children[0], n_left_cols);
    auto r_pairs = collect_equijoin_pairs(children[1], n_left_cols);
    pairs.insert(l_pairs.begin(), l_pairs.end());
    pairs.insert(r_pairs.begin(), r_pairs.end());
  } else if (op == cudf::binary_operator::LOGICAL_OR ||
             op == cudf::binary_operator::NULL_LOGICAL_OR) {
    auto l_pairs = collect_equijoin_pairs(children[0], n_left_cols);
    auto r_pairs = collect_equijoin_pairs(children[1], n_left_cols);
    std::set_intersection(l_pairs.begin(),
                          l_pairs.end(),
                          r_pairs.begin(),
                          r_pairs.end(),
                          std::inserter(pairs, pairs.begin()));
  }
  return pairs;
}

/**
 * @brief Determine whether input columns should propagate their uniqueness properties to output
 * columns based on the join condition
 *
 * @note For an EQUAL expression, if both child expressions are column references and one of them is
 * unique, all columns on the other side of the join will not have their rows duplicated, thus the
 * other side's columns get to keep their uniqueness. For a LOGICAL_AND expression, the function is
 * called recursively on each child expression and either child expression allowing propagation for
 * a particular side is sufficient for propagation of that side. For a LOGICAL_OR expression, the
 * function is called recursively on each child expression and propagation for a particular side
 * requires both child expressions to allow propagation for that side. For other types of
 * expression, we do not know enough to determine column uniqueness, thus, we drop all uniqueness
 * since there is a possibility of row duplication.
 *
 * @todo For a NULL_EQUALS expression, if the both keys are not nullable, then there is no
 * possibility of null matches so all columns get to keep their uniqueness.
 *
 * @param[in] expr the join condition or any of its sub-expression
 * @param[in] unique_input_cols indices of input columns with uniqueness property
 * @param[in] n_left_cols number of columns in the left child relation; used to determine if a
 * column reference is from the left or right child relation
 * @return pair of flags to propagate uniqueness properties of left/right columns
 */
std::pair<bool, bool> check_join_condition_for_propagation(
  gqe::expression* expr,
  const std::unordered_set<cudf::size_type>& unique_input_cols,
  const cudf::size_type n_left_cols)
{
  // default to false
  bool propagate_left{false}, propagate_right{false};

  if (expr->type() == gqe::expression::expression_type::binary_op) {
    auto bin_op   = static_cast<gqe::binary_op_expression*>(expr);
    auto children = bin_op->children();
    assert(children.size() == 2);
    if (bin_op->binary_operator() == cudf::binary_operator::EQUAL) {
      // check operands of equality are both column refs
      if (children[0]->type() == gqe::expression::expression_type::column_reference &&
          children[1]->type() == gqe::expression::expression_type::column_reference) {
        auto col_idx_0 = static_cast<gqe::column_reference_expression*>(children[0])->column_idx();
        auto col_idx_1 = static_cast<gqe::column_reference_expression*>(children[1])->column_idx();
        // warn if both child expressions are columns on the same side of the join relation
        if ((col_idx_0 < n_left_cols) == (col_idx_1 < n_left_cols))
          GQE_LOG_WARN(
            "Join predicate (sub)expression comparing column references from same child of join "
            "relation!");

        // Propagate uniqueness if applicable
        if (unique_input_cols.count(col_idx_0)) {
          ((col_idx_0 < n_left_cols) ? propagate_right : propagate_left) = true;
        }
        if (unique_input_cols.count(col_idx_1)) {
          ((col_idx_1 < n_left_cols) ? propagate_right : propagate_left) = true;
        }
      }
    } else if (bin_op->binary_operator() == cudf::binary_operator::LOGICAL_AND ||
               bin_op->binary_operator() == cudf::binary_operator::NULL_LOGICAL_AND) {
      // check child expressions: either side of AND operator can, on its own, allow propagation
      std::tie(propagate_left, propagate_right) =
        check_join_condition_for_propagation(children[0], unique_input_cols, n_left_cols);
      if (!propagate_left || !propagate_right) {
        auto [propagate_left_tmp, propagate_right_tmp] =
          check_join_condition_for_propagation(children[1], unique_input_cols, n_left_cols);
        propagate_left |= propagate_left_tmp;
        propagate_right |= propagate_right_tmp;
      }
    } else if (bin_op->binary_operator() == cudf::binary_operator::LOGICAL_OR ||
               bin_op->binary_operator() == cudf::binary_operator::NULL_LOGICAL_OR) {
      // check child expressions: either side of OR operator can, on its own, forbid propagation
      std::tie(propagate_left, propagate_right) =
        check_join_condition_for_propagation(children[0], unique_input_cols, n_left_cols);
      if (propagate_left || propagate_right) {
        auto [propagate_left_tmp, propagate_right_tmp] =
          check_join_condition_for_propagation(children[1], unique_input_cols, n_left_cols);
        propagate_left &= propagate_left_tmp;
        propagate_right &= propagate_right_tmp;
      }
    }
    // TODO: can propagate uniqueness if operator is NULL_EQUALS and columns are not nullable
  }

  return std::make_pair(propagate_left, propagate_right);
}

// Remap a child unique key through an (input_idx → {output_idx, ...}) map, emitting the
// cartesian product of per-component output positions via `emit`. Returns immediately
// (emitting nothing) if any key component has no mapped output — the key is dropped.
//
// Duplicate output positions for the same input column arise whenever a projection
// (explicit `project_relation`, or `projection_indices` on filter/join after
// projection_pushdown) materialises the same input column at multiple output slots,
// e.g. `SELECT pk, pk FROM t`. All such output positions are individually unique, and
// for composite keys every combination of per-component positions is unique too, so we
// enumerate the whole cartesian product and emit each tuple as a separate key-set.
//
// The enumeration uses a mixed-radix counter: `idx[k]` is the current choice within
// `opts[k]` (the output positions for component `k`), incremented odometer-style with
// right-to-left carry. When the carry propagates past the leftmost digit the whole
// product has been visited and we stop.
template <typename Emit>
void emit_remapped_keys(
  std::vector<cudf::size_type> const& key,
  std::unordered_map<cudf::size_type, std::vector<cudf::size_type>> const& in_to_outs,
  Emit&& emit)
{
  std::vector<std::vector<cudf::size_type> const*> opts;
  opts.reserve(key.size());
  for (auto in_idx : key) {
    auto it = in_to_outs.find(in_idx);
    if (it == in_to_outs.end()) return;  // component not projected → drop the whole key
    opts.push_back(&it->second);
  }

  std::vector<size_t> idx(opts.size(), 0);
  for (;;) {
    std::vector<cudf::size_type> remapped;
    remapped.reserve(opts.size());
    for (size_t k = 0; k < opts.size(); k++)
      remapped.push_back((*opts[k])[idx[k]]);
    emit(std::move(remapped));

    bool carry = true;
    for (size_t k = opts.size(); k-- > 0 && carry;) {
      if (++idx[k] < opts[k]->size())
        carry = false;
      else
        idx[k] = 0;
    }
    if (carry) break;
  }
}

}  // namespace

namespace gqe::optimizer {

class uniqueness_propagation::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(uniqueness_propagation const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit(gqe::logical::aggregate_relation* aggregate) override
  {
    visit_children(aggregate);  // post-order: children before parent
    // Aggregation puts the key columns first in the output
    auto number_of_keys = aggregate->keys_unsafe().size();
    if (number_of_keys == 0) {
      // Reduction results in a single row. All output columns are individually unique.
      auto num_cols = aggregate->measures_unsafe().size();  // TODO: implement num cols function?
      for (size_t idx = 0; idx < num_cols; idx++) {
        uniqueness_propagation::add_relation_unique_key(aggregate,
                                                        {static_cast<cudf::size_type>(idx)});
        _rule_applied = true;
      }
    } else if (number_of_keys == 1) {
      uniqueness_propagation::add_relation_unique_key(aggregate, {0});
      _rule_applied = true;
    } else {
      // The full tuple of group-by columns is always unique on the output (GROUP BY collapses
      // duplicate tuples), regardless of input uniqueness. Emit {0, 1, ..., N-1} as a composite
      // unique key. This generalizes the unconditional {0} emission in the N == 1 branch above.
      auto key_exprs = aggregate->keys_unsafe();
      std::vector<cudf::size_type> tuple_key;
      tuple_key.reserve(key_exprs.size());
      for (size_t idx = 0; idx < key_exprs.size(); idx++) {
        tuple_key.push_back(static_cast<cudf::size_type>(idx));
      }
      uniqueness_propagation::add_relation_unique_key(aggregate, std::move(tuple_key));
      _rule_applied = true;

      // Additionally, if a group-by expression evaluates to an individually-unique value
      // (e.g. it references an individually-unique input column), emit a singleton unique key
      // {idx} — a strictly stronger claim than the composite above.
      // Only singleton input keys contribute here; composite input keys (e.g.
      // (ps_partkey, ps_suppkey)) do not make either component individually unique.
      auto children = aggregate->children_unsafe();
      assert(children.size() == 1);
      std::unordered_set<cudf::size_type> input_unique_cols;
      for (auto const& key : children[0]->relation_traits().properties().unique_keys()) {
        if (key.size() == 1) input_unique_cols.insert(key[0]);
      }
      for (size_t idx = 0; idx < key_exprs.size(); idx++) {
        if (is_unique_expression(key_exprs[idx], input_unique_cols, aggregate->data_types())) {
          uniqueness_propagation::add_relation_unique_key(aggregate,
                                                          {static_cast<cudf::size_type>(idx)});
        }
      }
    }
  }

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);  // post-order: children before parent
    auto condition = join->condition();
    auto children  = join->children_unsafe();
    assert(children.size() == 2);
    auto n_left_cols       = children[0]->num_columns();
    auto const& left_keys  = children[0]->relation_traits().properties().unique_keys();
    auto const& right_keys = children[1]->relation_traits().properties().unique_keys();

    // Build per-column unique set from singletons (for check_join_condition_for_propagation)
    std::unordered_set<cudf::size_type> input_unique_cols;
    for (auto const& key : left_keys) {
      if (key.size() == 1) input_unique_cols.insert(key[0]);
    }
    for (auto const& key : right_keys) {
      if (key.size() == 1) input_unique_cols.insert(key[0] + n_left_cols);
    }

    bool propagate_left = false, propagate_right = false;
    switch (join->join_type()) {
      case gqe::join_type_type::inner: {
        // Deal with the single-column child unique keys to see if they can propagate to the
        // output.
        std::tie(propagate_left, propagate_right) =
          check_join_condition_for_propagation(condition, input_unique_cols, n_left_cols);
        // Composite child keys can also drive propagation: if every component of a LEFT composite
        // key is equijoined to some right column, each right row matches ≤ 1 left row →
        // propagate_right. Symmetric for right composite keys → propagate_left.
        if (!propagate_left || !propagate_right) {
          auto pairs = collect_equijoin_pairs(condition, n_left_cols);
          std::unordered_set<cudf::size_type> left_covered, right_covered_local;
          for (auto const& p : pairs) {
            left_covered.insert(p.first);
            right_covered_local.insert(p.second - n_left_cols);
          }
          auto all_covered = [](auto const& key, auto const& covered) {
            if (key.empty()) return false;
            for (auto c : key) {
              if (!covered.count(c)) return false;
            }
            return true;
          };
          if (!propagate_right) {
            for (auto const& key : left_keys) {
              if (key.size() >= 2 && all_covered(key, left_covered)) {
                propagate_right = true;
                break;
              }
            }
          }
          if (!propagate_left) {
            for (auto const& key : right_keys) {
              if (key.size() >= 2 && all_covered(key, right_covered_local)) {
                propagate_left = true;
                break;
              }
            }
          }
        }
        break;
      }
      // the following join types do not duplicate rows in the LHS columns but should not
      // propagate uniqueness for the RHS columns
      case gqe::join_type_type::left_semi:  // RHS columns do not appear in output
      case gqe::join_type_type::left_anti:  // RHS columns do not appear in output
      case gqe::join_type_type::single: {   // see TODO item below regarding NULL values
        propagate_left  = true;
        propagate_right = false;
        break;
      }
      // do not propagate column uniqueness for all other join types
      default:
        return;
        // TODO: specify GQE's semantics regarding uniquness of NULL values (background on
        // ambiguity in SQL standard: https://sqlite.org/nulls.html) to potentially allow
        // propagation for outer (e.g. left, full) joins, as well as for RHS columns of single join
    }

    if (!propagate_left && !propagate_right) return;
    // Propagate uniqueness: remap every child key (singleton or composite) through
    // projection_indices. A key is emitted iff all its components survive projection and
    // come from a propagating side. A join's projection_indices may list the same input
    // column multiple times (e.g. after projection_pushdown), so we record every output
    // slot per input column and enumerate the cartesian product for composite keys.
    auto projection_indices = join->projection_indices();
    std::unordered_map<cudf::size_type, std::vector<cudf::size_type>> in_to_outs;
    for (uint32_t o = 0; o < projection_indices.size(); ++o)
      in_to_outs[projection_indices[o]].push_back(static_cast<cudf::size_type>(o));

    auto try_propagate_key = [&](std::vector<cudf::size_type> const& key, bool from_left) {
      if (from_left ? !propagate_left : !propagate_right) return;
      // Child keys are in child-local coordinates; join's projection_indices are in global
      // coordinates (left cols [0, n_left_cols), right cols [n_left_cols, ...)). Shift
      // right keys before lookup.
      std::vector<cudf::size_type> key_global;
      key_global.reserve(key.size());
      for (auto c : key) {
        key_global.push_back(from_left ? c : static_cast<cudf::size_type>(c + n_left_cols));
      }
      emit_remapped_keys(key_global, in_to_outs, [&](std::vector<cudf::size_type> remapped) {
        uniqueness_propagation::add_relation_unique_key(join, std::move(remapped));
        _rule_applied = true;
      });
    };
    for (auto const& key : children[0]->relation_traits().properties().unique_keys())
      try_propagate_key(key, true);
    for (auto const& key : children[1]->relation_traits().properties().unique_keys())
      try_propagate_key(key, false);
  }

  void visit(gqe::logical::read_relation* read) override
  {
    visit_children(read);  // post-order (read has no children, so this is a no-op)
    auto catalog = _rule.get_catalog();
    assert(catalog);
    // Seed all unique key-sets (singletons and composites) from the catalog.
    // Translate column names → output column indices; skip keys with non-output columns.
    auto const& output_names = read->column_names();
    for (auto const& key_names : catalog->unique_keys(read->table_name())) {
      std::vector<cudf::size_type> key_indices;
      key_indices.reserve(key_names.size());
      bool all_present = true;
      for (auto const& name : key_names) {
        auto it = std::find(output_names.begin(), output_names.end(), name);
        if (it == output_names.end()) {
          all_present = false;
          break;
        }
        key_indices.push_back(static_cast<cudf::size_type>(it - output_names.begin()));
      }
      if (all_present) {
        uniqueness_propagation::add_relation_unique_key(read, std::move(key_indices));
      }
    }
    _rule_applied = true;
  }

  void visit(gqe::logical::set_relation* set_op) override
  {
    visit_children(set_op);  // post-order: children before parent
    switch (set_op->set_operator()) {
      case gqe::logical::set_relation::set_union: {
        // UNION DISTINCT deduplicates by full row: the composite {0..N-1} is always unique.
        // No narrower key can be inferred: a key unique on each input independently may still
        // have overlapping values across the two inputs (e.g. left {1,'x'} and right {1,'y'}
        // each have unique col-0, yet the union output has duplicate col-0 value 1).
        assert(set_op->children_unsafe().size() == 2);
        auto const num_cols = set_op->data_types().size();
        std::vector<cudf::size_type> tuple_key;
        tuple_key.reserve(num_cols);
        for (size_t i = 0; i < num_cols; i++)
          tuple_key.push_back(static_cast<cudf::size_type>(i));
        uniqueness_propagation::add_relation_unique_key(set_op, std::move(tuple_key));
        _rule_applied = true;
        break;
      }
      case gqe::logical::set_relation::set_union_all: {
        // UNION ALL can duplicate rows — drop all uniqueness.
        break;
      }
      case gqe::logical::set_relation::set_intersect:
      case gqe::logical::set_relation::set_minus: {
        // INTERSECT/MINUS: the result is a (dedup'd) subset of the left input. Every left unique
        // key propagates verbatim — singletons and composites alike. No index remapping needed
        // because set-op inputs are positionally aligned.
        auto children = set_op->children_unsafe();
        assert(children.size() == 2);
        for (auto const& key : children[0]->relation_traits().properties().unique_keys()) {
          uniqueness_propagation::add_relation_unique_key(set_op, key);
          _rule_applied = true;
        }
        break;
      }
    }
  }

  void visit(gqe::logical::project_relation* project) override
  {
    visit_children(project);  // post-order: children before parent
    auto children = project->children_unsafe();
    assert(children.size() == 1);
    auto const& child_props = children[0]->relation_traits().properties();

    auto output_expressions = project->output_expressions_unsafe();

    // Build input_idx → {output_idx, ...} map. A column projected multiple times (SELECT a, a)
    // gets all output positions recorded so every valid remap is emitted.
    std::unordered_map<cudf::size_type, std::vector<cudf::size_type>> input_to_outputs;
    for (size_t i = 0; i < output_expressions.size(); i++) {
      auto expr = output_expressions[i];
      if (expr->type() == gqe::expression::expression_type::column_reference) {
        auto col_ref = dynamic_cast<gqe::column_reference_expression*>(expr);
        input_to_outputs[static_cast<cudf::size_type>(col_ref->column_idx())].push_back(
          static_cast<cudf::size_type>(i));
      }
    }

    // For each child key-set emit the cartesian product of per-component output positions.
    // Singletons and composites flow through the same helper.
    for (auto const& key : child_props.unique_keys()) {
      emit_remapped_keys(key, input_to_outputs, [&](std::vector<cudf::size_type> remapped) {
        uniqueness_propagation::add_relation_unique_key(project, std::move(remapped));
        _rule_applied = true;
      });
    }
  }

  void visit(gqe::logical::filter_relation* filter) override
  {
    visit_children(filter);  // post-order: children before parent
    auto children = filter->children_unsafe();
    assert(children.size() == 1);
    auto const& child_props = children[0]->relation_traits().properties();

    // For each child unique key, propagate it to the output iff every component column
    // is still present in the filter's projection. A filter's projection_indices may list
    // the same input column multiple times (e.g. after projection_pushdown folds a
    // `SELECT pk, pk` into the filter), so we record every output slot per input column
    // and enumerate the cartesian product.
    auto projection_indices = filter->projection_indices();
    std::unordered_map<cudf::size_type, std::vector<cudf::size_type>> input_to_outputs;
    for (uint32_t out = 0; out < projection_indices.size(); ++out) {
      input_to_outputs[projection_indices[out]].push_back(static_cast<cudf::size_type>(out));
    }
    for (auto const& key : child_props.unique_keys()) {
      emit_remapped_keys(key, input_to_outputs, [&](std::vector<cudf::size_type> remapped) {
        uniqueness_propagation::add_relation_unique_key(filter, std::move(remapped));
        _rule_applied = true;
      });
    }
  }

  // fetch / sort / window / write all propagate child uniqueness verbatim — no row
  // duplication, no column reordering. Window appends a column to the right, but existing
  // unique-key indices remain valid.
  void visit(gqe::logical::fetch_relation* fetch) override { _propagate_single_child(fetch); }
  void visit(gqe::logical::sort_relation* sort) override { _propagate_single_child(sort); }
  void visit(gqe::logical::window_relation* window) override { _propagate_single_child(window); }
  void visit(gqe::logical::write_relation* write) override { _propagate_single_child(write); }

  void visit(gqe::logical::user_defined_relation* user_defined) override
  {
    visit_children(user_defined);  // post-order (no uniqueness inferred from user-defined output)
  }

 private:
  void _propagate_single_child(gqe::logical::relation* rel)
  {
    visit_children(rel);  // post-order: recurse before propagating
    auto children = rel->children_unsafe();
    assert(children.size() == 1);
    for (auto const& key : children[0]->relation_traits().properties().unique_keys()) {
      uniqueness_propagation::add_relation_unique_key(rel, key);
      _rule_applied = true;
    }
  }

  uniqueness_propagation const& _rule;
  bool& _rule_applied;
};

std::shared_ptr<gqe::logical::relation> uniqueness_propagation::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
