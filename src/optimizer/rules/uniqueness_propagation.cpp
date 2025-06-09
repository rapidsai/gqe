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

#include <gqe/catalog.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/optimizer/rules/uniqueness_propagation.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>

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

class uniqueness_propagation_helpers : public gqe::optimizer::uniqueness_propagation {
 public:
  static void propagate_uniqueness(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto children = rel->children_unsafe();
    assert(children.size() == 1);
    // Get input indices that are unique
    auto input_unique_cols = children[0]->relation_traits().properties().get_columns_with_property(
      gqe::optimizer::column_property::property_id::unique);
    for (auto idx : input_unique_cols) {
      gqe::optimizer::optimization_rule::set_relation_property(
        rel, idx, gqe::optimizer::column_property::property_id::unique);
      rule_applied = true;
    }
  }

  static void propagate_uniqueness_aggregate(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto aggregate = dynamic_cast<gqe::logical::aggregate_relation*>(rel);
    // Aggregation put the key columns first in the output
    auto number_of_keys = aggregate->keys_unsafe().size();
    if (number_of_keys == 0) {
      // Reduction results in a single row. All output columns are unique
      auto num_cols = aggregate->measures_unsafe().size();  // TODO: implement num cols function?
      for (size_t idx = 0; idx < num_cols; idx++) {
        set_relation_property(aggregate, idx, gqe::optimizer::column_property::property_id::unique);
        rule_applied = true;
      }
    } else if (number_of_keys == 1) {
      set_relation_property(aggregate, 0, gqe::optimizer::column_property::property_id::unique);
      rule_applied = true;
    } else {
      // Maintain uniqueness for all keys based on the input
      auto key_exprs = aggregate->keys_unsafe();
      auto children  = aggregate->children_unsafe();
      assert(children.size() == 1);
      auto input_unique_cols =
        children[0]->relation_traits().properties().get_columns_with_property(
          gqe::optimizer::column_property::property_id::unique);
      for (size_t idx = 0; idx < key_exprs.size(); idx++) {
        if (is_unique_expression(key_exprs[idx], input_unique_cols, rel->data_types())) {
          set_relation_property(
            aggregate, idx, gqe::optimizer::column_property::property_id::unique);
          rule_applied = true;
        }
      }
    }
  }

  static void propagate_uniqueness_filter(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto filter   = dynamic_cast<gqe::logical::filter_relation*>(rel);
    auto children = filter->children_unsafe();
    assert(children.size() == 1);
    // Get input indices that are unique
    auto input_unique_cols = children[0]->relation_traits().properties().get_columns_with_property(
      gqe::optimizer::column_property::property_id::unique);

    // propagate uniqueness property to projected columns
    auto projection_indices = filter->projection_indices();
    for (uint32_t output_col_idx = 0; output_col_idx < projection_indices.size();
         ++output_col_idx) {
      if (input_unique_cols.count(projection_indices[output_col_idx])) {
        gqe::optimizer::optimization_rule::set_relation_property(
          filter, output_col_idx, gqe::optimizer::column_property::property_id::unique);
        rule_applied = true;
      }
    }
  }

  static void propagate_uniqueness_join(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto join      = dynamic_cast<gqe::logical::join_relation*>(rel);
    auto condition = join->condition();
    auto children  = join->children_unsafe();
    assert(children.size() == 2);
    // Get left indices that are unique
    auto input_unique_cols = children[0]->relation_traits().properties().get_columns_with_property(
      gqe::optimizer::column_property::property_id::unique);
    // Merge offseted right indices that are unique with the left indices
    auto n_left_cols = children[0]->num_columns();
    for (auto ridx : children[1]->relation_traits().properties().get_columns_with_property(
           gqe::optimizer::column_property::property_id::unique)) {
      input_unique_cols.insert(ridx + n_left_cols);
    }

    bool propagate_left, propagate_right;
    switch (join->join_type()) {
      case gqe::join_type_type::inner: {
        std::tie(propagate_left, propagate_right) =
          check_join_condition_for_propagation(condition, input_unique_cols, n_left_cols);
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
        // TODO: specify GQE's semantics regarding uniquness of NULL values (background on ambiguity
        // in SQL standard: https://sqlite.org/nulls.html) to potentially allow propagation for
        // outer (e.g. left, full) joins, as well as for RHS columns of single join
    }

    if (!propagate_left && !propagate_right) return;
    // Propagate uniqueness to projected columns
    auto projection_indices = join->projection_indices();
    for (uint32_t output_col_idx = 0; output_col_idx < projection_indices.size();
         ++output_col_idx) {
      auto input_col_idx = projection_indices[output_col_idx];
      if ((input_col_idx < n_left_cols && propagate_left) ||
          (input_col_idx >= n_left_cols && propagate_right)) {
        if (input_unique_cols.count(input_col_idx)) {
          gqe::optimizer::optimization_rule::set_relation_property(
            join, output_col_idx, gqe::optimizer::column_property::property_id::unique);
          rule_applied = true;
        }
      }
    }
  }

  static void propagate_uniqueness_project(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto children = rel->children_unsafe();
    assert(children.size() == 1);
    // Get input indices that are unique
    auto input_unique_cols = children[0]->relation_traits().properties().get_columns_with_property(
      gqe::optimizer::column_property::property_id::unique);
    auto proj               = dynamic_cast<gqe::logical::project_relation*>(rel);
    auto output_expressions = proj->output_expressions_unsafe();
    for (size_t i = 0; i < output_expressions.size(); i++) {
      auto expr = output_expressions[i];
      if (expr->type() == gqe::expression::expression_type::column_reference) {
        auto col_ref = dynamic_cast<gqe::column_reference_expression*>(expr);
        auto col_idx = col_ref->column_idx();
        if (input_unique_cols.count(col_idx)) {
          gqe::optimizer::optimization_rule::set_relation_property(
            proj, i, gqe::optimizer::column_property::property_id::unique);
          rule_applied = true;
        }
      }
    }
  }

  static void propagate_uniqueness_set(gqe::logical::relation* rel, bool& rule_applied)
  {
    auto children = rel->children_unsafe();
    assert(children.size() == 2);
    // Get input indices that are unique
    auto input_unique_cols_l =
      children[0]->relation_traits().properties().get_columns_with_property(
        gqe::optimizer::column_property::property_id::unique);
    auto input_unique_cols_r =
      children[1]->relation_traits().properties().get_columns_with_property(
        gqe::optimizer::column_property::property_id::unique);
    for (auto idx : input_unique_cols_l) {
      if (input_unique_cols_r.count(idx)) {
        gqe::optimizer::optimization_rule::set_relation_property(
          rel, idx, gqe::optimizer::column_property::property_id::unique);
        rule_applied = true;
      }
    }
  }
};

}  // namespace

std::shared_ptr<gqe::logical::relation> gqe::optimizer::uniqueness_propagation::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  switch (logical_relation->type()) {
    case relation_t::aggregate: {
      uniqueness_propagation_helpers::propagate_uniqueness_aggregate(logical_relation.get(),
                                                                     rule_applied);
      break;
    }
    case relation_t::join: {
      uniqueness_propagation_helpers::propagate_uniqueness_join(logical_relation.get(),
                                                                rule_applied);
      break;
    }
    case relation_t::read: {
      // Get column uniqueness from catalog and add to the list in the same order as output
      auto read    = dynamic_cast<logical::read_relation*>(logical_relation.get());
      auto catalog = this->get_catalog();
      assert(catalog);
      cudf::size_type output_col_idx = 0;
      for (const auto& col_name : read->column_names()) {  // Iterate w.r.t output column order
        auto is_unique = catalog->column_is_unique(read->table_name(), col_name);
        if (is_unique) {
          set_relation_property(read, output_col_idx, column_property::property_id::unique);
        }
        output_col_idx++;
      }
      rule_applied = true;
      break;
    }
    case relation_t::set: {
      auto set = dynamic_cast<logical::set_relation*>(logical_relation.get());
      switch (set->set_operator()) {
        case logical::set_relation::set_union: {
          // Propagate uniqueness only in trivial case of single-column union
          if (set->data_types().size() == 1)
            uniqueness_propagation_helpers::propagate_uniqueness_set(logical_relation.get(),
                                                                     rule_applied);
          // Otherwise drop uniqueness of each input column by not propagating uniqueness
          // TODO: The composit of all columns is unique, when composite column uniqueness is
          // supported
          break;
        }
        case logical::set_relation::set_union_all: {
          // Drop uniqueness of each input column by not propagating uniqueness
          break;
        }
        case logical::set_relation::set_intersect: {
          // All columns keep their uniqueness
          // TODO: The composit of all columns is unique, when composite column uniqueness is
          // supported Propagate uniqueness to the output if both sides are unique
          uniqueness_propagation_helpers::propagate_uniqueness_set(logical_relation.get(),
                                                                   rule_applied);
          break;
        }
        case logical::set_relation::set_minus: {
          // All columns keep their uniqueness
          // TODO: The composit of all columns is unique, when composite column uniqueness is
          // supported Propagate uniqueness to the output if both sides are unique
          uniqueness_propagation_helpers::propagate_uniqueness_set(logical_relation.get(),
                                                                   rule_applied);
          break;
        }
      }
      break;
    }
    case relation_t::project: {
      // Propagate existing uniqueness. No addition of rows, but there could be column order changes
      // and operation performed on columns.
      uniqueness_propagation_helpers::propagate_uniqueness_project(logical_relation.get(),
                                                                   rule_applied);
      break;
    }
    case relation_t::filter: {
      // Propagate existing uniqueness for projected columns. No addition of rows
      uniqueness_propagation_helpers::propagate_uniqueness_filter(logical_relation.get(),
                                                                  rule_applied);
      break;
    }
    case relation_t::fetch:   // Propagate existing uniqueness. No addition of rows
    case relation_t::sort:    // Propagate existing uniqueness. No addition of rows
    case relation_t::window:  // Propagate existing uniqueness. There should not be changes in
                              // indices since window function should just add a column to the right
    case relation_t::write: {
      // All columns keep their uniqueness
      // Propagate uniqueness to the output
      uniqueness_propagation_helpers::propagate_uniqueness(logical_relation.get(), rule_applied);
      break;
    }
    case relation_t::user_defined:
    default: {
      return logical_relation;
    }
  }
  return logical_relation;
}
