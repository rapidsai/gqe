/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
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
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>

#include <numeric>

namespace gqe {

std::shared_ptr<physical::relation> physical_plan_builder::build(
  logical::relation const* logical_relation)
{
  // Search the cache to see whether the physical relation has already been generated
  auto cache_iter = _cache.find(logical_relation);
  if (cache_iter != _cache.end()) return std::shared_ptr<physical::relation>(cache_iter->second);

  // Recursively transform the children logical relations into physical relations
  auto const children_logical   = logical_relation->children_unsafe();
  auto const subqueries_logical = logical_relation->subqueries_unsafe();

  std::vector<std::shared_ptr<physical::relation>> children_physical;
  std::vector<std::shared_ptr<physical::relation>> subqueries_physical;
  children_physical.reserve(children_logical.size());
  subqueries_physical.reserve(subqueries_logical.size());

  for (auto const child_logical : children_logical)
    children_physical.push_back(build(child_logical));

  for (auto const subquery_logical : subqueries_logical)
    subqueries_physical.push_back(build(subquery_logical));

  // Transform the current logical relation depending on the relation type
  std::shared_ptr<physical::relation> out_physical_relation;

  switch (logical_relation->type()) {
    case logical::relation::relation_type::read: {
      auto const logical_read_relation =
        dynamic_cast<logical::read_relation const*>(logical_relation);
      auto const partial_filter_ptr = logical_read_relation->partial_filter_unsafe();

      out_physical_relation = std::make_shared<physical::read_relation>(
        std::move(subqueries_physical),
        logical_read_relation->column_names(),
        logical_read_relation->table_name(),
        partial_filter_ptr ? partial_filter_ptr->clone() : nullptr);
      break;
    }
    case logical::relation::relation_type::write: {
      assert(children_physical.size() == 1);
      auto& input_physical = children_physical[0];

      auto const logical_insert_relation =
        dynamic_cast<logical::write_relation const*>(logical_relation);

      out_physical_relation =
        std::make_shared<physical::write_relation>(std::move(input_physical),
                                                   logical_insert_relation->column_names(),
                                                   logical_insert_relation->table_name());
      break;
    }
    case logical::relation::relation_type::join: {
      assert(children_physical.size() == 2);
      auto const logical_join_relation =
        dynamic_cast<logical::join_relation const*>(logical_relation);

      // Currently GQE supports the following join types: inner, left, left_semi, left_anti, full,
      // single. There are three categories among these join types.
      // Cannot use broadcast join: full, single
      // Can only broadcast the right table: left, left_semi, left_anti
      // Free to broadcast either left or right table: inner
      //
      // So, we default the broadcast policy to broadcast_policy::right and only consider
      // broadcast_policy::left if it's an inner join.
      physical::broadcast_policy policy = physical::broadcast_policy::right;

      // If the join type is inner join, we broadcast the smaller table.
      if (logical_join_relation->join_type() == join_type_type::inner) {
        auto const left_num_rows  = _estimator(children_logical[0]).num_rows;
        auto const right_num_rows = _estimator(children_logical[1]).num_rows;
        if (left_num_rows < right_num_rows) policy = physical::broadcast_policy::left;
      }

      out_physical_relation = std::make_shared<physical::broadcast_join_relation>(
        std::move(children_physical[0]),
        std::move(children_physical[1]),
        std::move(subqueries_physical),
        logical_join_relation->join_type(),
        logical_join_relation->condition()->clone(),
        logical_join_relation->projection_indices(),
        policy);
      break;
    }
    case logical::relation::relation_type::project: {
      assert(children_physical.size() == 1);
      auto const logical_project_relation =
        dynamic_cast<logical::project_relation const*>(logical_relation);

      std::vector<std::unique_ptr<expression>> exprs;
      for (auto const& expr : logical_project_relation->output_expressions_unsafe())
        exprs.push_back(expr->clone());

      out_physical_relation = std::make_shared<physical::project_relation>(
        std::move(children_physical[0]), std::move(subqueries_physical), std::move(exprs));
      break;
    }
    case logical::relation::relation_type::fetch: {
      assert(children_physical.size() == 1);
      auto const logical_fetch_relation =
        dynamic_cast<logical::fetch_relation const*>(logical_relation);

      out_physical_relation =
        std::make_shared<physical::fetch_relation>(std::move(children_physical[0]),
                                                   logical_fetch_relation->offset(),
                                                   logical_fetch_relation->count());
      break;
    }
    case logical::relation::relation_type::filter: {
      assert(children_physical.size() == 1);
      auto const logical_filter_relation =
        dynamic_cast<logical::filter_relation const*>(logical_relation);

      out_physical_relation =
        std::make_shared<physical::filter_relation>(std::move(children_physical[0]),
                                                    std::move(subqueries_physical),
                                                    logical_filter_relation->condition()->clone());
      break;
    }
    case logical::relation::relation_type::sort: {
      assert(children_physical.size() == 1);
      auto const logical_sort_relation =
        dynamic_cast<logical::sort_relation const*>(logical_relation);

      auto const in_keys = logical_sort_relation->expressions_unsafe();
      std::vector<std::unique_ptr<expression>> keys;
      keys.reserve(in_keys.size());
      for (auto const& key : in_keys)
        keys.push_back(key->clone());

      out_physical_relation = std::make_shared<physical::concatenate_sort_relation>(
        std::move(children_physical[0]),
        std::move(subqueries_physical),
        std::move(keys),
        logical_sort_relation->column_orders(),
        logical_sort_relation->null_orders());
      break;
    }
    case logical::relation::relation_type::aggregate: {
      assert(children_physical.size() == 1);
      auto const logical_aggregate_relation =
        dynamic_cast<logical::aggregate_relation const*>(logical_relation);

      auto const in_keys = logical_aggregate_relation->keys_unsafe();
      std::vector<std::unique_ptr<expression>> keys;
      keys.reserve(in_keys.size());
      for (auto const& key : in_keys)
        keys.push_back(key->clone());

      auto const in_values = logical_aggregate_relation->measures_unsafe();
      std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values;
      values.reserve(in_values.size());
      for (auto const& [kind, expr] : in_values)
        values.emplace_back(kind, expr->clone());

      out_physical_relation =
        std::make_shared<physical::concatenate_aggregate_relation>(std::move(children_physical[0]),
                                                                   std::move(subqueries_physical),
                                                                   std::move(keys),
                                                                   std::move(values));
      break;
    }
    case logical::relation::relation_type::window: {
      auto const logical_window_relation =
        dynamic_cast<logical::window_relation const*>(logical_relation);
      // if there is only a partition by and no order by, this should be converted to a physical
      // aggregate and join relation
      auto aggr_func          = logical_window_relation->aggr_func();
      auto arguments          = logical_window_relation->arguments_unsafe();
      auto partition_by       = logical_window_relation->partition_by_unsafe();
      auto order_by           = logical_window_relation->order_by_unsafe();
      auto order_dirs         = logical_window_relation->order_dirs();
      auto window_lower_bound = logical_window_relation->window_lower_bound();
      auto window_upper_bound = logical_window_relation->window_upper_bound();

      if ((std::holds_alternative<gqe::window_frame_bound::bounded>(window_lower_bound) ||
           std::holds_alternative<gqe::window_frame_bound::bounded>(window_upper_bound)) &&
          (order_by.size() == 0)) {
        throw std::runtime_error("Custom window frame only supported when order_by is present.");
      }

      // Convert to aggregate + join
      if (partition_by.size() > 0 && order_by.size() == 0) {
        std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values;
        for (auto const& argument : arguments)
          values.emplace_back(aggr_func, argument->clone());
        if (values.size() > 1) {
          throw std::runtime_error("Logical window relation can have at most one argument.");
        }

        std::vector<std::unique_ptr<expression>> aggregate_key_cols;
        for (auto const& partition_expr : partition_by) {
          aggregate_key_cols.push_back(partition_expr->clone());
        }

        auto const num_input_cols = children_logical[0]->num_columns();

        // Initialize join condition
        std::unique_ptr<expression> join_condition = std::make_unique<equal_expression>(
          partition_by[0]->clone(), std::make_shared<column_reference_expression>(num_input_cols));

        // If there's more than one join condition
        for (std::size_t partition_idx = 1; partition_idx < partition_by.size(); ++partition_idx) {
          join_condition = std::make_unique<logical_and_expression>(
            std::shared_ptr<expression>(std::move(join_condition)),
            std::make_shared<equal_expression>(
              partition_by[partition_idx]->clone(),
              std::make_shared<column_reference_expression>(num_input_cols + partition_idx)));
        }

        // In this case, no indices need to be dropped
        std::vector<cudf::size_type> join_indices(num_input_cols);
        std::iota(join_indices.begin(), join_indices.end(), 0);

        // include all the original columns plus the aggregated argument col that is appended
        // to the end of the key columns in the broadcast join
        join_indices.push_back(num_input_cols + partition_by.size());

        auto aggregate_relation =
          std::make_shared<physical::concatenate_aggregate_relation>(children_physical[0],
                                                                     std::move(subqueries_physical),
                                                                     std::move(aggregate_key_cols),
                                                                     std::move(values));

        out_physical_relation = std::make_shared<physical::broadcast_join_relation>(
          std::move(children_physical[0]),
          std::move(aggregate_relation),
          std::vector<std::shared_ptr<physical::relation>>(),
          join_type_type::inner,
          std::move(join_condition),
          std::move(join_indices),
          physical::broadcast_policy::right);
      }
      // Use physical window relation
      else if (order_by.size() > 0) {
        std::vector<std::unique_ptr<expression>> argument_cols;
        for (auto const& argument_expr : arguments) {
          argument_cols.push_back(argument_expr->clone());
        }
        std::vector<std::unique_ptr<expression>> partition_by_cols;
        for (auto const& partition_expr : partition_by) {
          partition_by_cols.push_back(partition_expr->clone());
        }
        std::vector<std::unique_ptr<expression>> order_by_cols;
        for (auto const& order_expr : order_by) {
          order_by_cols.push_back(order_expr->clone());
        }

        auto const num_input_cols = children_logical[0]->num_columns();

        auto input_with_row_id =
          std::make_shared<physical::gen_ident_col_relation>(children_physical[0]);

        // reference the primary key col we just added to the input
        std::vector<std::unique_ptr<expression>> ident_cols;
        ident_cols.emplace_back(std::make_unique<column_reference_expression>(num_input_cols));

        auto window_relation = std::make_shared<physical::window_relation>(
          input_with_row_id,
          std::vector<std::shared_ptr<physical::relation>>(),
          aggr_func,
          std::move(ident_cols),
          std::move(argument_cols),
          std::move(partition_by_cols),
          std::move(order_by_cols),
          std::move(order_dirs),
          window_lower_bound,
          window_upper_bound);

        // Initialize join condition
        std::unique_ptr<expression> join_condition = std::make_unique<equal_expression>(
          std::make_shared<column_reference_expression>(num_input_cols),
          std::make_shared<column_reference_expression>(num_input_cols + 1));

        // Drop the primary key col and append the window col
        std::vector<cudf::size_type> join_indices(num_input_cols);
        std::iota(join_indices.begin(), join_indices.end(), 0);
        join_indices.push_back(num_input_cols + 2);

        out_physical_relation = std::make_shared<physical::broadcast_join_relation>(
          std::move(input_with_row_id),
          std::move(window_relation),
          std::vector<std::shared_ptr<physical::relation>>(),
          join_type_type::inner,
          std::move(join_condition),
          std::move(join_indices),
          physical::broadcast_policy::right);
      } else {
        throw std::runtime_error(
          "GQE currently doesn't support frame-only window functions. Logical window relation must "
          "have partition-by or order-by clause.");
      }

      break;
    }
    case logical::relation::relation_type::set: {
      assert(children_physical.size() == 2);
      auto const logical_set_relation =
        dynamic_cast<logical::set_relation const*>(logical_relation);

      switch (logical_set_relation->set_operator()) {
        case logical::set_relation::set_union_all: {
          out_physical_relation = std::make_shared<physical::union_all_relation>(
            std::move(children_physical[0]), std::move(children_physical[1]));
          break;
        }
        default: throw std::logic_error("Unsupported set operator");
      }
      break;
    }
    case logical::relation::relation_type::user_defined: {
      if (subqueries_physical.size())
        throw std::logic_error("Subqueries are not supported in user-defined relations");

      auto const logical_user_defined_relation =
        dynamic_cast<logical::user_defined_relation const*>(logical_relation);

      out_physical_relation = std::make_shared<physical::user_defined_relation>(
        std::move(children_physical),
        logical_user_defined_relation->task_functor(),
        logical_user_defined_relation->last_child_break_pipeline());
      break;
    }
    default:
      throw std::logic_error("Cannot convert the logical relation to physical: " +
                             logical_relation->to_string());
  }

  _cache[logical_relation] = out_physical_relation;

  return out_physical_relation;
}

}  // namespace gqe
