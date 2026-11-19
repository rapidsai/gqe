/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/window.hpp>

#include <numeric>
#include <stdexcept>

namespace gqe {

std::shared_ptr<physical::relation> physical_plan_builder::plan_window(
  logical::window_relation const& node,
  std::vector<logical::relation*> const& children_logical,
  std::vector<std::shared_ptr<physical::relation>> children_physical,
  std::vector<std::shared_ptr<physical::relation>> subqueries_physical)
{
  // if there is only a partition by and no order by, this should be converted to a physical
  // aggregate and join relation
  auto aggr_func          = node.aggr_func();
  auto arguments          = node.arguments_unsafe();
  auto partition_by       = node.partition_by_unsafe();
  auto order_by           = node.order_by_unsafe();
  auto order_dirs         = node.order_dirs();
  auto window_lower_bound = node.window_lower_bound();
  auto window_upper_bound = node.window_upper_bound();

  if ((std::holds_alternative<gqe::window_frame_bound::bounded>(window_lower_bound) ||
       std::holds_alternative<gqe::window_frame_bound::bounded>(window_upper_bound)) &&
      (order_by.size() == 0)) {
    throw std::runtime_error("Custom window frame only supported when order_by is present.");
  }

  bool const use_like_shift_and       = _params && _params->filter_use_like_shift_and;
  bool const use_agg_perfect_hashing  = _params && _params->aggregation_use_perfect_hash;
  bool const use_join_perfect_hashing = _params && _params->join_use_perfect_hash;

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
                                                                 std::move(values),
                                                                 nullptr,
                                                                 use_agg_perfect_hashing,
                                                                 use_like_shift_and);

    // both hash map cache and mark join are disabled for this inner join case
    return std::make_shared<physical::broadcast_join_relation>(
      std::move(children_physical[0]),
      std::move(aggregate_relation),
      std::vector<std::shared_ptr<physical::relation>>(),
      join_type_type::inner,
      std::move(join_condition),
      std::move(join_indices),
      physical::broadcast_policy::right,
      gqe::unique_keys_policy::none,
      use_join_perfect_hashing,
      nullptr,
      nullptr,
      /*use_hash_map_cache=*/false,
      /*use_mark_join=*/false,
      use_like_shift_and);
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
      window_upper_bound,
      use_like_shift_and);

    // Initialize join condition
    std::unique_ptr<expression> join_condition = std::make_unique<equal_expression>(
      std::make_shared<column_reference_expression>(num_input_cols),
      std::make_shared<column_reference_expression>(num_input_cols + 1));

    // Drop the primary key col and append the window col
    std::vector<cudf::size_type> join_indices(num_input_cols);
    std::iota(join_indices.begin(), join_indices.end(), 0);
    join_indices.push_back(num_input_cols + 2);

    return std::make_shared<physical::broadcast_join_relation>(
      std::move(input_with_row_id),
      std::move(window_relation),
      std::vector<std::shared_ptr<physical::relation>>(),
      join_type_type::inner,
      std::move(join_condition),
      std::move(join_indices),
      physical::broadcast_policy::right,
      gqe::unique_keys_policy::none,
      use_join_perfect_hashing,
      nullptr,
      nullptr,
      false,
      false,
      use_like_shift_and);
  } else {
    throw std::runtime_error(
      "GQE currently doesn't support frame-only window functions. Logical window relation must "
      "have partition-by or order-by clause.");
  }
}

}  // namespace gqe
