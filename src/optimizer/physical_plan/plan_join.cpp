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
#include <gqe/logical/join.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/join.hpp>

#include <cassert>

namespace gqe {

namespace {

/**
 * @brief Whether the join condition is composed only of equality leaves
 *        (`EQUAL` / `NULL_EQUALS`) connected by AND nodes.
 *
 * Assumes the input is a canonicalized boolean expression: NOTs pushed down
 * (e.g. `!(a || b)` rewritten to `!a && !b`) and ORs lifted out. Under
 * that canonical form, the only "equality-only" shape is an AND-tree whose
 * leaves are `EQUAL` or `NULL_EQUALS`.
 */
bool is_equality_only_condition(expression const* condition)
{
  // substrait_parser guarantees join condition is non-null (cross-join uses a
  // literal `true` condition); the assert documents that invariant.
  assert(condition != nullptr);
  if (condition->type() != expression::expression_type::binary_op) return false;
  auto const* bin        = static_cast<binary_op_expression const*>(condition);
  auto const child_exprs = condition->children();
  assert(child_exprs.size() == 2);
  switch (bin->binary_operator()) {
    case cudf::binary_operator::EQUAL:
    case cudf::binary_operator::NULL_EQUALS: return true;
    case cudf::binary_operator::LOGICAL_AND:
    case cudf::binary_operator::NULL_LOGICAL_AND:
      return is_equality_only_condition(child_exprs[0]) &&
             is_equality_only_condition(child_exprs[1]);
    default: return false;
  }
}

}  // namespace

std::shared_ptr<physical::relation> physical_plan_builder::plan_join(
  logical::join_relation const& node,
  std::vector<logical::relation*> const& children_logical,
  std::vector<std::shared_ptr<physical::relation>> children_physical,
  std::vector<std::shared_ptr<physical::relation>> subqueries_physical)
{
  assert(children_physical.size() == 2);

  // Currently GQE supports the following join types: inner, left, left_semi, left_anti, full,
  // single. There are three categories among these join types.
  // Cannot use broadcast join: full, single
  // Can only broadcast the right table: left
  // Free to broadcast either left or right table: inner, left_semi, left_anti
  //
  // For left_semi and left_anti based on the broadcasted side the materialization
  // differs with performance implications, hence, we need more benchmarking to decide the
  // logic of the side to be broadcasted
  // - Mark Join for left_{semi,anti} join is only enabled on left-side broadcast.
  //
  // So, we default the broadcast policy to broadcast_policy::right and only consider
  // broadcast_policy::left if it's an {inner,left_semi,left_anti} join.
  physical::broadcast_policy policy = physical::broadcast_policy::right;

  // If the join type is inner/left_{semi,anti} join, we broadcast the smaller table.
  if (node.join_type() == join_type_type::inner || node.join_type() == join_type_type::left_semi ||
      node.join_type() == join_type_type::left_anti) {
    auto const left_num_rows  = _estimator(children_logical[0]).num_rows;
    auto const right_num_rows = _estimator(children_logical[1]).num_rows;
    if (left_num_rows < right_num_rows) policy = physical::broadcast_policy::left;
  }

  // If possible, enable unique keys join when building on broadcast side.
  gqe::unique_keys_policy unique_keys_pol = gqe::unique_keys_policy::none;
  if (policy == physical::broadcast_policy::right) {
    if (node.unique_keys_policy() == gqe::unique_keys_policy::right ||
        node.unique_keys_policy() == gqe::unique_keys_policy::either)
      unique_keys_pol = gqe::unique_keys_policy::right;
  } else {
    if (node.unique_keys_policy() == gqe::unique_keys_policy::left ||
        node.unique_keys_policy() == gqe::unique_keys_policy::either)
      unique_keys_pol = gqe::unique_keys_policy::left;
  }
  // Build side for unique keys join should have been determined by this point
  assert(unique_keys_pol != gqe::unique_keys_policy::either);

  // Apply global parameter gates (when params are provided via the node_manager path).

  // join_use_unique_keys is true by default, unless overridden by the gqe-cli
  if (_params && !_params->join_use_unique_keys) unique_keys_pol = gqe::unique_keys_policy::none;

  // Mark join: only valid for broadcast-left semi/anti.  Numeric-key precondition is also
  // required but can only be checked at execution time, so the executor still asserts it.
  bool const use_mark_join = (_params && _params->join_use_mark_join) &&
                             policy == physical::broadcast_policy::left &&
                             (node.join_type() == join_type_type::left_semi ||
                              node.join_type() == join_type_type::left_anti);

  /**
   * Hash-map cache is enabled when:
   * - mark-join is in use, OR
   * - the user opt-in is set AND the join condition is equality-only.
   */
  bool const is_equality_only = is_equality_only_condition(node.condition());
  bool const use_hash_map_cache =
    use_mark_join || ((_params && _params->join_use_hash_map_cache) && is_equality_only);

  bool const use_like_shift_and  = _params && _params->filter_use_like_shift_and;
  bool const use_perfect_hashing = _params && _params->join_use_perfect_hash;

  // we construct the join with null left/right filter conditions. If left/right filter conditions
  // are provided, then use_hash_map_cache cannot be used when it's not unique-key-join, perfect
  // join, or mark join.
  return std::make_shared<physical::broadcast_join_relation>(std::move(children_physical[0]),
                                                             std::move(children_physical[1]),
                                                             std::move(subqueries_physical),
                                                             node.join_type(),
                                                             node.condition()->clone(),
                                                             node.projection_indices(),
                                                             policy,
                                                             unique_keys_pol,
                                                             use_perfect_hashing,
                                                             /*left_filter_condition=*/nullptr,
                                                             /*right_filter_condition=*/nullptr,
                                                             use_hash_map_cache,
                                                             use_mark_join,
                                                             use_like_shift_and);
}

}  // namespace gqe
