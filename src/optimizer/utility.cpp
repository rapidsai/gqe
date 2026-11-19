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

#include "utility.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/optimizer/relation_properties.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe::optimizer::utility {

namespace {

/** @brief Equality keys, null policy, and residual accumulated during join-condition traversal. */
struct partial_join_condition {
  std::vector<cudf::size_type> left_keys;
  std::vector<cudf::size_type> right_keys;
  std::optional<cudf::null_equality> null_equality;
  std::unique_ptr<gqe::expression> residual;
  bool is_valid{true};
};

[[nodiscard]] std::unique_ptr<gqe::expression> combine_residuals(
  std::unique_ptr<gqe::expression> left, std::unique_ptr<gqe::expression> right)
{
  if (left == nullptr) { return right; }
  if (right == nullptr) { return left; }
  return std::make_unique<gqe::logical_and_expression>(std::move(left), std::move(right));
}

[[nodiscard]] partial_join_condition analyze_subtree(gqe::expression const* condition,
                                                     cudf::size_type left_column_count)
{
  using expression_type = gqe::expression::expression_type;
  if (condition == nullptr) { return partial_join_condition{.is_valid = false}; }
  if (condition->type() != expression_type::binary_op) {
    return partial_join_condition{.residual = condition->clone()};
  }

  auto const* binary  = static_cast<gqe::binary_op_expression const*>(condition);
  auto const children = condition->children();
  if (children.size() != 2) { return partial_join_condition{.residual = condition->clone()}; }

  auto const op = binary->binary_operator();
  if (op == cudf::binary_operator::LOGICAL_AND || op == cudf::binary_operator::NULL_LOGICAL_AND) {
    auto left  = analyze_subtree(children[0], left_column_count);
    auto right = analyze_subtree(children[1], left_column_count);
    if (!left.is_valid || !right.is_valid ||
        (left.null_equality.has_value() && right.null_equality.has_value() &&
         left.null_equality != right.null_equality)) {
      return partial_join_condition{.is_valid = false};
    }
    left.left_keys.insert(left.left_keys.end(), right.left_keys.begin(), right.left_keys.end());
    left.right_keys.insert(left.right_keys.end(), right.right_keys.begin(), right.right_keys.end());
    if (!left.null_equality.has_value()) { left.null_equality = right.null_equality; }
    left.residual = combine_residuals(std::move(left.residual), std::move(right.residual));
    return left;
  }

  if ((op != cudf::binary_operator::EQUAL && op != cudf::binary_operator::NULL_EQUALS) ||
      children[0]->type() != expression_type::column_reference ||
      children[1]->type() != expression_type::column_reference) {
    return partial_join_condition{.residual = condition->clone()};
  }

  auto const first =
    static_cast<gqe::column_reference_expression const*>(children[0])->column_idx();
  auto const second =
    static_cast<gqe::column_reference_expression const*>(children[1])->column_idx();
  bool const first_is_left  = first >= 0 && first < left_column_count;
  bool const second_is_left = second >= 0 && second < left_column_count;
  if (first < 0 || second < 0 || first_is_left == second_is_left) {
    return partial_join_condition{.residual = condition->clone()};
  }

  return partial_join_condition{
    .left_keys     = {first_is_left ? first : second},
    .right_keys    = {(first_is_left ? second : first) - left_column_count},
    .null_equality = op == cudf::binary_operator::NULL_EQUALS ? cudf::null_equality::EQUAL
                                                              : cudf::null_equality::UNEQUAL};
}

}  // namespace

std::optional<equijoin_keys> try_extract_equijoin_keys(
  gqe::expression const* condition, cudf::size_type num_columns_left_child_relation)
{
  auto analysis = analyze_join_condition(condition, num_columns_left_child_relation);
  if (!analysis.has_value() || analysis->residual != nullptr || analysis->keys.left.empty()) {
    return std::nullopt;
  }
  return std::move(analysis->keys);
}

std::optional<join_condition_analysis> analyze_join_condition(
  gqe::expression const* condition, cudf::size_type num_columns_left_child_relation)
{
  GQE_EXPECTS(num_columns_left_child_relation >= 0,
              "analyze_join_condition: left column count must be non-negative",
              std::invalid_argument);
  auto result = analyze_subtree(condition, num_columns_left_child_relation);
  if (!result.is_valid) { return std::nullopt; }
  return join_condition_analysis{
    .keys     = {.left          = std::move(result.left_keys),
                 .right         = std::move(result.right_keys),
                 .null_equality = result.null_equality.value_or(cudf::null_equality::UNEQUAL)},
    .residual = std::move(result.residual)};
}

bool keys_cover_unique_key(relation_properties const& props,
                           std::vector<cudf::size_type> const& equi_keys)
{
  std::unordered_set<cudf::size_type> equi_set(equi_keys.begin(), equi_keys.end());
  for (auto const& key : props.unique_keys()) {
    if (std::all_of(key.begin(), key.end(), [&](auto c) { return equi_set.contains(c); }))
      return true;
  }
  return false;
}

}  // namespace gqe::optimizer::utility
