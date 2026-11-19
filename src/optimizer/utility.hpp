/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {
class expression;

namespace optimizer {

class relation_properties;

namespace utility {

/** @brief Equality keys in input-local coordinates together with their shared null policy. */
struct equijoin_keys {
  std::vector<cudf::size_type> left;  ///< Indices in the logical left input.

  std::vector<cudf::size_type> right;  ///< Indices in the logical right input.

  cudf::null_equality null_equality;  ///< Null policy shared by every key comparison.
};

/** @brief Extracted equality keys and the remaining top-level conjunction. */
struct join_condition_analysis {
  equijoin_keys keys;  ///< Cross-input equality keys in logical input coordinates.

  std::unique_ptr<gqe::expression> residual;  ///< Remaining predicate, or null for pure equality.
};

/**
 * @brief This function determine whether each projection index is from the left or right child.
 * Then offset it accordingly assuming a left/right swap.
 *
 * @param projection_indices List of projection indices to offset
 * @param n_cols_left Number of columns from left child
 * @param n_cols_right Number of columns from right child
 */
inline void swap_projection_indices_inplace(std::vector<cudf::size_type>& projection_indices,
                                            cudf::size_type n_cols_left,
                                            cudf::size_type n_cols_right)
{
  for (auto& projection_index : projection_indices) {
    assert(projection_index < (n_cols_left + n_cols_right));
    projection_index = projection_index < n_cols_left ? projection_index + n_cols_right
                                                      : projection_index - n_cols_left;
  }
}

/**
 * @brief Try to extract cross-input equality keys and one consistent null policy.
 *
 * The condition may contain `EQUAL` or `NULL_EQUALS` leaves joined by `LOGICAL_AND` or
 * `NULL_LOGICAL_AND`. Every comparison must reference one column from each input, and every leaf
 * must use the same null policy. Each recursive call returns a complete value, so a rejected
 * subtree cannot leave partially populated key vectors.
 *
 * @param[in] condition Join predicate to inspect.
 * @param[in] num_columns_left_child_relation Output column count of the left child, used to split
 * global column indices into per-side indices.
 *
 * @return Extracted keys, or `std::nullopt` if the expression is not a supported conjunction.
 */
[[nodiscard]] std::optional<equijoin_keys> try_extract_equijoin_keys(
  gqe::expression const* condition, cudf::size_type num_columns_left_child_relation);

/**
 * @brief Extract equality leaves from a top-level conjunction and clone its residual predicate.
 *
 * Only `EQUAL` and `NULL_EQUALS` leaves between bare columns on opposite logical inputs become
 * hash keys. Traversal continues through `LOGICAL_AND` and `NULL_LOGICAL_AND`, combining residuals
 * with `logical_and_expression`. Other expression shapes remain intact in the residual. All
 * extracted equality leaves must share one null policy.
 *
 * @param[in] condition Join predicate to analyze.
 * @param[in] num_columns_left_child_relation Logical-left input column count.
 *
 * @return Value-oriented analysis, or `std::nullopt` for mixed extracted null policies.
 */
[[nodiscard]] std::optional<join_condition_analysis> analyze_join_condition(
  gqe::expression const* condition, cudf::size_type num_columns_left_child_relation);

/**
 * @brief Return true if any unique key-set registered on @p props is fully covered by @p equi_keys.
 *
 * A registered key-set is covered iff every one of its columns appears in the equi-join keys, i.e.
 * the join keys are at least as restrictive as that unique key. A singleton key {col} is covered
 * iff col is among the equi-join keys.
 *
 * @param props     Relation properties carrying the registered unique key-sets.
 * @param equi_keys Per-side equi-join key column indices.
 */
bool keys_cover_unique_key(relation_properties const& props,
                           std::vector<cudf::size_type> const& equi_keys);

}  // namespace utility
}  // namespace optimizer
}  // namespace gqe
