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

#pragma once

#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <span>
#include <vector>

namespace gqe {

/**
 * @brief Collect unique column references from expressions in first-encounter order.
 *
 * The traversal visits each expression tree depth-first and preserves the order in which logical
 * column indices first appear. Repeated references are returned once. The expressions and their
 * children are borrowed for the duration of this O(E * R) host traversal, where E is the number of
 * expression nodes and R is the number of unique references.
 *
 * @param[in] expressions Non-null root expressions to traverse.
 * @return Unique logical column indices in first-encounter order.
 *
 * @throws std::invalid_argument If a root or child expression is null.
 */
[[nodiscard]] std::vector<cudf::size_type> collect_column_references(
  std::span<expression const* const> expressions);

/**
 * @brief Validate logical column references against a relation width.
 *
 * @param[in] column_indices Logical data-column indices.
 * @param[in] logical_column_count Number of logical data columns in the relation.
 *
 * @throws std::out_of_range If the relation width is negative or an index is outside
 *         `[0, logical_column_count)`.
 */
void validate_column_references(std::span<cudf::size_type const> column_indices,
                                cudf::size_type logical_column_count);

}  // namespace gqe
