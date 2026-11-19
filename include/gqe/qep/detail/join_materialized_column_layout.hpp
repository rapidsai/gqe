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

#include <cudf/types.hpp>

#include <span>
#include <vector>

namespace gqe::qep::detail {

/**
 * @brief Logical-to-materialized layout for one selectively materialized join input.
 *
 * Join columns, such as keys or predicate inputs, appear first in configured order. Additional
 * requested columns, such as projected payloads, follow in their supplied order. Repeated logical
 * indices reuse the first materialized column.
 *
 * The layout retains the original logical column count and a dense inverse mapping for checked
 * resolution after the source logical relation is no longer available. It does not interpret join
 * predicates or require equality comparisons.
 */
class join_materialized_column_layout {
 public:
  /**
   * @brief Build and validate one selectively materialized column layout.
   *
   * @param[in] logical_column_count Original logical input column count.
   * @param[in] key_indices Logical join-column indices, such as keys or predicate inputs.
   * @param[in] projected_payload_indices Additional side-local column indices, such as projected
   * payloads.
   *
   * @throws std::invalid_argument If the logical column count or any index is invalid.
   */
  join_materialized_column_layout(cudf::size_type logical_column_count,
                                  std::span<cudf::size_type const> key_indices,
                                  std::span<cudf::size_type const> projected_payload_indices);

  /** @brief Return the original logical input column count. */
  [[nodiscard]] cudf::size_type logical_column_count() const noexcept
  {
    return _logical_column_count;
  }

  /** @brief Return unique logical indices in deterministic materialization order. */
  [[nodiscard]] std::span<cudf::size_type const> logical_indices() const noexcept
  {
    return _logical_indices;
  }

  /** @brief Resolve one logical column to its materialized table index. */
  [[nodiscard]] cudf::size_type materialized_index(cudf::size_type logical_index) const;

  /** @brief Resolve logical indices to materialized table indices, preserving repetitions. */
  [[nodiscard]] std::vector<cudf::size_type> remap(
    std::span<cudf::size_type const> logical_indices) const;

 private:
  cudf::size_type _logical_column_count;
  std::vector<cudf::size_type> _logical_indices;
  std::vector<cudf::size_type> _logical_to_materialized;
};

}  // namespace gqe::qep::detail
