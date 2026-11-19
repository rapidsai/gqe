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

#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/state.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Non-owning facade over a column-bearing QEP logical relation.
 *
 * A logical relation contains one or more logical data columns in schema order. Each column is an
 * owned `cudf_column`, a borrowed `cudf_column_view`, or an `indirect_column`. An optional leading
 * `valid_mask` or `valid_mask_view` applies to the complete relation and is not part of logical
 * column indexing. Count-only state is a separate shape.
 *
 * # Ownership and Lifetime
 *
 * The facade borrows its input `state_container`. The container and its `shared_state` slots must
 * outlive every facade access. Owned direct and indirect states retain their storage through shared
 * ownership in the input. Storage referenced by `cudf_column_view` or `valid_mask_view` remains
 * externally owned and must outlive every use.
 *
 * The facade performs host-side validation of non-null state slots, supported logical-column
 * kinds, equal logical row counts, and mask geometry. It enqueues no device work, allocates no
 * device memory, and does not turn indirect columns into cuDF columns.
 *
 * # Thread Safety
 *
 * Concurrent read-only access is safe while the borrowed container, state objects, and externally
 * owned borrowed storage remain unchanged. Shared ownership does not make externally owned storage
 * mutable or thread-safe.
 */
class logical_table_view {
 public:
  /**
   * @brief Recognize and validate a column-bearing logical relation.
   *
   * @param[in] container State container to inspect.
   *
   * @return A borrowing facade, or `std::nullopt` for empty, count-only, and non-relational shapes.
   *
   * @throws std::logic_error If a claimed relation has a null slot, a null owned column, no data
   *         column after a leading mask, an unsupported logical-column kind, or mismatched row
   *         geometry.
   */
  [[nodiscard]] static std::optional<logical_table_view> try_from(state_container_view container);

  /** @brief Return the logical data-column count, excluding the optional leading mask. */
  [[nodiscard]] cudf::size_type num_columns() const noexcept
  {
    return static_cast<cudf::size_type>(_columns.size());
  }

  /** @brief Return the common input row geometry without materializing a column. */
  [[nodiscard]] cudf::size_type row_count() const noexcept { return _row_count; }

  /** @brief Return all logical data-column states in schema order. */
  [[nodiscard]] state_container_view column_states() const noexcept { return _columns; }

  /**
   * @brief Return the backing shared state for one logical data column.
   *
   * @param[in] logical_column_index Logical index excluding the optional leading mask.
   *
   * @throws std::out_of_range If `logical_column_index` is outside `[0, num_columns())`.
   */
  [[nodiscard]] shared_state const& column_state(cudf::size_type logical_column_index) const;

  /**
   * @brief Return one logical column's cuDF type without materializing it.
   *
   * @param[in] logical_column_index Logical index excluding the optional leading mask.
   *
   * @throws std::out_of_range If `logical_column_index` is outside `[0, num_columns())`.
   */
  [[nodiscard]] cudf::data_type column_type(cudf::size_type logical_column_index) const;

  /** @brief Return the optional leading relation-level valid mask. */
  [[nodiscard]] std::optional<cudf::column_view> valid_mask() const noexcept { return _mask; }

  /**
   * @brief Return the exact source state for the leading valid mask.
   *
   * @return A pointer into the borrowed container, or `nullptr` when the relation is unmasked.
   */
  [[nodiscard]] shared_state const* valid_mask_state() const noexcept { return _mask_state; }

 private:
  logical_table_view(state_container_view columns,
                     cudf::size_type row_count,
                     std::optional<cudf::column_view> mask,
                     shared_state const* mask_state) noexcept;

  state_container_view _columns;
  cudf::size_type _row_count;
  std::optional<cudf::column_view> _mask;
  shared_state const* _mask_state;
};

}  // namespace qep
}  // namespace gqe
