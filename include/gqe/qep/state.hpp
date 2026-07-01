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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Task-private state.
 *
 * This is a base class for task-private states. Non-private states should be
 * preferred whenever possible.
 *
 * # Design
 *
 * Task-private state is needed to avoid including many specialized headers in this file. An example
 * use-case is the `cudf::hash_join` class, which is only useful for a cuDF-based hash join task but
 * has no other purpose.
 *
 * In contrast, a cuco::static_multiset is a general-purpose data structure that can be used for
 * different tasks.
 */
class task_private_state {
 public:
  virtual ~task_private_state() = 0;
};

/**
 * @brief Query Execution Plan (QEP) state.
 *
 * An explicit list of states that can be used to assemble a QEP state.
 *
 * # Design
 *
 * The state kinds are explicit to facilitate a modular task design. The intention is that
 * states can be reused across different tasks.
 */
namespace state_kind {
/**
 * @brief Owned cuDF column.
 */
struct cudf_column {
  std::unique_ptr<cudf::column> column;
};

/**
 * @brief Borrowed cuDF column.
 */
struct cudf_column_view {
  cudf::column_view column;
};

/**
 * @brief Table row count.
 *
 * Carries the row count for containers that have no columns to derive it from (e.g. the
 * count-only chunk emitted for `SELECT COUNT(*) FROM t`). For any container that has at least
 * one column, the row count is the column's `size()` and no `row_count` slot is needed.
 */
struct row_count {
  cudf::size_type value;
};

/**
 * @brief Owned Boolean valid-row mask column.
 *
 * Tagged variant of an owned `cudf::column` that distinguishes a Boolean valid-row mask
 * from regular table columns at the structural level. Used by the masked-table
 * convention `[valid_mask, columns...]`. The mask column must have
 * `cudf::type_id::BOOL8`; rows where the value is `true` are valid (they survive a
 * subsequent gather).
 *
 * # Limitations
 *
 * Masks are currently owned only. A `valid_mask_view` kind can be added later when a
 * pass-through use case appears.
 */
struct valid_mask {
  std::unique_ptr<cudf::column> column;
};

/**
 * @brief Task-private state.
 */
struct task_private {
  std::unique_ptr<task_private_state> data;
};

/**
 * @brief State kind.
 */
using type = std::variant<cudf_column, cudf_column_view, row_count, valid_mask, task_private>;
}  // namespace state_kind

/**
 * @brief Query Execution Plan (QEP) shared state.
 *
 * A convenience wrapper for a shared ownership of a state using `std::shared_ptr`.
 */
using shared_state = std::shared_ptr<state_kind::type>;

template <typename... Args>
[[nodiscard]] shared_state make_shared_state(Args&&... args)
{
  return std::make_shared<state_kind::type>(std::forward<Args>(args)...);
}

/**
 * @brief A container for Query Execution Plan (QEP) states.
 *
 * Collects the states of a QEP task into a common container type. This can be an intermediate
 * state between two tasks, or the final query result.
 *
 * # Lifetime
 *
 * The `state_container` is a temporary intermediate value that only lives in between tasks.
 * However, its constituent states can outlive the container, in that a shared pointer copy is
 * obtained by a different `state_container`. This allows the container to be disassembled, while
 * some of its constituent states are propagated in-situ and the unused states are destroyed. The
 * shared pointer is used to track the lifetime of its constituent states, such that multiple
 * tasks can share ownership. Each state is destroyed as soon as it is no longer used.
 *
 * # Thread Safety
 *
 * The `state_container` "shape" is immutable after construction. This ensures thread-safety when
 * accessing states.
 *
 * *QEP tasks are responsible for ensuring thread-safety when accessing states.*
 *
 * # Design
 *
 * The "shape" of the container, i.e., the order and types of states it contains, is defined at
 * runtime. Consider this a form of duck-typing. In future, when common shapes between tasks have
 * been identified, this design decision may be revised to use named types expressed by C++
 * structs.
 *
 * Shape helpers recognize and transform, as well as build common shapes. For example,
 * `to_table_view` recognizes cuDF tables with mixed owned/borrowed columns, and
 * `masked_table_view::try_from` recognizes when the table additionally has a row mask.
 *
 * ## cuDF Tables
 *
 * Conversion to and from cuDF tables is supported (provided that all states are
 * convertible to `cudf::column` or `cudf::column_view`).
 *
 * ## Row count
 *
 * The row count is the geometry of the table — read host-side from `cudf::column::size()` on
 * any column-bearing slot. A dedicated `row_count` slot only appears in column-less containers
 * (e.g. the chunk emitted for `SELECT COUNT(*) FROM t`). `get_row_count` reads from whichever
 * is available without GPU sync.
 */
using state_container = std::vector<shared_state>;

/**
 * @brief A state container view.
 *
 * A convenience wrapper around `std::span` to reference a `state_container` without taking
 * ownership. Inner `shared_ptr`s cannot be rebound, but their pointees remain mutable.
 */
using state_container_view = std::span<shared_state const>;

/**
 * @brief Try to convert the state container to a `cudf::table_view` at zero cost.
 *
 * Succeeds when every slot is a `cudf_column` or `cudf_column_view`, optionally preceded by
 * a single `row_count` slot at index 0 (skipped — count-only containers convert to a
 * zero-column table view).
 *
 * Returns `std::nullopt` for empty containers, shapes that would require additional work
 * to materialise, or shapes that have no table-view projection.
 *
 * @param[in] container The state container to inspect.
 *
 * @return The table view, or `std::nullopt` if `container` is not zero-cost convertible.
 */
[[nodiscard]] std::optional<cudf::table_view> to_table_view(state_container_view container);

/**
 * @brief Read the row count from a state container without GPU sync.
 *
 * Reads from whichever host-side source is available at index 0:
 *
 *  - `cudf_column` / `cudf_column_view` → `column.size()`.
 *  - `valid_mask`                       → `column->size()` (the mask is itself a column).
 *  - `row_count`                        → the slot's value (column-less containers only).
 *
 * Every branch is a host-side read; no sync is performed.
 *
 * @param[in] container The state container to inspect.
 *
 * @throws std::logic_error If the container is empty or the state at index 0 carries no row
 *         count.
 *
 * @return The row count.
 */
[[nodiscard]] cudf::size_type get_row_count(state_container_view container);

/**
 * @brief Build a regular-table-shaped state container with empty columns.
 *
 * Returns `[cudf_column (empty, type=data_types[0]), ...]`. Useful for operators that need to
 * emit a schema-correct empty result (e.g., when a fold task receives no input chunks). The
 * row count is implicit at zero via the empty columns.
 *
 * @param[in] data_types Output column types in emission order. Must be non-empty.
 *
 * @return A state container shaped as a regular table with empty columns of the requested
 *         types.
 */
[[nodiscard]] state_container make_empty_state_container(
  std::span<cudf::data_type const> data_types);

/**
 * @brief Shallow-copy a state container view into an owned `state_container`.
 *
 * Each element is copy-constructed (refcount bump on the inner shared_ptr); no underlying
 * data is deep-copied.
 *
 * @param[in] src The state container view to shallow-copy.
 *
 * @return A new state container that shares ownership of the same inner state.
 */
[[nodiscard]] state_container make_mutable_state_copy(state_container_view src);

/**
 * @brief State container builder.
 *
 * A `state_container_builder` is used to construct an immutable `state_container`.
 *
 * # Thread Safety
 *
 * The state container builder is not thread-safe.
 */
class state_container_builder {
 public:
  state_container_builder();
  ~state_container_builder()                                             = default;
  state_container_builder(const state_container_builder&)                = delete;
  state_container_builder& operator=(const state_container_builder&)     = delete;
  state_container_builder(state_container_builder&&) noexcept            = default;
  state_container_builder& operator=(state_container_builder&&) noexcept = default;

  /**
   * @brief Add a state to the container.
   *
   * @param[in] state The state to add.
   * @return A reference to the builder.
   */
  state_container_builder& add_state(state_kind::type&& state);
  state_container_builder& add_state(shared_state state);

  /**
   * @brief Add a cuDF table.
   *
   * The table is disassembled into its constituent columns, which are then added as states.
   * The row count is implicit in the columns' `size()`.
   *
   * @param[in] table The table to add.
   * @return A reference to the builder.
   */
  state_container_builder& add_state(cudf::table&& table);

  /**
   * @brief Add a cuDF table view.
   *
   * The table view is disassembled into its constituent columns, which are then added as
   * states. The row count is implicit in the columns' `size()`.
   *
   * @param[in] table_view The table view to add.
   * @return A reference to the builder.
   */
  state_container_builder& add_state(cudf::table_view table_view);

  /**
   * @brief Build the state.
   *
   * @return A state container holding the added states as shared pointers.
   */
  state_container build();

 private:
  std::vector<shared_state>
    _states;  ///< The added states. Semantically, this is not yet a `state_container`.
};

}  // namespace qep
}  // namespace gqe
