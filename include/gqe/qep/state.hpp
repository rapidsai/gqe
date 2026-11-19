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

#include <gqe/query_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

class deferred_column_state;
class indirect_column_state;
class keyed_accumulator_state;
class source_reference;

/** @brief Shared handle to a keyed-accumulator device hash map. */
using keyed_accumulator_handle = std::shared_ptr<keyed_accumulator_state>;

/**
 * @brief Immutable owned position-list (gather-map) column.
 *
 * The column contains `cudf::size_type` source-row indices and has no nulls. Repeated and
 * non-monotonic positions are valid.
 */
class position_list_state {
 public:
  /**
   * @brief Construct a position list with complete owned-buffer deallocation metadata.
   *
   * @param[in] column Position-list column whose data, null mask, and recursive child storage are
   *                   covered by `deallocation_streams`.
   * @param[in] deallocation_streams Deallocation streams retained by the buffers owned recursively
   *                                 by `column`.
   *
   * @pre `deallocation_streams` contains the deallocation stream retained by every device buffer
   *      owned recursively by `column`.
   */
  position_list_state(std::unique_ptr<cudf::column> column,
                      std::span<query_stream const> deallocation_streams);

  [[nodiscard]] cudf::column const& column() const noexcept { return *_column; }

  [[nodiscard]] std::span<query_stream const> deallocation_streams() const noexcept
  {
    return _deallocation_streams;
  }

 private:
  std::vector<query_stream> _deallocation_streams;
  std::unique_ptr<cudf::column> _column;
};

/** @brief Shared handle to an immutable position list. */
using position_list_handle = std::shared_ptr<position_list_state const>;

/**
 * @brief Build a validated position list whose owned buffers deallocate on one deallocation stream.
 *
 * @param[in] positions Position-list column to transfer into the typed state.
 * @param[in] deallocation_stream Deallocation stream covering every buffer owned recursively by
 *                                `positions`.
 *
 * @pre Every device buffer owned recursively by `positions` retains `deallocation_stream` for
 *      deallocation.
 */
[[nodiscard]] position_list_handle make_position_list(std::unique_ptr<cudf::column> positions,
                                                      query_stream const& deallocation_stream);

/**
 * @brief Build a validated position list with complete owned-buffer deallocation metadata.
 *
 * @param[in] positions Position-list column to transfer into the typed state.
 * @param[in] deallocation_streams Deallocation streams collectively covering the data, null mask,
 *                                 and recursive child buffers owned by `positions`.
 *
 * @pre `deallocation_streams` contains the deallocation stream retained by every device buffer
 *      owned recursively by `positions`.
 */
[[nodiscard]] position_list_handle make_position_list(
  std::unique_ptr<cudf::column> positions, std::span<query_stream const> deallocation_streams);

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
 *
 * # Stream Lifetime
 *
 * A task-private payload may retain stream-dependent objects without recording deallocation-stream
 * metadata in `shared_state`. Such a payload relies on the executor destroying all QEP-owned state
 * before its `query_context`. A payload that is allowed to escape the QEP task graph must instead
 * retain every `query_stream` required by its objects. Stream ownership does not replace the task's
 * responsibility to order destruction after all in-flight device work.
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
  /**
   * @brief Construct a row-count state.
   *
   * @throws std::invalid_argument If `value` is negative.
   */
  explicit row_count(cudf::size_type value);

  cudf::size_type value;
};

/**
 * @brief Statistics produced by a terminal table write.
 *
 * This state is distinct from `row_count`: it carries the storage-specific `table_statistics`
 * produced by a write and consumed by the task-manager boundary. Fields that a storage writer
 * cannot report remain zero.
 */
struct write_statistics {
  gqe::table_statistics value;
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
 */
class valid_mask {
 public:
  explicit valid_mask(std::unique_ptr<cudf::column> column);

  [[nodiscard]] cudf::column const& column() const noexcept { return *_column; }

 private:
  std::unique_ptr<cudf::column> _column;
};

/**
 * @brief Borrowed Boolean valid-row mask column.
 *
 * The view owns no device storage. Its source must outlive every use of the view, following the
 * same borrowed-memory lifetime rule as `cudf_column_view`.
 */
class valid_mask_view {
 public:
  /**
   * @brief Construct a borrowed valid-row mask.
   *
   * @throws std::invalid_argument If `column` does not have `cudf::type_id::BOOL8`.
   */
  explicit valid_mask_view(cudf::column_view column);

  [[nodiscard]] cudf::column_view const& column() const noexcept { return _column; }

 private:
  cudf::column_view _column;
};

/**
 * @brief Generic state-container wrapper for a typed position list.
 *
 * Internal objects that require a position list should retain `position_list_handle` directly.
 * This wrapper is only for heterogeneous `state_container` boundaries.
 */
class position_list {
 public:
  explicit position_list(position_list_handle state);

  [[nodiscard]] position_list_handle const& state() const noexcept { return _state; }

 private:
  position_list_handle _state;
};

/**
 * @brief Immutable indirect logical-column state.
 *
 * Holds a shared descriptor that identifies one column of a gather source.
 */
class indirect_column {
 public:
  explicit indirect_column(std::shared_ptr<indirect_column_state const> state);

  [[nodiscard]] std::shared_ptr<indirect_column_state const> const& state() const noexcept
  {
    return _state;
  }

 private:
  std::shared_ptr<indirect_column_state const> _state;
};

/**
 * @brief Portable reference to one fetchable unit of source work.
 *
 * Holds a shared immutable `source_reference` describing a memory-bounded unit of a
 * data source (for example one in-memory row group or one Parquet row-group span). References are
 * host-only and carry no device data. They are produced by reference-generating iterate tasks and
 * consumed by resolve tasks; no other operator handles this kind.
 */
class reference {
 public:
  /**
   * @brief Construct a reference state.
   *
   * @throws std::invalid_argument If `state` is null.
   */
  explicit reference(std::shared_ptr<source_reference const> state);

  [[nodiscard]] std::shared_ptr<source_reference const> const& state() const noexcept
  {
    return _state;
  }

 private:
  std::shared_ptr<source_reference const> _state;
};

/**
 * @brief Handle to one logical column whose materialization is deferred.
 *
 * Holds a shared `deferred_column_state` that identifies one column of a shared deferred source.
 * Columns of the same source materialize in one batched decode; see
 * `gqe/qep/shapes/deferred_column.hpp` for the consumption contract.
 */
class deferred_column {
 public:
  /**
   * @brief Construct a deferred-column state.
   *
   * @throws std::invalid_argument If `state` is null.
   */
  explicit deferred_column(std::shared_ptr<deferred_column_state const> state);

  [[nodiscard]] std::shared_ptr<deferred_column_state const> const& state() const noexcept
  {
    return _state;
  }

 private:
  std::shared_ptr<deferred_column_state const> _state;
};

/**
 * @brief Task-private state.
 */
struct task_private {
  std::unique_ptr<task_private_state> data;
};

/**
 * @brief Shared keyed-accumulator device hash map.
 *
 * Wraps a shared handle to a `keyed_accumulator_state` (see
 * `gqe/qep/shapes/keyed_accumulator.hpp`), the pipeline-breaker state between the task that builds
 * the map and its successors (the scan task, and a future groupjoin probe task). The kind is
 * first-class rather than `task_private` because multiple successor task kinds recognize it without
 * a downcast.
 *
 * Unlike `position_list` and `indirect_column`, the handle's pointee is non-const: tasks mutate
 * the device storage through atomics. Tasks are responsible for thread-safety, per the
 * container contract.
 *
 * The map never travels as a pipelined chunk. Generic shape helpers reject it: it has no
 * column-view projection and no host-readable row count.
 */
class keyed_accumulator {
 public:
  /**
   * @brief Wrap a keyed-accumulator handle for a state-container boundary.
   *
   * @throws std::logic_error If `state` is null.
   */
  explicit keyed_accumulator(keyed_accumulator_handle state);

  [[nodiscard]] keyed_accumulator_handle const& state() const noexcept { return _state; }

 private:
  keyed_accumulator_handle _state;
};

/**
 * @brief State kind.
 */
using type = std::variant<cudf_column,
                          cudf_column_view,
                          row_count,
                          write_statistics,
                          valid_mask,
                          valid_mask_view,
                          position_list,
                          indirect_column,
                          reference,
                          deferred_column,
                          task_private,
                          keyed_accumulator>;
}  // namespace state_kind

namespace detail {
class shared_state_factory;
}

/**
 * @brief Shared immutable QEP state and its device-storage deallocation streams.
 *
 * The visible `state_kind::type` is const after construction. This constness is shallow for state
 * kinds whose representation contains a mutable pointee. In particular, callers must not mutate
 * or release the `cudf::column` owned by a `cudf_column` state. State kinds may retain explicitly
 * mutable pointees when mutation is part of their contract, such as task-private state and keyed
 * accumulators.
 *
 * ## Stream Ownership
 *
 * A deallocation stream is represented by the `query_stream` capability for the CUDA stream that
 * an owning device buffer retains and later supplies to its memory resource when deallocated. It
 * is not necessarily the stream that most recently used the buffer. Construction records this
 * metadata; it does not rebind the buffers to the supplied streams. For column-bearing state kinds,
 * the metadata must cover every recursively owned device buffer and remain unchanged for the
 * lifetime of the state. Retaining the capabilities keeps explicit streams alive until all buffers
 * that use them have been deallocated.
 *
 * `cudaStreamPerThread` is a thread-relative special stream handle rather than a stable stream
 * identity: its raw value resolves to the per-thread default stream of the host thread that
 * performs deallocation. See `order_deallocation_after_stream_use` for the resulting cross-thread
 * release restriction.
 *
 * Borrowed `cudf_column_view` and `valid_mask_view` state kinds contribute no deallocation streams
 * of their own. A directly wrapped borrowed view therefore has no deallocation streams and relies
 * on its external owner. An internally anchored view retains its source and reports the union of
 * the source and visible state's deallocation streams.
 * `task_private` and `keyed_accumulator` are opaque to this metadata mechanism. They may rely on
 * query-scoped stream ownership while they remain QEP-owned, but must retain their required
 * capabilities if they escape the task graph. Their task or implementation remains responsible for
 * ordering destruction of any device storage after its final use.
 */
class shared_state {
 public:
  shared_state() noexcept = default;
  shared_state(std::nullptr_t) noexcept {}

  shared_state(shared_state const&)     = default;
  shared_state(shared_state&&) noexcept = default;

  shared_state& operator=(shared_state other) noexcept
  {
    _state.swap(other._state);
    _deallocation_streams.swap(other._deallocation_streams);
    return *this;
  }

  [[nodiscard]] state_kind::type const* get() const noexcept { return _state.get(); }
  [[nodiscard]] state_kind::type const& operator*() const noexcept { return *_state; }
  [[nodiscard]] state_kind::type const* operator->() const noexcept { return _state.get(); }
  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_state); }

  void reset() noexcept
  {
    _state.reset();
    _deallocation_streams.clear();
  }

  [[nodiscard]] std::span<query_stream const> deallocation_streams() const noexcept
  {
    return _deallocation_streams;
  }

  friend bool operator==(shared_state const& lhs, shared_state const& rhs) noexcept
  {
    return lhs.get() == rhs.get();
  }
  friend bool operator==(shared_state const& state, std::nullptr_t) noexcept
  {
    return state.get() == nullptr;
  }
  friend bool operator==(std::nullptr_t, shared_state const& state) noexcept
  {
    return state == nullptr;
  }

 private:
  shared_state(std::shared_ptr<state_kind::type const> state,
               std::vector<query_stream> deallocation_streams)
    : _deallocation_streams{std::move(deallocation_streams)}, _state{std::move(state)}
  {
  }

  // Declared first so it is destroyed after the state and its device buffers.
  std::vector<query_stream> _deallocation_streams;
  std::shared_ptr<state_kind::type const> _state;

  friend class detail::shared_state_factory;
};

/**
 * @brief Create shared state without supplying deallocation-stream metadata.
 *
 * Use this overload for state kinds other than `cudf_column` and `valid_mask`. Position-list and
 * indirect-column states preserve the deallocation-stream metadata already stored in their typed
 * state. Directly owned columns and masks must use an overload that records their retained
 * deallocation stream or stream set.
 *
 * @throws std::invalid_argument If `state` directly owns column storage.
 */
[[nodiscard]] shared_state make_shared_state(state_kind::type state);

/**
 * @brief Create a shared state for a column or mask that deallocates on one deallocation stream.
 *
 * @param[in] state Owning column or mask to wrap. All recursive owned buffers must be covered by
 *                  `deallocation_stream`.
 * @param[in] deallocation_stream Deallocation stream covering the directly owned storage in
 *                                `state`.
 *
 * @pre Every directly owned device buffer in `state` retains `deallocation_stream` for
 *      deallocation.
 */
[[nodiscard]] shared_state make_shared_state(state_kind::type state,
                                             query_stream const& deallocation_stream);

/**
 * @brief Create a shared state using the supplied deallocation streams for directly owned storage.
 *
 * The set must cover the streams retained by the data, null-mask, and recursive child buffers of
 * every owned column. This function records the set but does not rebind those buffers.
 *
 * @param[in] state State to wrap.
 * @param[in] deallocation_streams Complete set of deallocation streams for directly owned storage.
 *
 * @pre `deallocation_streams` contains the deallocation stream retained by every directly owned
 *      device buffer in `state`.
 */
[[nodiscard]] shared_state make_shared_state(state_kind::type state,
                                             std::span<query_stream const> deallocation_streams);

namespace detail {

/** @brief Append deallocation streams not already present in `destination`. */
void append_unique_deallocation_streams(std::vector<query_stream>& destination,
                                        std::span<query_stream const> source);

}  // namespace detail

/**
 * @brief Retain `anchor` and `visible` while exposing the visible state's identity.
 *
 * The returned state carries the union of both handles' deallocation streams.
 *
 * @param[in] anchor State that must remain alive as long as `visible`.
 * @param[in] visible Shared state to expose through the returned handle.
 * @return State exposing `visible` while retaining both inputs.
 */
[[nodiscard]] shared_state make_anchored_state(shared_state anchor, shared_state visible);

/**
 * @brief Wrap and anchor a borrowed column view.
 *
 * @param[in] anchor State that owns the storage referenced by `visible`.
 * @param[in] visible Borrowed column view to wrap.
 * @return State exposing `visible` while retaining `anchor`.
 */
[[nodiscard]] shared_state make_anchored_state(shared_state anchor,
                                               state_kind::cudf_column_view visible);

/**
 * @brief Wrap and anchor a borrowed valid-mask view.
 *
 * @param[in] anchor State that owns the storage referenced by `visible`.
 * @param[in] visible Borrowed valid-mask view to wrap.
 * @return State exposing `visible` while retaining `anchor`.
 */
[[nodiscard]] shared_state make_anchored_state(shared_state anchor,
                                               state_kind::valid_mask_view visible);

/**
 * @brief A container for Query Execution Plan (QEP) states.
 *
 * Collects the states of a QEP task into a common container type. This can be an intermediate
 * state between two tasks, or the final query result.
 *
 * # Lifetime
 *
 * The `state_container` is a temporary intermediate value that only lives in between tasks.
 * However, its constituent states can outlive the container when another `state_container` obtains
 * a copy of a shared-state handle. This allows the container to be disassembled, while some of its
 * constituent states are propagated in-situ and the unused states are destroyed. Shared ownership
 * tracks the lifetime of constituent states so that multiple tasks may retain them. Each state is
 * destroyed as soon as it is no longer used.
 *
 * The executor destroys every QEP-owned state before the corresponding `query_context`, whose
 * stream manager keeps query-owned streams alive. State intentionally allowed to escape that scope
 * must independently retain every `query_stream` required by its stream-dependent objects.
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
 * ownership. The `shared_state` elements and their immutable visible states cannot be rebound
 * through the view.
 */
using state_container_view = std::span<shared_state const>;

/**
 * @brief Return a checked typed task-private state from one container slot.
 *
 * @tparam State Concrete task-private state type.
 * @param[in] container State container to inspect.
 * @param[in] slot Zero-based slot containing the task-private state.
 *
 * @throws std::logic_error If `slot` is out of range, the slot or private payload is null, the
 *         slot does not contain `state_kind::task_private`, or its payload is not `State`.
 *
 * @return Mutable reference to the typed task-private payload.
 */
template <typename State>
  requires std::derived_from<State, task_private_state>
[[nodiscard]] State& get_task_private_state(state_container_view container, std::size_t slot)
{
  GQE_EXPECTS(slot < container.size(), "get_task_private_state: slot is out of range");
  GQE_EXPECTS(container[slot] != nullptr, "get_task_private_state: null state slot");

  auto* private_state = std::get_if<state_kind::task_private>(container[slot].get());
  GQE_EXPECTS(private_state != nullptr,
              "get_task_private_state: state slot does not contain task_private");
  GQE_EXPECTS(private_state->data != nullptr,
              "get_task_private_state: task_private payload is null");

  auto* typed_state = dynamic_cast<State*>(private_state->data.get());
  GQE_EXPECTS(typed_state != nullptr,
              "get_task_private_state: task_private payload has an unexpected type");
  return *typed_state;
}

/**
 * @brief Try to convert one shared state to a `cudf::column_view` at zero cost.
 *
 * Returns the owned column's view or the borrowed column view. Returns `std::nullopt` when
 * `state` has another state kind. A null state slot or null owned-column storage is malformed
 * and throws instead.
 *
 * @param[in] state The state to inspect.
 *
 * @throws std::logic_error If `state` is null or carries null owned-column storage.
 *
 * @return The column view, or `std::nullopt` if `state` has a non-column kind.
 */
[[nodiscard]] std::optional<cudf::column_view> to_column_view(shared_state const& state);

/**
 * @brief Try to obtain a non-owning pointer to an indirect-column descriptor.
 *
 * Returns the `indirect_column_state` held by a `state_kind::indirect_column`.
 * Returns `nullptr` when `state` has another state kind. A null state slot is
 * malformed and throws instead.
 *
 * The returned pointer does not extend the descriptor's lifetime. Keep `state`
 * or another shared owner of the descriptor alive while using the pointer.
 * This function performs no device work.
 *
 * @param[in] state The state to inspect.
 *
 * @throws std::logic_error If `state` is null.
 *
 * @return A non-owning pointer to the immutable descriptor, or `nullptr` for
 *         another state kind.
 */
[[nodiscard]] indirect_column_state const* try_get_indirect_column(shared_state const& state);

/**
 * @brief Try to convert one shared state to a Boolean valid-mask view at zero cost.
 *
 * Returns the owned mask's view or the borrowed mask view. Returns `std::nullopt` when `state` has
 * another state kind. A null state slot is malformed and throws instead.
 *
 * The returned view is non-owning. For `valid_mask`, retain `state` while the view is used. For
 * `valid_mask_view`, its external owner must outlive every use of the returned view.
 *
 * @param[in] state The state to inspect.
 *
 * @throws std::logic_error If `state` is null.
 *
 * @return The valid-mask view, or `std::nullopt` if `state` has a non-mask kind.
 */
[[nodiscard]] std::optional<cudf::column_view> to_valid_mask_view(shared_state const& state);

/**
 * @brief Try to convert the state container to a `cudf::table_view` at zero cost.
 *
 * Succeeds when every slot is a `cudf_column` or `cudf_column_view`, optionally preceded by
 * a single `row_count` slot at index 0 (skipped — count-only containers convert to a
 * zero-column table view).
 *
 * Returns `std::nullopt` for empty containers, shapes that would require additional work
 * to materialize, or shapes that have no table-view projection.
 *
 * @param[in] container The state container to inspect.
 *
 * @throws std::logic_error If a slot is null or a `cudf_column` carries null storage.
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
 *  - `valid_mask`                       → `column().size()` (the mask is itself a column).
 *  - `valid_mask_view`                  → `column().size()`.
 *  - `position_list`                    → its typed position-list size.
 *  - `indirect_column`                  → its position-list size.
 *  - `reference`                        → the referenced chunk's retained row count.
 *  - `deferred_column`                  → the deferred source's row count.
 *  - `row_count`                        → the slot's value (column-less containers only).
 *  - `keyed_accumulator`                → throws; the map's group count is device-side only.
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
 * row count is implicit at zero via the empty columns. Because `cudf::make_empty_column` allocates
 * no device buffers, the returned states carry no deallocation-stream metadata.
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
 * Each `shared_state` is copy-constructed. Its underlying state and device storage are shared
 * rather than deep-copied.
 *
 * @param[in] src The state container view to shallow-copy.
 *
 * @return A new state container that shares ownership of the same inner state.
 */
[[nodiscard]] state_container make_mutable_state_copy(state_container_view src);

/**
 * @brief Order state deallocation after previously enqueued reads on `consuming_stream`.
 *
 * The caller must retain every dependency while its final reads and this call are enqueued. The
 * helper installs one event edge from `consuming_stream` to each distinct foreign deallocation
 * stream. It does not release C++ ownership. Work submitted after this call requires another
 * ordering point before the last owner is released.
 *
 * Explicit deallocation streams have stable identities across host threads. Once this helper has
 * ordered such a stream, the last state owner may be released from any host thread. The state
 * retains the corresponding `query_stream` through deallocation.
 *
 * `cudaStreamPerThread` does not identify one stream across host threads. For dependencies whose
 * metadata contains that special handle, this helper orders deallocation only on the calling host
 * thread's per-thread default stream. The last owning reference must therefore be destroyed on the
 * calling host thread after this call. Before the last owner may instead be destroyed on another
 * host thread, the caller must synchronize `consuming_stream` or establish equivalent ordering for
 * that thread's per-thread default stream.
 *
 * The normal path does not synchronize the host. If event ordering fails, the helper synchronizes
 * `consuming_stream` before propagating the failure so that unwinding may safely release owners.
 *
 * @param[in] dependencies States read by the previously enqueued logical batch.
 * @param[in] consuming_stream Stream on which the final reads were enqueued.
 *
 * @pre Every dependency has complete and accurate deallocation-stream metadata for its recursively
 *      owned device buffers.
 */
void order_deallocation_after_stream_use(state_container_view dependencies,
                                         query_stream const& consuming_stream);

/**
 * @brief State container builder.
 *
 * A `state_container_builder` is used to construct an immutable `state_container`. A
 * `valid_mask` or `valid_mask_view` added in any order is placed before all other states in the
 * built container, following the masked-table shape convention. Adding more than one valid mask
 * is invalid.
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
   * A `valid_mask` or `valid_mask_view` is retained separately and placed first by `build()`.
   * Other states preserve their relative insertion order.
   * Use this overload for state kinds other than `cudf_column` and `valid_mask`. Directly owned
   * columns and masks must use the overload that records their retained deallocation stream.
   *
   * @param[in] state The state to add.
   *
   * @throws std::invalid_argument If `state` directly owns column storage or is a valid mask when
   *         one was already added.
   *
   * @return A reference to the builder.
   */
  state_container_builder& add_state(state_kind::type&& state);

  /**
   * @brief Add a state with the deallocation stream for its directly owned column storage.
   *
   * @param[in] state State to add. For an owning column or mask, `deallocation_stream` must cover
   *                  every recursively owned buffer.
   * @param[in] deallocation_stream Deallocation stream for directly owned storage in `state`.
   *
   * @pre Every directly owned device buffer in `state` retains `deallocation_stream` for
   *      deallocation.
   *
   * @return A reference to the builder.
   */
  state_container_builder& add_state(state_kind::type&& state,
                                     query_stream const& deallocation_stream);

  /**
   * @brief Add an already shared state without changing its ownership metadata.
   *
   * Copies or moves the handle into the builder. The underlying state, any lifetime anchor, and
   * the recorded deallocation streams are preserved.
   *
   * @param[in] state Shared state to add.
   *
   * @throws std::invalid_argument If `state` is null or is a valid mask when one was already added.
   *
   * @return A reference to the builder.
   */
  state_container_builder& add_state(shared_state state);

  /**
   * @brief Add a cuDF table whose recursively owned buffers deallocate on one deallocation stream.
   *
   * @param[in] table Table to disassemble into owned column states.
   * @param[in] deallocation_stream Deallocation stream covering every buffer owned by `table`.
   *
   * @pre Every device buffer owned recursively by `table` retains `deallocation_stream` for
   *      deallocation.
   *
   * @return A reference to the builder.
   */
  state_container_builder& add_state(cudf::table&& table, query_stream const& deallocation_stream);

  /**
   * @brief Add a borrowed cuDF table view.
   *
   * The table view is disassembled into borrowed `cudf_column_view` states. No ownership,
   * lifetime anchor, or deallocation-stream metadata is added. The row count is implicit in the
   * columns' `size()`.
   *
   * @param[in] table_view The table view to add.
   *
   * @pre Storage referenced by `table_view` outlives every use of the states produced by this
   *      builder, including asynchronous device work.
   *
   * @return A reference to the builder.
   */
  state_container_builder& add_state(cudf::table_view table_view);

  /**
   * @brief Build the state.
   *
   * Moves all added states into the result and resets the builder to empty, so it may be reused.
   *
   * @return A state container holding the added shared-state handles.
   */
  state_container build();

 private:
  std::optional<shared_state> _valid_mask;  ///< Optional leading masked-table valid mask.
  std::vector<shared_state>
    _states;  ///< Added non-mask states. Semantically, this is not yet a `state_container`.
};

/**
 * @brief Construct a container holding one task-private state.
 *
 * @tparam T State type derived from `task_private_state` so it can be stored as a private payload.
 * @tparam Args Argument types forwarded to the state constructor.
 * @param[in] args Arguments forwarded to the state constructor.
 * @return A container owning the constructed task-private state.
 */
template <typename T, typename... Args>
  requires std::derived_from<T, task_private_state>  // Only task-private payloads can be wrapped.
[[nodiscard]] state_container make_private_state_container(Args&&... args)
{
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<T>(std::forward<Args>(args)...)})
    .build();
}

}  // namespace qep
}  // namespace gqe
