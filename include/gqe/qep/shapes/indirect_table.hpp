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

#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cuda/stream>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {
namespace qep {

class indirect_table_source_state;
class indirect_gather_source_state;

namespace detail {

/** @brief Construction key that restricts public constructors to their owning static factories. */
template <typename T>
class passkey {
  friend T;

 public:
  passkey(passkey const&) = default;
  passkey(passkey&&)      = default;
  ~passkey()              = default;

 private:
  passkey() = default;
};

}  // namespace detail

/** @brief Shared handle to an immutable contiguous relation source. */
using indirect_table_source = std::shared_ptr<indirect_table_source_state const>;

/** @brief Shared handle to an immutable relation source, position list, and bounds policy. */
using indirect_gather_source = std::shared_ptr<indirect_gather_source_state const>;

/**
 * @brief Immutable contiguous relation source for indirect columns.
 *
 * A contiguous relation source contains one row-aligned table whose columns may each be represented
 * by either a `cudf_column` or a `cudf_column_view` state. A gather source pairs this relation
 * source with a position list that selects rows. Each indirect column identifies one
 * relation-source column at those positions. The relation source shares ownership of every retained
 * column state. A `cudf_column` state owns its cuDF storage. A `cudf_column_view` state borrows
 * storage whose external owner must keep it alive during every use of the view.
 *
 * Once published through an indirect column, the retained state kinds, row counts, and column data
 * do not change while any indirect column references the source. Borrowed column storage must also
 * remain unchanged for that period.
 */
class indirect_table_source_state {
 public:
  /**
   * @brief Build a validated immutable contiguous relation source.
   *
   * `columns` must contain at least one state. Each state must be a non-null `cudf_column` or
   * `cudf_column_view`, and every column must have the same physical row count. The container is
   * moved into the source.
   *
   * @param[in] columns Row-aligned states, each either `cudf_column` or `cudf_column_view`.
   *
   * @return Shared immutable relation source.
   *
   * @throws std::logic_error If a state or its owned storage is null.
   * @throws std::invalid_argument If the source is empty, contains another state kind, or has
   *         mismatched column row counts.
   */
  [[nodiscard]] static indirect_table_source make(state_container columns);

  /** @brief Construct through the owning static factory; the passkey blocks direct use. */
  indirect_table_source_state(detail::passkey<indirect_table_source_state>,
                              state_container columns);

  [[nodiscard]] cudf::size_type row_count() const noexcept { return _row_count; }

  [[nodiscard]] cudf::size_type num_columns() const noexcept { return _num_columns; }

  /** @brief Return the union of retained source-column deallocation streams. */
  [[nodiscard]] std::span<query_stream const> deallocation_streams() const noexcept
  {
    return _deallocation_streams;
  }

  /**
   * @brief Return one retained source column as a zero-cost cuDF view.
   *
   * @param[in] source_column_index Index in the contiguous relation source.
   *
   * @throws std::out_of_range If `source_column_index` is outside `[0, num_columns())`.
   */
  [[nodiscard]] cudf::column_view column(cudf::size_type source_column_index) const;

  /**
   * @brief Return all retained source columns as one `cudf::table_view`.
   *
   * The relation source keeps owned column storage alive. The external owner of a borrowed column
   * must remain alive until every use of the returned view completes.
   */
  [[nodiscard]] cudf::table_view contiguous_table_view() const;

 private:
  state_container _columns;
  std::vector<query_stream> _deallocation_streams;
  cudf::size_type _row_count{};
  cudf::size_type _num_columns{};
};

/**
 * @brief Immutable relation source, position list, and bounds policy shared by indirect columns.
 *
 * Once published through an indirect column, the relation-source handle, position-list state and
 * values, and bounds policy do not change while any indirect column references the gather source.
 *
 * Multiple indirect columns that reference the same gather-source instance can be materialized in
 * one gather operation.
 */
class indirect_gather_source_state {
 public:
  /**
   * @brief Build a validated immutable gather source.
   *
   * @param[in] relation Relation source to gather.
   * @param[in] positions Typed position-list handle.
   * @param[in] policy Bounds policy to apply when gathering `relation` with `positions`.
   *
   * @return Shared immutable gather source.
   *
   * @throws std::invalid_argument If `relation` or `positions` is null.
   */
  [[nodiscard]] static indirect_gather_source make(
    indirect_table_source relation,
    position_list_handle positions,
    cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK);

  /** @brief Construct through the owning static factory; the passkey blocks direct use. */
  indirect_gather_source_state(detail::passkey<indirect_gather_source_state>,
                               indirect_table_source relation,
                               position_list_handle positions,
                               cudf::out_of_bounds_policy policy);

  /** @brief Return the relation source that supplies source columns. */
  [[nodiscard]] indirect_table_source const& relation_source() const noexcept { return _relation; }

  /** @brief Return the position list that defines the indirect columns' logical row count. */
  [[nodiscard]] position_list_handle const& position_list() const noexcept { return _positions; }

  /** @brief Return the bounds policy passed to `cudf::gather` during materialization. */
  [[nodiscard]] cudf::out_of_bounds_policy bounds_policy() const noexcept { return _policy; }

  /** @brief Return the union of relation and position-list deallocation streams. */
  [[nodiscard]] std::span<query_stream const> deallocation_streams() const noexcept
  {
    return _deallocation_streams;
  }

 private:
  indirect_table_source _relation;
  position_list_handle _positions;
  std::vector<query_stream> _deallocation_streams;
  cudf::out_of_bounds_policy _policy;
};

/**
 * @brief Immutable descriptor for one indirect logical column.
 *
 * Retains one gather source and one source-column index for its lifetime.
 */
class indirect_column_state {
 public:
  /**
   * @brief Build one validated indirect-column state.
   *
   * @param[in] gather_source Shared relation, position list, and bounds policy.
   * @param[in] source_column_index Column index within the gather source's relation.
   *
   * @return Shared state containing the validated indirect column.
   *
   * @throws std::invalid_argument If `gather_source` is null, or `source_column_index` is out of
   *         range.
   */
  [[nodiscard]] static shared_state make(indirect_gather_source gather_source,
                                         cudf::size_type source_column_index);

  /** @brief Construct through the owning static factory; the passkey blocks direct use. */
  indirect_column_state(detail::passkey<indirect_column_state>,
                        indirect_gather_source gather_source,
                        cudf::size_type source_column_index);

  /** @brief Return the relation, position list, and bounds policy for this column. */
  [[nodiscard]] indirect_gather_source const& gather_source() const noexcept
  {
    return _gather_source;
  }

  /** @brief Return the index into `gather_source()->relation_source()`. */
  [[nodiscard]] cudf::size_type source_column_index() const noexcept
  {
    return _source_column_index;
  }

 private:
  indirect_gather_source _gather_source;
  cudf::size_type _source_column_index;
};

/**
 * @brief Identity of an indirect logical column within a validated table view.
 */
struct indirect_column_view {
  std::size_t output_index;  ///< Ordinal in the owning `indirect_table_view::columns`; meaningful
                             ///< only in the context of that table view.
  indirect_gather_source gather_source;  ///< Pointer identity determines gather grouping.
  cudf::size_type source_column_index;   ///< Index into `gather_source->relation_source()`.
};

/**
 * @brief A validated table view containing at least one indirect column.
 *
 * Logical columns may be `cudf_column`, `cudf_column_view`, or `indirect_column` states in any
 * order. An optional `valid_mask` or `valid_mask_view` may precede them. At least one logical
 * column must be an `indirect_column` state. `try_from` delegates common relation and geometry
 * validation to `logical_table_view`, then projects indirect-column metadata.
 */
struct indirect_table_view {
  /**
   * @brief Recognize and validate a state container containing indirect columns.
   *
   * @param[in] container State container to inspect.
   *
   * @return A validated indirect-table view, or `std::nullopt` if the container has no
   * `indirect_column` state.
   *
   * @throws std::logic_error If a container slot is null. For a container with an `indirect_column`
   * state, also if a `cudf_column` state has a null column, a logical column has another state
   * kind, or logical row counts differ.
   */
  [[nodiscard]] static std::optional<indirect_table_view> try_from(state_container_view container);

  cudf::size_type row_count;              ///< Validated logical row count.
  std::optional<cudf::column_view> mask;  ///< Optional leading BOOL8 valid mask.
  shared_state mask_state;                ///< Source state retained when `mask` exists.
  std::vector<shared_state> columns;      ///< Logical columns in output order.
  std::vector<indirect_column_view> indirect_columns;  ///< Indirect columns in output order.
};

/**
 * @brief Slice a validated indirect table by logical row range.
 *
 * # Representation
 *
 * A complete-range slice returns the original mask and logical-column states by shared ownership.
 * Any other valid range returns borrowed slices of direct columns and the optional mask, anchored
 * to their source states. Each distinct indirect gather source is rebuilt with one owned slice of
 * its position list, while columns that shared a gather source continue to share the rebuilt
 * source. Relation sources, bounds policies, and logical column order are preserved.
 *
 * # Lifetime and CUDA Ordering
 *
 * Position-list slices are returned allocations and use `mr`. All cuDF work is enqueued on
 * `stream_handle`; this function does not synchronize. Rebuilt indirect-column slots retain their
 * original source states as host-lifetime anchors for asynchronous position-list copies. The
 * caller or executor must stream-order destruction of the returned container after that work. An
 * external owner of storage referenced by an input `cudf_column_view` or `valid_mask_view` must
 * also outlive every use of the corresponding returned view.
 *
 * # Complexity
 *
 * With C logical columns and S distinct gather sources, slicing any range other than the complete
 * range has expected O(C + S) host complexity and O(S) auxiliary host space, excluding returned
 * states, returned device allocations, and cuDF work. Gather sources are grouped by pointer
 * identity.
 *
 * @param[in] input Validated indirect table to slice.
 * @param[in] begin Inclusive logical row index.
 * @param[in] end Exclusive logical row index.
 * @param[in] stream_handle CUDA stream lifetime capability for slicing work.
 * @param[in] mr Device memory resource for owned position-list slices.
 *
 * @return One indirect-table-shaped state container for `[begin, end)`.
 *
 * @throws std::invalid_argument If the requested range is outside `[0, input.row_count]`.
 */
[[nodiscard]] state_container slice_indirect_table(indirect_table_view const& input,
                                                   cudf::size_type begin,
                                                   cudf::size_type end,
                                                   query_stream const& stream_handle,
                                                   rmm::device_async_resource_ref mr);

}  // namespace qep
}  // namespace gqe
