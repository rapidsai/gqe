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

#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cuda/stream>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <span>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Move-only result containing selectively materialized logical columns.
 *
 * Requested columns appear in request order, and repeated logical indices produce repeated result
 * entries. Repeated indices are processed independently without a shared-work guarantee.
 * Materialization has two modes:
 *
 * - Without validity-mask compaction, direct column states retain their owning or borrowed form
 *   and indirect columns become owning `cudf_column` states. The result contains data columns only.
 *   Any relation-level validity mask remains unchanged and available from the input
 *   `logical_table_view`.
 * - With validity-mask compaction, a present mask is applied and the requested columns become
 *   owning `cudf_column` states, so no relation-level mask is needed to interpret the result. An
 *   unmasked input behaves as it does without compaction.
 *
 * # Ownership and Lifetime
 *
 * The result retains selected direct states because they are genuine zero-copy outputs. Newly
 * materialized columns own independent storage. Inputs used only by asynchronous materialization
 * remain alive until their final reads are enqueued, then their deallocation streams are ordered
 * after those reads. Borrowed input storage remains externally owned and must outlive all work
 * that reads it. Copying is disabled. Moving transfers the materialized states and request
 * metadata to the destination.
 *
 * `release_column_states()` transfers the ordered data states into a `state_container`. Releasing
 * a zero-column result is invalid because an empty `state_container` denotes absent upstream
 * output. Use the `row_count` shape when publishing a column-less result.
 *
 * # CUDA Ordering and Memory Resources
 *
 * Materialization enqueues all cuDF and RMM work on the supplied stream and does not synchronize.
 * Every output and temporary allocation uses the supplied memory resource. The caller must keep
 * that resource and externally owned borrowed storage alive until the enqueued work and destruction
 * of its allocations complete. The executor is responsible for producer-consumer stream ordering.
 *
 * # Thread Safety
 *
 * Distinct results may be read concurrently. Mutating one result by moving it or releasing its
 * states requires exclusive host access.
 */
class column_materialization {
 public:
  ~column_materialization() = default;

  column_materialization(column_materialization const&)                = delete;
  column_materialization& operator=(column_materialization const&)     = delete;
  column_materialization(column_materialization&&) noexcept            = default;
  column_materialization& operator=(column_materialization&&) noexcept = default;

  /**
   * @brief Return the logical indices in request order.
   *
   * The returned span is invalidated when this object is moved from or destroyed.
   */
  [[nodiscard]] std::span<cudf::size_type const> logical_column_indices() const noexcept
  {
    return _logical_column_indices;
  }

  /**
   * @brief Return a convenience table view over the request-order materialized columns.
   *
   * The returned view borrows column storage. That storage must remain owned by this result, its
   * moved-to destination, or the `state_container` returned by `release_column_states()`. Storage
   * referenced by a borrowed input column remains externally owned as described by the class
   * contract.
   */
  [[nodiscard]] cudf::table_view table_view() const;

  /**
   * @brief Return the request-order materialized shared states.
   *
   * The returned span is invalidated when this object is moved from, consumed by
   * `release_column_states()`, or destroyed.
   */
  [[nodiscard]] state_container_view states() const;

  /**
   * @brief Return the result row count.
   *
   * Preserving materialization reports the input geometry. Compacted materialization reports the
   * rows selected by the validity mask.
   */
  [[nodiscard]] cudf::size_type row_count() const noexcept { return _row_count; }

  /**
   * @brief Return retained compacted-to-input row positions, or null when not requested or
   * unmasked.
   *
   * The returned reference borrows this result's handle. Copy the handle to retain the position
   * storage independently of this result's lifetime or subsequent moves. The position storage is
   * owned independently of the materialized column states.
   */
  [[nodiscard]] position_list_handle const& row_positions() const noexcept
  {
    return _row_positions;
  }

  /**
   * @brief Transfer request-order data-column states into a state container.
   *
   * This operation consumes the materialization. The object is valid but has an unspecified value
   * afterward, following normal moved-from object semantics.
   *
   * Without validity-mask compaction, the returned container omits the input relation's validity
   * mask. Publishing that container without restoring the mask would incorrectly treat masked-out
   * rows as valid.
   *
   * @throws std::logic_error If no column was requested.
   */
  [[nodiscard]] state_container release_column_states() &&;

 private:
  friend column_materialization materialize_columns(logical_table_view,
                                                    std::span<cudf::size_type const>,
                                                    query_stream const&,
                                                    rmm::device_async_resource_ref);
  friend column_materialization materialize_compacted_columns(logical_table_view,
                                                              std::span<cudf::size_type const>,
                                                              bool,
                                                              query_stream const&,
                                                              rmm::device_async_resource_ref);

  column_materialization(std::vector<cudf::size_type> logical_column_indices,
                         state_container states,
                         cudf::size_type row_count);

  std::vector<cudf::size_type> _logical_column_indices;
  state_container _states;
  cudf::size_type _row_count;
  position_list_handle _row_positions;
};

// ----- Materialization without validity-mask compaction -----

/**
 * @brief Materialize selected logical columns while preserving relation row geometry.
 *
 * Direct columns are zero-copy. Requested indirect columns are grouped by gather-source pointer
 * identity and materialized with one gather per source group. Result columns retain request order,
 * including repetitions. The optional validity mask remains on `input` and is not applied or
 * published by the result.
 *
 * @param[in] input Validated logical relation. Its borrowed container must outlive this call.
 * @param[in] logical_column_indices Logical data-column indices in result order.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Device memory resource for every allocation.
 *
 * @return Move-only materialization retaining its output and asynchronous dependencies.
 *
 * @throws std::out_of_range If any requested logical index is invalid.
 */
[[nodiscard]] column_materialization materialize_columns(
  logical_table_view input,
  std::span<cudf::size_type const> logical_column_indices,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize every logical column while preserving relation row geometry.
 *
 * This is equivalent to requesting `[0, input.num_columns())` with `materialize_columns`.
 *
 * @param[in] input Validated logical relation. Its borrowed container must outlive this call.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Device memory resource for every allocation.
 *
 * @return Move-only materialization in logical schema order.
 */
[[nodiscard]] column_materialization materialize_all_columns(
  logical_table_view input,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ----- Materialization with validity-mask compaction -----

/**
 * @brief Materialize selected logical columns after applying the relation-level valid mask.
 *
 * Unmasked input is equivalent to preserving materialization. For masked input, the validity mask
 * selects rows and no mask remains in the result. Each distinct requested indirect
 * position list is compacted once before one gather per requested gather-source group. Requested
 * direct columns and distinct indirect position lists are compacted together. Intrinsic cuDF null
 * masks retain normal cuDF semantics.
 *
 * A zero-column request is valid. It performs no indirect gather and still computes the surviving
 * row count from a mask.
 *
 * @param[in] input Validated logical relation. Its borrowed container must outlive this call.
 * @param[in] logical_column_indices Logical data-column indices in result order.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Device memory resource for every allocation.
 *
 * @return Move-only compacted materialization retaining output and asynchronous dependencies.
 *
 * @throws std::out_of_range If any requested logical index is invalid.
 */
[[nodiscard]] column_materialization materialize_compacted_columns(
  logical_table_view input,
  std::span<cudf::size_type const> logical_column_indices,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize selected columns and optionally retain their original logical row positions.
 *
 * This overload shares the compaction and lifetime contracts above. Requested row positions are
 * compacted together with data and indirect position lists to avoid repeating mask selection.
 * Unmasked inputs return a null row-position handle because their row coordinates are unchanged.
 *
 * @param[in] input Validated logical relation. Its borrowed container must outlive this call.
 * @param[in] logical_column_indices Logical data-column indices in result order.
 * @param[in] retain_row_positions Whether to retain compacted-to-input positions for masked input.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Device memory resource for every allocation.
 * @return Move-only compacted columns with optional independently owned row positions.
 * @throws std::out_of_range If any requested logical index is invalid.
 */
[[nodiscard]] column_materialization materialize_compacted_columns(
  logical_table_view input,
  std::span<cudf::size_type const> logical_column_indices,
  bool retain_row_positions,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize and compact every logical column.
 *
 * This is equivalent to requesting `[0, input.num_columns())` with
 * `materialize_compacted_columns`.
 *
 * @param[in] input Validated logical relation. Its borrowed container must outlive this call.
 * @param[in] stream_handle CUDA stream for all device work.
 * @param[in] mr Device memory resource for every allocation.
 *
 * @return Move-only compacted materialization in logical schema order.
 */
[[nodiscard]] column_materialization materialize_all_compacted_columns(
  logical_table_view input,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace qep
}  // namespace gqe
