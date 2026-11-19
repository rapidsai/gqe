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
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief Shared source owning the materialization recipe for one deferred chunk.
 *
 * A deferred chunk's columns have not been decoded into cuDF columns yet. The source holds the
 * source-specific recipe (for example a Parquet footer and row-group span, or an in-memory row
 * group and pruning result) and materializes columns on demand. A resolve task produces one source
 * per chunk and emits one `state_kind::deferred_column` per logical column, all sharing the
 * source, so consumers can request exactly the columns they need and a chunk can mix deferred
 * columns with other column kinds.
 *
 * # Selective Materialization
 *
 * `materialize_columns` decodes the requested source columns in one batched operation and hands
 * the owned columns to the caller. The source is an immutable recipe: it retains no decoded data,
 * so a column lives exactly as long as the containers that forward it, and a repeated request
 * decodes again. Consumers forward materialized columns downstream instead of re-requesting them.
 * Column metadata (`num_rows`, `column_type`) is available without device work.
 *
 * # Lifetime
 *
 * A source is valid only within the query that resolved it and must be materialized or dropped
 * before the query completes.
 *
 * # Thread Safety and Streams
 *
 * The source is immutable after construction, so concurrent `materialize_columns` calls are safe
 * and each result is stream-ordered only on its own call's stream.
 */
class deferred_table_source_state {
 public:
  virtual ~deferred_table_source_state() = default;

  deferred_table_source_state(deferred_table_source_state const&)            = delete;
  deferred_table_source_state& operator=(deferred_table_source_state const&) = delete;

  /** @brief Return the chunk's row count without materializing any column. */
  [[nodiscard]] cudf::size_type num_rows() const noexcept { return _num_rows; }

  /** @brief Return the chunk's logical column count. */
  [[nodiscard]] cudf::size_type num_columns() const noexcept
  {
    return static_cast<cudf::size_type>(_column_types.size());
  }

  /**
   * @brief Return one column's cuDF type without materializing it.
   *
   * @throws std::out_of_range If `source_column_index` is outside `[0, num_columns())`.
   */
  [[nodiscard]] cudf::data_type column_type(cudf::size_type source_column_index) const;

  /**
   * @brief Materialize the requested source columns in one batched decode.
   *
   * @param[in] source_column_indices Source column indices in result order. Repeats are valid and
   * decode once within the request.
   * @param[in] stream_handle CUDA stream for the materialization.
   * @param[in] mr Memory resource for the materialization.
   *
   * @throws std::out_of_range If any requested index is outside `[0, num_columns())`.
   *
   * @return One owned-column state per requested index, in request order. Repeated indices share
   * one state.
   */
  [[nodiscard]] std::vector<shared_state> materialize_columns(
    std::span<cudf::size_type const> source_column_indices,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 protected:
  /**
   * @brief Construct a deferred source.
   *
   * @param[in] num_rows Row count of the deferred chunk. Must not be negative.
   * @param[in] column_types Exact cuDF type of each logical column. Must not be empty.
   */
  deferred_table_source_state(cudf::size_type num_rows, std::vector<cudf::data_type> column_types);

  /**
   * @brief Decode the given source columns in one batched operation.
   *
   * Implementations must not consume their inputs: the source is immutable and may be asked for
   * the same columns again.
   *
   * @param[in] source_column_indices Strictly increasing distinct source column indices.
   * @param[in] stream_handle CUDA stream for the materialization. Every device buffer owned by the
   * returned columns must be allocated on this stream.
   * @param[in] mr Memory resource for the materialization.
   *
   * @return One decoded column per requested index, aligned with `source_column_indices`. Each
   * column must have the declared `column_type` and `num_rows`, and its owned device buffers must
   * have been allocated from `mr` on `stream`.
   */
  [[nodiscard]] virtual std::vector<std::unique_ptr<cudf::column>> materialize_columns_impl(
    std::span<cudf::size_type const> source_column_indices,
    query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const = 0;

 private:
  cudf::size_type _num_rows;
  std::vector<cudf::data_type> _column_types;
};

/**
 * @brief Immutable descriptor for one deferred logical column.
 *
 * Retains one shared deferred source and one source-column index for its lifetime.
 */
class deferred_column_state {
 public:
  /**
   * @brief Build one validated deferred-column state.
   *
   * @param[in] source Shared deferred source. Must not be null.
   * @param[in] source_column_index Column index within the source.
   *
   * @throws std::invalid_argument If `source` is null.
   * @throws std::out_of_range If `source_column_index` is outside `[0, num_columns())`.
   */
  deferred_column_state(std::shared_ptr<deferred_table_source_state> source,
                        cudf::size_type source_column_index);

  /** @brief Return the shared deferred source. */
  [[nodiscard]] std::shared_ptr<deferred_table_source_state> const& source() const noexcept
  {
    return _source;
  }

  /** @brief Return the column index within the source. */
  [[nodiscard]] cudf::size_type source_column_index() const noexcept
  {
    return _source_column_index;
  }

 private:
  std::shared_ptr<deferred_table_source_state> _source;
  cudf::size_type _source_column_index;
};

/**
 * @brief Build a deferred-chunk state container `[deferred_column...]`.
 *
 * @param[in] source The shared deferred source. Must not be null.
 *
 * @return A state container holding one `deferred_column` slot per source column, in source
 * column order.
 */
[[nodiscard]] state_container make_deferred_column_container(
  std::shared_ptr<deferred_table_source_state> source);

/**
 * @brief Materialize the deferred columns in selected slots, passing every other slot through.
 *
 * The listed slots that hold deferred columns are materialized in one batched decode and
 * replaced by the owned columns. Listed slots of any other kind, and every unlisted slot
 * (including unlisted deferred columns), are returned as shallow copies in place, so a consumer
 * can materialize exactly the columns it needs and forward the rest still deferred. All deferred
 * columns in a container must share one source, as a resolve task emits.
 *
 * @param[in] container The state container to convert.
 * @param[in] slot_indexes Container slot indexes to materialize. Repeats are valid.
 * @param[in] stream_handle CUDA stream for a materialization, if one happens.
 * @param[in] mr Memory resource for a materialization, if one happens.
 *
 * @throws std::out_of_range If any slot index is outside `[0, container.size())`.
 * @throws std::logic_error If the selected deferred columns come from different sources.
 *
 * @return A state container with the selected deferred columns materialized.
 */
[[nodiscard]] state_container materialize_deferred_columns(
  state_container_view container,
  std::span<std::size_t const> slot_indexes,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Materialize every deferred column in a container, passing every other slot through.
 *
 * Equivalent to the slot-selecting overload applied to every slot.
 *
 * @param[in] container The state container to convert.
 * @param[in] stream_handle CUDA stream for a materialization, if one happens.
 * @param[in] mr Memory resource for a materialization, if one happens.
 *
 * @throws std::logic_error If the deferred columns come from different sources.
 *
 * @return A state container without deferred columns.
 */
[[nodiscard]] state_container materialize_deferred_columns(
  state_container_view container,
  query_stream const& stream_handle = query_stream::default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace qep
}  // namespace gqe
