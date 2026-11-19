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

#include <gqe/context_reference.hpp>
#include <gqe/qep/task.hpp>
#include <gqe/storage/in_memory.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <variant>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief Streaming in-memory write task.
 *
 * Adapts the existing in-memory row-group write path to a `qep::fold_task`. Each successful
 * `next()` call materializes one independent row group in the destination table. `finalize()`
 * returns the statistics for the row groups successfully appended by this fold.
 *
 * # Input shape
 *
 * Plain, non-nested table (`[(cudf_column | cudf_column_view)+]`). The input column at position
 * `i` is stored at `destination_column_indexes[i]`, and its type must exactly match
 * `data_types[i]`. Both the column count and types are validated for every chunk. Masked tables
 * must be materialized by a gather task before reaching this operator.
 *
 * A successful call owns the stored row group independently of the input container. The caller may
 * release the input container after `next()` subject to the usual QEP CUDA stream-ordering
 * contract.
 *
 * # Output shape
 *
 * `finalize()` returns `[write_statistics{statistics}]`. Finalizing without any `next()` calls
 * returns zero rows and row groups and leaves the destination table unchanged. A zero-row input
 * chunk still appends one zero-row row group.
 *
 * # Storage
 *
 * Both process-local and shared-memory row-group appenders are supported.
 *
 * # Thread safety and ordering
 *
 * The QEP contract permits concurrent `next()` calls on one accumulator. Row-group construction
 * may proceed concurrently; the destination appender serializes publication, and the accumulator
 * updates its statistics under a lock. However, ordering is not guaranteed to match invocation
 * order.
 *
 * # Limitations (v1)
 *
 * - `destination_column_indexes` must be a permutation of `[0, data_types.size())`; partial-table
 *   writes and duplicate destination columns are not supported.
 * - Nested LIST and STRUCT columns are not supported because `data_types` does not describe their
 *   child schemas across chunks.
 * - Each input chunk becomes one row group; the task does not coalesce or repartition chunks.
 */
class chunked_in_memory_write_task : public qep::fold_task {
 public:
  /** @brief Row-group publication handle for either supported in-memory table implementation. */
  using row_group_appender_t =
    std::variant<in_memory_table::row_group_appender, shared_in_memory_table::row_group_appender>;

  /**
   * @brief Construct a chunked in-memory write task.
   *
   * @param[in] row_group_appender Non-owning, thread-safe destination publication handle.
   * @param[in] memory_kind Memory kind of the destination table.
   * @param[in] destination_column_indexes Destination index for each input column. Must be a
   *   permutation of `[0, data_types.size())`.
   * @param[in] data_types Expected input type per column, in input-column order.
   *
   * @throws std::invalid_argument If the schema is empty, the vectors have different lengths,
   *   the destination indexes are not a permutation, or a type is nested.
   */
  chunked_in_memory_write_task(row_group_appender_t row_group_appender,
                               memory_kind::type memory_kind,
                               std::vector<cudf::size_type> destination_column_indexes,
                               std::vector<cudf::data_type> data_types);

  /**
   * @copydoc gqe::qep::fold_task::initialize
   *
   * @throws std::invalid_argument If either the query context or task-manager context is null.
   */
  [[nodiscard]] qep::state_container initialize(
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::next
   *
   * @throws std::invalid_argument If `inputs` is not a plain table or its schema does not match
   *   the constructor-supplied schema.
   */
  void next(
    qep::state_container_view inputs,
    qep::state_container_view accumulator,
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::finalize
   */
  [[nodiscard]] qep::state_container finalize(
    qep::state_container&& accumulator,
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<qep::task> clone() const override;

 private:
  row_group_appender_t _row_group_appender;
  memory_kind::type _memory_kind;
  std::vector<cudf::size_type> _destination_column_indexes;
  std::vector<cudf::data_type> _data_types;
};

}  // namespace storage
}  // namespace gqe
