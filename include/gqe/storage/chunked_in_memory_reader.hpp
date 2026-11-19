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
#include <gqe/qep/traits/input_residency_trait.hpp>
#include <gqe/qep/traits/splittable_source_trait.hpp>

#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace gqe {

class expression;

namespace storage {

class row_group_interface;

/**
 * @brief Streaming QEP reader for GQE in-memory row groups.
 *
 * Each successful call to `next()` emits one owned chunk from the configured row groups and
 * `std::nullopt` signals end-of-stream. A chunk corresponds to at most one source row group after
 * conservative zone-map pruning. Fully pruned row groups do not produce chunks.
 *
 * # Output Shape
 *
 * Chunks with projected columns are emitted as `[cudf_column, ...]`, in `column_indexes` order.
 * The output columns own their materialized device data.
 * Nullable contiguous fixed-width and string columns are supported. Projected compressed-sliced
 * columns containing null values are rejected during `initialize()`.
 *
 * # Count-Only Mode
 *
 * An empty column list enables count-only mode. The iterator emits exactly one `[row_count{N}]`
 * chunk, where `N` is the aggregate retained row count, without materializing columns. Partial
 * filters are not supported in count-only mode.
 *
 * # Empty Input
 *
 * If no row group produces output, the iterator emits exactly one schema-carrying empty chunk
 * before end-of-stream: a typed zero-row table in projected-column mode, or `[row_count{0}]` in
 * count-only mode. This preserves the output shape for iterate-task adapters.
 *
 * # Predicate Pruning
 *
 * The optional partial filter is used only for conservative zone-map pruning. A row that satisfies
 * the predicate is guaranteed to be included, but a row that does not satisfy it may still be
 * emitted. Exact filtering remains the responsibility of a downstream filter task.
 *
 * # Lifetime
 *
 * Row-group pointers are non-owning. Their pointees, the containing in-memory table, and any shared
 * storage backing them must outlive this task and every iterator state initialized from it.
 *
 * # Thread Safety
 *
 * The task object is a stateless functor after construction. Mutable cursor and pruning state live
 * in the iterator returned by `initialize()`. Concurrent calls to `next()` on the same
 * iterator may use different CUDA streams and must drain each row group exactly once.
 *
 * # Source Splitting
 *
 * Implements `qep::splittable_source_trait` with one work unit per row group.
 * Splits preserve source order and independently own their iteration state.
 * Source row groups must outlive the task and every split created from it.
 */
class chunked_in_memory_read_task final : public qep::iterate_task,
                                          public qep::input_residency_trait,
                                          public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct a chunked in-memory read task.
   *
   * @param[in] row_groups Non-owning pointers to the row groups to read.
   * @param[in] column_indexes Source columns to emit, in output order. Empty enables count-only
   * mode.
   * @param[in] output_data_types Exact stored type for each requested column. Must have the same
   * length as `column_indexes`.
   * @param[in] source_metadata Scheduling metadata for the storage backing the row groups.
   * @param[in] partial_filter Optional logical predicate used for conservative zone-map pruning.
   */
  chunked_in_memory_read_task(std::vector<row_group_interface const*> row_groups,
                              std::vector<cudf::size_type> column_indexes,
                              std::vector<cudf::data_type> output_data_types,
                              qep::source_morsel_metadata source_metadata,
                              std::unique_ptr<gqe::expression> partial_filter = nullptr);

  ~chunked_in_memory_read_task() override;

  [[nodiscard]] qep::source_morsel_metadata input_residency() const noexcept override
  {
    return _source_metadata;
  }

  /** @copydoc gqe::qep::splittable_source_trait::work_unit_count */
  [[nodiscard]] std::size_t work_unit_count() const noexcept override;

  /** @copydoc gqe::qep::splittable_source_trait::split */
  [[nodiscard]] std::unique_ptr<qep::iterate_task> split(
    qep::source_work_range range) const override;

  /**
   * @copydoc gqe::qep::iterate_task::initialize
   */
  [[nodiscard]] qep::state_container initialize(
    qep::state_container_view inputs,
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::iterate_task::next
   */
  [[nodiscard]] std::optional<qep::state_container> next(
    qep::state_container_view iterator,
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<qep::task> clone() const override;

 private:
  std::vector<row_group_interface const*> _row_groups;
  std::vector<cudf::size_type> _column_indexes;
  qep::source_morsel_metadata _source_metadata;
  std::vector<cudf::data_type> _output_data_types;
  std::unique_ptr<gqe::expression> _partial_filter;
};

}  // namespace storage
}  // namespace gqe
