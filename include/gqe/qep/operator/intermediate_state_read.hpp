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

#include <gqe/qep/task.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Emit chunks from materialized QEP state.
 *
 * # Supported Shapes
 *
 * - Plain-table chunks contain `cudf_column_view` slots.
 * - Masked-table chunks contain a leading `valid_mask_view` followed by `cudf_column_view` slots.
 * - Indirect-table chunks preserve direct and indirect logical columns plus any leading valid mask.
 * - Count-only chunks contain one `row_count` slot.
 *
 * Zero-row input emits one empty chunk that preserves its shape. Count-only input emits one chunk
 * containing the complete row count regardless of the configured row bound.
 *
 * # Lifetime
 *
 * Plain, masked, and indirect chunks retain the source state needed by borrowed views and indirect
 * descriptors, so owned input storage remains valid after iterator finalization. When an input
 * slot is already borrowed, its external owner must still outlive every emitted chunk.
 *
 * Retaining source states provides host-side ownership; it does not order CUDA work. The executor
 * owns the iterator and emitted state and must stream-order their destruction after GPU work that
 * accesses them. The task does not synchronize CUDA streams.
 *
 * # Chunking
 *
 * - Complete-range output reuses the validated input states directly.
 * - Bounded plain or masked output performs one batched cuDF slice during `initialize`.
 * - Bounded indirect output slices each distinct gather source's position list once per chunk on
 *   the supplied stream and result memory resource.
 * - Count-only input never calls cuDF.
 *
 * `next` only claims an immutable precomputed chunk.
 *
 * # Concurrency
 *
 * Concurrent `next` calls atomically claim exhaustive, non-overlapping chunks. Return order is not
 * guaranteed. Bounded readers therefore provide no row-sequence guarantee even if their input is
 * ordered. Normalization uses `single_output` for ordered input, so the complete sequence is
 * emitted once and downstream preserving tasks process at most one present chunk.
 */
class intermediate_state_read_task final : public iterate_task {
 public:
  /**
   * @brief Return whether this reader emits the complete input once.
   */
  [[nodiscard]] bool is_single_output() const noexcept { return !_max_rows_per_output.has_value(); }

  /**
   * @brief Tag selecting one chunk containing all input rows.
   */
  struct single_output_t {
    explicit constexpr single_output_t() = default;
  };

  inline static constexpr single_output_t single_output{};

  /**
   * @brief Construct a reader that emits the complete input once.
   */
  explicit intermediate_state_read_task(single_output_t);

  /**
   * @brief Construct a reader that emits at most `max_rows_per_output` rows per table output.
   *
   * Count-only input emits one output containing its complete row count.
   *
   * @param[in] max_rows_per_output Positive maximum number of rows in each output.
   *
   * @throws std::invalid_argument If `max_rows_per_output` is not positive.
   */
  explicit intermediate_state_read_task(cudf::size_type max_rows_per_output);

  /** @copydoc iterate_task::initialize */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc iterate_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::optional<cudf::size_type> _max_rows_per_output;
};

}  // namespace qep
}  // namespace gqe
