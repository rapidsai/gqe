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

#include <cstdint>
#include <memory>

namespace gqe::qep {

/**
 * @brief Emit `count` rows of the input stream, skipping the first `offset` rows.
 *
 * Implements the SQL fetch operation (`LIMIT`/`OFFSET`). With unordered input, the task returns
 * an arbitrary subset of the requested size. When the caller supplies an ordered result as one
 * complete chunk, the task selects the requested window in that sequence and retains its relative
 * row order.
 * Column-bearing logical relations are fully compacted before window intersection, so the window
 * counts surviving rows for direct, indirect, masked, and masked-indirect input alike.
 *
 * # Semantics
 *
 * Each `next` call consumes its input's row range of the stream and emits the intersection of
 * that range with the window `[offset, offset + count)`:
 *
 *  - The range lies inside the window. The fully compacted relation is emitted in full.
 *  - The range overlaps a window boundary. The task emits an owned copy of the intersecting
 *    slice, bounded by `count` rows.
 *  - The range lies past the window. The task returns `std::nullopt`, the early-exit signal.
 *  - The range lies before the window, or it is the first range of an empty window. The task
 *    emits a schema-correct zero-row slice, so the schema always reaches downstream consumers.
 *
 * Concurrent `next` calls consume disjoint ranges, so the emitted outputs never overlap and total
 * at most `count` rows. With independently processed chunks, which rows fall inside the window
 * depends on the unspecified order in which concurrent calls consume the stream.
 *
 * # Input and Output Shapes
 *
 * The input is any column-bearing logical relation or a count-only `[row_count]` container.
 * Count-only output preserves that shape. Column-bearing output is a plain table;
 * indirect columns are gathered and the validity mask is applied.
 *
 * # Limitations
 *
 * The task does not enforce input sequence across independently processed chunks. To preserve
 * ordered LIMIT/OFFSET semantics, the caller must supply the complete ordered result as at most
 * one present relational chunk per execution. `pipeline_normalization` enforces this requirement
 * for normalized ordered paths. Callers that invoke the task directly must enforce it themselves.
 */
class fetch_task : public stateful_transform_task {
 public:
  /**
   * @brief Construct a fetch task.
   *
   * @param[in] offset Number of leading stream rows to skip. Must be non-negative.
   * @param[in] count Maximum number of rows to emit after `offset`. Must be non-negative, and
   *   `offset + count` must not overflow `int64_t`.
   */
  fetch_task(int64_t offset, int64_t count);

  /** @copydoc stateful_transform_task::initialize */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc stateful_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  int64_t _offset;
  int64_t _count;
};

}  // namespace gqe::qep
