/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda/stream>
#include <cudf/table/table_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace gqe {
class expression;
}

namespace gqe::qep::mark_join_detail {

/**
 * @brief Private, host-callable CUDA core for mark joins.
 *
 * This interface has no QEP task or state-container dependencies. Callers own the map handles
 * and payload columns. Keeping CuCo storage opaque lets task orchestration compile as ordinary
 * C++ without exposing device templates.
 *
 * # Ownership and stream ordering
 *
 * Each probe accumulator owns one mutable map and retains the payload while the map is in use.
 * The caller orders map construction before probing and all probes before scanning.
 * Probes may run concurrently on one mutable map because marks are atomic. Map operations
 * reverse-order the allocation stream after their uses before returning. Callers provide the
 * corresponding lifetime ordering for payloads, bitmaps, and selection buffers.
 *
 * # Device implementation
 *
 * Probing and position compaction use warp-cooperative traversal and buffering, with 1024-thread
 * blocks and occupancy-based grids. Residual AST scratch must fit that block size. Probing throws
 * before launching when it exceeds the device's per-block shared-memory limit.
 *
 * Build hashes and probe hash/index pairs are materialized before insertion and probing.
 * Probe kernels count first-time marks within each map. The high-selectivity specialization skips
 * entries already marked by earlier probe rows or chunks.
 *
 * A completed map is scanned into an exactly sized position list. Completion unions those positions
 * across independent maps into a row bitmap because their counts cannot be summed. Each scan then
 * counts its chunk's selected rows before allocating and compacting positions. This preserves
 * indirect output.
 *
 * All GQE-owned device allocations use `mr` and `stream`. cuDF controls key and AST
 * table-descriptor scratch because its factory has no memory-resource argument. CUDA failures and
 * unsupported predicates throw. Numeric key types and equal key counts are validated by the task
 * layer.
 */
class hash_index;

/** @brief Destroy opaque CuCo storage in the CUDA translation unit. */
struct hash_index_deleter {
  void operator()(hash_index* index) const;
};

using hash_index_handle = std::unique_ptr<hash_index, hash_index_deleter>;

/**
 * @brief Own a zero-initialized mutable bitmap outside immutable relational column state.
 *
 * A bitmap contains one uint8_t per build row. Mutable access is reserved for the owning
 * accumulator. Readers use the const overload of `data` after the completion barrier.
 *
 * Call `order_after` after cross-stream uses and before releasing the last owner. The
 * allocation stream must remain valid through destruction. Initialization must precede writes.
 * Scans writing to the same bitmap must be stream-ordered, and all writes must precede readers.
 */
class mark_bitmap {
 public:
  /**
   * @brief Allocate and zero a bitmap.
   * @param[in] rows Nonnegative number of build rows, supplied by validated table metadata.
   * @param[in] stream Stream on which device work is ordered.
   * @param[in] mr Resource for device allocations, including temporary storage.
   */
  mark_bitmap(cudf::size_type rows, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /** @brief Mutable device data for the owning accumulator or local scan buffer. */
  [[nodiscard]] uint8_t* data() noexcept { return _marks.data(); }

  /** @brief Read-only device data for a completed probe. */
  [[nodiscard]] uint8_t const* data() const noexcept { return _marks.data(); }

  /**
   * @brief Order deallocation after the last use without host synchronization on the normal path.
   * @param[in] stream Stream on which device work is ordered.
   */
  void order_after(cuda::stream_ref stream) const;

 private:
  cuda::stream_ref _stream;
  rmm::device_uvector<uint8_t> _marks;
};

// ----- Hash map construction and probing -----

/**
 * @brief Build an unmarked map from numeric keys.
 *
 * @param[in] keys Validated numeric equality keys.
 * @param[in] mask Optional BOOL8 row-selection mask. False and null entries are excluded.
 * @param[in] stream Stream on which device work is ordered.
 * @param[in] mr Resource for device allocations, including temporary storage.
 */
[[nodiscard]] hash_index_handle make_hash_index(cudf::table_view keys,
                                                std::optional<cudf::column_view> mask,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @brief Mark matching build rows, optionally evaluating a residual predicate.
 *
 * Empty inputs perform no work.
 *
 * @param[in,out] index Execution-owned mutable map shared by probe calls.
 * @param[in] left Numeric build keys used to construct the map.
 * @param[in] right Numeric probe keys with matching types and key count.
 * @param[in] left_payload Payload used by the residual predicate.
 * @param[in] right_payload Payload used by the residual predicate.
 * @param[in] left_column_count Full logical build width, including unmaterialized columns.
 * @param[in] left_columns Logical indices of the materialized build columns.
 * @param[in] right_columns Logical indices of the materialized probe columns.
 * @param[in] probe_mask Optional BOOL8 mask. False and null probe rows are skipped.
 * @param[in] predicate Optional predicate in logical [left..., right...] coordinates.
 * @param[in] nulls Equality policy for null keys.
 * @param[in] stream Stream on which device work is ordered.
 * @param[in] mr Resource for device allocations, including temporary storage.
 */
void probe_hash_index(hash_index& index,
                      cudf::table_view left,
                      cudf::table_view right,
                      cudf::table_view left_payload,
                      cudf::table_view right_payload,
                      cudf::size_type left_column_count,
                      std::span<cudf::size_type const> left_columns,
                      std::span<cudf::size_type const> right_columns,
                      std::optional<cudf::column_view> probe_mask,
                      expression const* predicate,
                      cudf::null_equality nulls,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr);

// ----- Completed mark scanning -----

/**
 * @brief Scan a completed map into positions and union them into a row bitmap.
 *
 * The map's probe count sizes a temporary position list.
 * Each build row occurs once in a map, so its bitmap entry is written without atomics.
 * Calls writing to the same bitmap must be stream-ordered by the caller.
 *
 * @param[in] index Map whose probes have completed.
 * @param[in,out] marks Device bitmap with one uint8_t per build row. Zero means unmatched.
 * @param[in] stream Stream on which device work is ordered.
 * @param[in] mr Resource for device allocations, including temporary storage.
 */
void scan_hash_index(hash_index const& index,
                     uint8_t* marks,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr);

/**
 * @brief Count selected rows and compact their indices.
 *
 * The count is read before allocating exactly enough output positions. Output order is
 * unspecified. Position writes are ordered on `stream`.
 * @param[in] marks Completed device bitmap with `rows` entries.
 * @param[in] rows Number of bitmap entries.
 * @param[in] mask Optional build mask. Only active rows can be selected for either join type.
 * @param[in] is_anti Whether to select unmarked rather than marked rows.
 * @param[in] stream Stream on which device work is ordered.
 * @param[in] mr Resource for device allocations, including temporary storage.
 * @return Exactly sized device vector of selected row indices.
 */
[[nodiscard]] rmm::device_uvector<cudf::size_type> select_marked_rows(
  uint8_t const* marks,
  cudf::size_type rows,
  std::optional<cudf::column_view> mask,
  bool is_anti,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace gqe::qep::mark_join_detail
