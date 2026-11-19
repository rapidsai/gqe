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

#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief Streaming Parquet write task.
 *
 * Wraps `cudf::io::chunked_parquet_writer` as a `qep::fold_task`. The writer is constructed
 * in `initialize`. Each `next` call
 * streams one chunk to the intended on-disk file; `finalize` closes the writer. If no chunks arrive
 * via next() call, `finalize` writes an empty Parquet file.
 *
 * # Input Shape
 *
 * Plain, non-nested table (`[(cudf_column | cudf_column_view)+]`). Each chunk's column
 * types must match the constructor-supplied `data_types`; the schema is validated on every
 * chunk and `next` throws on mismatch.
 *
 * # Output Shape
 *
 * `finalize()` returns `[write_statistics{statistics}]` carrying the total rows written, physical
 * Parquet row-group count, and output schema. Per-column compression statistics remain zero because
 * the current Parquet writer integration does not expose them. This is a terminal shape — no
 * downstream task consumes a write task's output.
 *
 * # Stream coordination
 *
 * cudf::io::chunked_parquet_writer captures the CUDA stream in its constructor. However,
 * chunked_parquet_write_task contract allows concurrent writes on multiple streams. To bridge the
 * gap, streams are coordinated as follows. Writes are scheduled on an internal, dedicated stream
 * for the cudf::io::chunked_parquet_writer to use. Due to the WAR dependency, each call to next()
 * records a CUDA event on its input stream and enqueues a cudaStreamWaitEvent on the write stream.
 * Due to a (potential) RAW dependency, an event wait is inserted in the reverse direction after
 * enqueuing the Parquet write. Thereby, Parquet writes are ensured to have the correct stream order
 * in regards to prior operations on the input stream and operations that may occur after the write.
 *
 * # Thread Safety
 *
 * The QEP contract permits concurrent `next` calls on the same accumulator (different
 * streams). Concurrent calls are serialized on a `std::mutex` held in the task-private
 * state — `cudf::io::chunked_parquet_writer::write()` is not documented as thread-safe
 * and mutates persistent internal state on every call.
 *
 * # Limitations (v1)
 *
 * - Single output file path. Sharded multi-file outputs are not supported; a sharding QEP
 *   operator would split the input stream upstream.
 * - Requires non-replicated fold state: one logical fold accumulator writes one side-effecting
 *   output file.
 * - Hive-partitioned output not supported.
 * - Nested LIST and STRUCT columns are not supported because `data_types` does not describe their
 *   child schemas.
 * - Compression and other parquet writer options are left at cuDF defaults.
 */
class chunked_parquet_write_task : public qep::fold_task {
 public:
  /**
   * @brief Construct a chunked Parquet write task.
   *
   * @param[in] file_path Output Parquet file path.
   * @param[in] column_names Output column names, in chunk-column order. Must have the
   *   same length as `data_types`.
   * @param[in] data_types Expected data type per column. Each input chunk's column types
   *   must match exactly; mismatches throw at `next` time.
   */
  chunked_parquet_write_task(std::string file_path,
                             std::vector<std::string> column_names,
                             std::vector<cudf::data_type> data_types);

  /**
   * @copydoc gqe::qep::fold_task::initialize
   */
  [[nodiscard]] qep::state_container initialize(
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::next
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
  std::string _file_path;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
};

}  // namespace storage
}  // namespace gqe
