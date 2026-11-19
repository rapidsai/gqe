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
#include <gqe/storage/hive_partitioning.hpp>

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief Streaming Parquet read task.
 *
 * Wraps `cudf::io::chunked_parquet_reader` as a `qep::iterate_task`. Each
 * call to `next` returns one chunk's worth of data, type-enforced to the
 * requested data types. Returns `std::nullopt` to signal end-of-stream.
 *
 * # Output Shape
 *
 * Chunks with projected columns are emitted as `[cudf_column, ...]`.
 *
 * # Count-Star Mode
 *
 * When `column_names` is empty, the task does not materialize any Parquet columns. Instead,
 * it reads the dataset row count from Parquet metadata and emits a single chunk shaped
 * `[row_count{N}]`, where `N` is the total row count. Downstream operators must read this
 * value from the `row_count` state slot rather than from `cudf::table::num_rows()`.
 *
 * # Empty Input
 *
 * An empty file list emits one schema-carrying zero-row chunk in projected-column mode or one
 * zero row-count chunk in count-only mode. Hive pruning that removes every file also emits one
 * schema-carrying zero-row chunk. No Parquet reader is constructed when no files remain to read.
 *
 * # Hive Partition Pruning
 *
 * Pass `hive_pruner_options` to enable pruning for Hive-partitioned datasets. The
 * partition column value is encoded in the directory name, e.g.
 * `lineitem/l_shipmode=1/part-0001.parquet`, and is absent from the on-disk Parquet schema.
 *
 * The reader parses each file path for `<column_name>=<value>`, reads the allowed values from
 * the predecessor input state, drops files whose parsed key is not present in that haystack,
 * omits the partition column from the Parquet read, and synthesizes the partition column per
 * emitted chunk.
 *
 * The haystack input must be a single integral column. Files that do not match the Hive path
 * pattern are rejected in strict mode; Hive default-partition entries are treated as null and
 * dropped.
 *
 * # Source Splitting
 *
 * Implements `qep::splittable_source_trait` with one work unit per file.
 * Splits preserve source order and independently own their iteration state.
 */
class chunked_parquet_read_task : public qep::iterate_task,
                                  public qep::input_residency_trait,
                                  public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct a Parquet read task.
   *
   * @param[in] file_paths Files to read.
   * @param[in] column_names Columns to load, in output order. Empty enables count-star mode.
   * @param[in] output_data_types Expected data type per column. A column
   *   whose on-disk type differs is cast to the expected type. Must have
   *   the same length as `column_names`.
   * @param[in] pruner When set, enables Hive-partition pruning. The
   *   predecessor task's output supplies the in-list haystack (a single
   *   integral-typed column).
   */
  chunked_parquet_read_task(std::vector<std::string> file_paths,
                            std::vector<std::string> column_names,
                            std::vector<cudf::data_type> output_data_types,
                            std::optional<hive_pruner_options> pruner = std::nullopt);

  [[nodiscard]] qep::source_morsel_metadata input_residency() const noexcept override
  {
    return {.residency = qep::data_residency::host};
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
   * @copydoc gqe::qep::iterate_task::finalize
   */
  void finalize(
    qep::state_container&& iterator,
    context_reference ctx_ref,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<qep::task> clone() const override;

 private:
  std::vector<std::string> _file_paths;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _output_data_types;
  std::optional<hive_pruner_options> _pruner;
};

}  // namespace storage
}  // namespace gqe
