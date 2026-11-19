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

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

class parquet_footer_cache;

/**
 * @brief Reference-generating QEP reader for Parquet files.
 *
 * The reference half of the reference/materialize split for Parquet sources. Each successful call
 * to `next()` mints one portable `parquet_reference` addressing a contiguous row-group span, and
 * `std::nullopt` signals end-of-stream. No column data is read; a downstream `parquet_resolve_task`
 * resolves the references into `deferred_column` state.
 *
 * # Footer Cache
 *
 * `initialize()` parses each file's footer exactly once through the injected
 * `parquet_footer_cache`, which is shared with the resolve task so both construct hybrid-scan
 * readers from the same parsed metadata. The cache handle must not be null.
 *
 * # Output Shape
 *
 * Chunks are emitted as `[reference]`. Spans are the row-group passes the hybrid-scan reader
 * derives from `parquet_pass_read_size`, so each reference is memory-bounded. References carry
 * the file path, the span, the exact row count, and the
 * file's Hive partition key when pruning is configured.
 *
 * # Count-Star Mode
 *
 * An empty column list enables count-star mode. The iterator emits exactly one `[row_count{N}]`
 * chunk, where `N` is the total row count from footer metadata, without minting references.
 * Count-star mode does not support Hive pruning.
 *
 * # Empty Input
 *
 * If the file list is empty or Hive pruning removes every file, the iterator emits exactly one
 * `[reference]` chunk holding the portable empty-input reference before end-of-stream. The resolve
 * task maps it to a schema-carrying empty table.
 *
 * # Hive Partition Pruning
 *
 * With `hive_pruner_options`, `initialize()` consumes the haystack input, drops files whose
 * parsed key is absent, and resolves each surviving file's partition key at mint time. Because
 * every reference carries its own key, chunks may be consumed in any order — there is no
 * in-order chunk cursor. Files that do not match the Hive path pattern are rejected in strict
 * mode; Hive default-partition entries are dropped.
 *
 * # Thread Safety
 *
 * The task object is a stateless functor after construction. Mutable cursor state lives in the
 * iterator returned by `initialize()`. Concurrent calls to `next()` on the same iterator must
 * mint each reference exactly once.
 *
 * # Source Splitting
 *
 * Implements `qep::splittable_source_trait` with one work unit per file.
 * Splits preserve source order and independently own their iteration state.
 * Splits share the footer cache and retain each reference's file coordinates.
 */
class parquet_reference_iterate_task final : public qep::iterate_task,
                                             public qep::input_residency_trait,
                                             public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct a Parquet reference-generating read task.
   *
   * @param[in] file_paths Files to read.
   * @param[in] column_names Columns the resolve task loads, in output order. Empty enables
   * count-star mode.
   * @param[in] output_data_types Expected data type per column. Must have the same length as
   * `column_names`.
   * @param[in] pruner When set, enables Hive-partition pruning. The predecessor task's output
   * supplies the in-list haystack. Not supported in count-star mode.
   * @param[in] footer_cache Shared footer cache populated during reference generation. Must not
   * be null.
   */
  parquet_reference_iterate_task(std::vector<std::string> file_paths,
                                 std::vector<std::string> column_names,
                                 std::vector<cudf::data_type> output_data_types,
                                 std::optional<hive_pruner_options> pruner,
                                 std::shared_ptr<parquet_footer_cache> footer_cache);

  ~parquet_reference_iterate_task() override;

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
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::iterate_task::next
   */
  [[nodiscard]] std::optional<qep::state_container> next(
    qep::state_container_view iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
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
  std::shared_ptr<parquet_footer_cache> _footer_cache;
};

}  // namespace storage
}  // namespace gqe
