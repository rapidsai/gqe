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
#include <gqe/storage/hive_partitioning.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

class parquet_footer_cache;
class parquet_reference;

/**
 * @brief Resolve task binding Parquet references into deferred tables.
 *
 * The resolve half of the reference/materialize split for Parquet sources. Each `next()` consumes
 * one `[reference]` chunk minted by `parquet_reference_iterate_task` and emits one
 * `deferred_column` state per projected column, all sharing one deferred source. Consumers
 * materialize the columns they need via `qep::materialize_deferred_columns` or the source's
 * `materialize_columns`.
 *
 * # Footer Cache
 *
 * `next()` resolves each reference's parsed footer through a `parquet_footer_cache` shared with
 * the reference-generating iterate task. A cache miss loads the footer from disk.
 *
 * # Validation
 *
 * A reference is rejected if its row-group span is outside the file, if its row count differs
 * from the footer's count for that span, or if a projected file column is not in the file schema.
 * Stored types are enforced by cast at materialization, as in every Parquet read path.
 *
 * # Input Shapes
 *
 * - A `[reference]` chunk holding a `parquet_reference` produces a `[deferred_column...]` chunk
 *   with one slot per projected column.
 * - The empty-input reference produces a schema-correct empty table directly.
 * - A count-only `[row_count]` chunk passes through unchanged. A task constructed with an empty
 *   projection accepts only such chunks and rejects references.
 *
 * # Hive Partition Synthesis
 *
 * With `hive_pruner_options`, the reference's mint-time partition key is synthesized into the
 * output at the configured column position during materialization. References without a key are
 * rejected. When every requested column is the partition column, materialization synthesizes the
 * column without reading file bytes.
 *
 * # Materialization
 *
 * Materializing a deferred column reads file bytes and blocks the calling thread on that I/O.
 *
 * # Thread Safety
 *
 * `next()` is externally thread-safe. Each call operates only on its own input chunk and the
 * cache's thread-safe lookup.
 */
class parquet_resolve_task final : public qep::stateless_transform_task {
 public:
  /**
   * @brief Construct a Parquet resolve task.
   *
   * @param[in] footer_cache Shared footer cache used to resolve references in `next()`. Must
   * not be null.
   * @param[in] column_names Columns to load, in output order. Empty for count-only plans, whose
   * chunks pass through.
   * @param[in] output_data_types Expected data type per column. Must have the same length as
   * `column_names`.
   * @param[in] pruner When set, enables Hive partition-key synthesis at the configured column.
   */
  parquet_resolve_task(std::shared_ptr<parquet_footer_cache> footer_cache,
                       std::vector<std::string> column_names,
                       std::vector<cudf::data_type> output_data_types,
                       std::optional<hive_pruner_options> pruner = std::nullopt);

  /**
   * @copydoc gqe::qep::stateless_transform_task::next
   */
  [[nodiscard]] std::optional<qep::state_container> next(
    qep::state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<qep::task> clone() const override;

 private:
  /** @brief Resolve and validate one row-group-span reference into a deferred-column container. */
  [[nodiscard]] qep::state_container resolve(std::shared_ptr<parquet_reference const> reference,
                                             context_reference ctx_ref) const;

  std::shared_ptr<parquet_footer_cache> _footer_cache;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _output_data_types;
  std::optional<hive_pruner_options> _pruner;
};

}  // namespace storage
}  // namespace gqe
