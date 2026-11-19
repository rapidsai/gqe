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
#include <gqe/qep/traits/splittable_source_trait.hpp>
#include <gqe/storage/hive_partitioning.hpp>

#include <cudf/types.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief One-shot Parquet read task for the static task graph.
 *
 * Reads all assigned files in a single `next()` call. Subsequent `next()` calls return
 * `std::nullopt`.
 *
 * # Output shape
 *
 * Column-bearing reads emit `[cudf_column, ...]`. Count-only reads (`column_names` empty)
 * emit `[row_count{N}]`, with `N` recovered from Parquet metadata.
 *
 * # Hive partitioning
 *
 * Pass `hive_pruner_options` to read a single integral haystack column from `initialize()`'s
 * input, prune files whose Hive key is absent from the haystack, and synthesize the partition
 * column after reading the surviving files.
 *
 * # Executor Source Splitting
 *
 * Implements `qep::splittable_source_trait` so an executor can assign an exact half-open file
 * range as an independent streaming reader.
 */
class oneshot_parquet_read_task : public qep::iterate_task, public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct a one-shot Parquet read task.
   *
   * @param[in] file_paths Files to read.
   * @param[in] column_names Columns to load, in output order. Empty enables count-only mode.
   * @param[in] output_data_types Expected output type per column. Must have the same length as
   *   `column_names`.
   * @param[in] pruner Optional Hive-partition pruning configuration.
   */
  oneshot_parquet_read_task(std::vector<std::string> file_paths,
                            std::vector<std::string> column_names,
                            std::vector<cudf::data_type> output_data_types,
                            std::optional<hive_pruner_options> pruner = std::nullopt);

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

  /** @copydoc gqe::qep::splittable_source_trait::work_unit_count */
  [[nodiscard]] std::size_t work_unit_count() const noexcept override;

  /** @copydoc gqe::qep::splittable_source_trait::split */
  [[nodiscard]] std::unique_ptr<qep::iterate_task> split(
    qep::source_work_range range) const override;

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
