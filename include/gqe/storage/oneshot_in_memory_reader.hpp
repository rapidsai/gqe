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
 * @brief One-shot QEP reader for GQE in-memory row groups.
 *
 * Each initialized iterator emits exactly one owned chunk containing all rows assigned to that
 * task, then signals end-of-stream. An empty split still emits a schema-carrying empty chunk.
 *
 * An empty column list enables count-only mode and emits `[row_count{N}]` without materializing
 * columns. Optional predicate pruning is conservative: it only removes row-group partitions that
 * the row groups' zone maps prove cannot match. Count-only mode is only supported if no predicate
 * is to be applied.
 *
 * Nullable contiguous fixed-width and string columns are supported. Projected compressed-sliced
 * columns containing null values are rejected during `initialize()`. Nullable columns that are
 * not projected, including all columns in count-only mode, are permitted.
 *
 * Nested LIST and STRUCT columns are not supported because `output_data_types` does not describe
 * their child schemas.
 *
 * Row-group references are non-owning. Their pointees must outlive this task and every task
 * returned by `split()` or `clone()`. Splitting copies only the
 * assigned row-group addresses into the returned task and does not modify the source task.
 *
 * Implements `qep::splittable_source_trait` so an executor can assign an exact half-open
 * row-group range as an independent streaming reader.
 */
class oneshot_in_memory_read_task final : public qep::iterate_task,
                                          public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct a one-shot in-memory read task.
   *
   * @param[in] row_groups Non-owning references to the row groups to read.
   * @param[in] column_indexes Columns to emit, in output order. Empty enables count-only mode.
   * @param[in] output_data_types Exact stored type for each requested column.
   * @param[in] partial_filter Optional logical predicate used for conservative zone-map pruning.
   *
   * @throws std::invalid_argument If a requested output type is LIST or STRUCT.
   */
  oneshot_in_memory_read_task(std::vector<row_group_interface const*> row_groups,
                              std::vector<cudf::size_type> column_indexes,
                              std::vector<cudf::data_type> output_data_types,
                              std::unique_ptr<gqe::expression> partial_filter = nullptr);

  ~oneshot_in_memory_read_task() override;

  /**
   * @copydoc gqe::qep::iterate_task::initialize
   *
   * @throws std::invalid_argument If a projected nullable column's materialization path does not
   * support null masks.
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
  std::vector<row_group_interface const*> _row_groups;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _output_data_types;
  std::unique_ptr<gqe::expression> _partial_filter;
};

}  // namespace storage
}  // namespace gqe
