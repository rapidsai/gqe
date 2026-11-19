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
#include <string>
#include <vector>

namespace gqe {

class expression;

namespace storage {

class row_group_interface;

/**
 * @brief Reference-generating QEP reader for GQE in-memory row groups.
 *
 * The reference half of the reference/materialize split. Each successful call to `next()` mints
 * one portable `in_memory_reference` addressing one surviving row group, and `std::nullopt`
 * signals end-of-stream. No device data is touched; a downstream `in_memory_resolve_task` resolves
 * the references into `deferred_column` state.
 *
 * # Output Shape
 *
 * Chunks are emitted as `[reference]`, one per row group surviving conservative zone-map pruning.
 * References carry the table name, the row-group index, the retained row count, and the row
 * group's pruning result.
 *
 * # Count-Only Mode
 *
 * An empty column list enables count-only mode. The iterator emits exactly one `[row_count{N}]`
 * chunk, where `N` is the aggregate retained row count, without minting references.
 *
 * # Empty Input
 *
 * If pruning removes every row group, the iterator emits exactly one `[reference]` chunk holding
 * the portable empty-input reference before end-of-stream. The resolve task maps it to a
 * schema-carrying empty table, which preserves the output shape for iterate-task adapters.
 *
 * # Predicate Pruning
 *
 * The optional partial filter is used only for conservative zone-map pruning, evaluated once in
 * `initialize()`. Exact filtering remains the responsibility of a downstream filter task.
 *
 * # Lifetime
 *
 * Row-group pointers are non-owning and are read only during `initialize()`. Emitted references
 * carry no pointers; they address the table by name and row-group index, and are valid only
 * within the query execution that minted them.
 *
 * # Thread Safety
 *
 * The task object is a stateless functor after construction. Mutable cursor and pruning state
 * live in the iterator returned by `initialize()`. Concurrent calls to `next()` on the same
 * iterator must mint each reference exactly once.
 *
 * # Source Splitting
 *
 * Implements `qep::splittable_source_trait` with one work unit per row group.
 * Splits preserve source order and independently own their iteration state.
 * Row-group indexes retain their original table coordinates in every split.
 * Source row groups must outlive the task and every split created from it.
 */
class in_memory_reference_iterate_task final : public qep::iterate_task,
                                               public qep::input_residency_trait,
                                               public qep::splittable_source_trait {
 public:
  /**
   * @brief Construct an in-memory reference-generating read task.
   *
   * @param[in] table_name Catalog name resolving ranks use to reach the table.
   * @param[in] row_groups Non-owning pointers to the row groups to read.
   * @param[in] row_group_indexes Positional table index of each entry in `row_groups`. Must have
   * the same length as `row_groups`.
   * @param[in] column_indexes Source columns the resolve task emits, in output order. Empty enables
   * count-only mode.
   * @param[in] output_data_types Exact stored type for each requested column. Must have the same
   * length as `column_indexes`.
   * @param[in] source_metadata Scheduling metadata for the storage backing the row groups.
   * @param[in] partial_filter Optional logical predicate used for conservative zone-map pruning.
   */
  in_memory_reference_iterate_task(std::string table_name,
                                   std::vector<row_group_interface const*> row_groups,
                                   std::vector<std::size_t> row_group_indexes,
                                   std::vector<cudf::size_type> column_indexes,
                                   std::vector<cudf::data_type> output_data_types,
                                   qep::source_morsel_metadata source_metadata,
                                   std::unique_ptr<gqe::expression> partial_filter = nullptr);

  ~in_memory_reference_iterate_task() override;

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
  std::string _table_name;
  std::vector<row_group_interface const*> _row_groups;
  std::vector<std::size_t> _row_group_indexes;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _output_data_types;
  qep::source_morsel_metadata _source_metadata;
  std::unique_ptr<gqe::expression> _partial_filter;
};

}  // namespace storage
}  // namespace gqe
