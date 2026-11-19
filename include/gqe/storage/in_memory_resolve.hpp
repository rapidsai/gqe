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
#include <optional>
#include <vector>

namespace gqe {
namespace storage {

class in_memory_reference;
class table_provider;

/**
 * @brief Resolve task binding in-memory references into deferred tables.
 *
 * The resolve half of the reference/materialize split. Each `next()` consumes one `[reference]`
 * chunk minted by `in_memory_reference_iterate_task`, resolves the referenced row group through a
 * `storage::table_provider`, and emits one `deferred_column` state per projected column, all
 * sharing one deferred source. Consumers materialize the columns they need via
 * `qep::materialize_deferred_columns` or the source's `materialize_columns` (see
 * `gqe/qep/shapes/deferred_column.hpp`).
 *
 * # Input Shapes
 *
 * - A `[reference]` chunk holding an `in_memory_reference` produces a `[deferred_column...]`
 *   chunk with one slot per projected column.
 * - The empty-input reference produces a schema-correct empty table directly.
 * - A count-only `[row_count]` chunk passes through unchanged, so count-only plans tolerate an
 *   inserted resolve task. A task constructed with an empty projection accepts only such chunks
 *   and rejects references.
 *
 * # Validation
 *
 * A reference is rejected if its row-group index is outside the table, if any projected column
 * index is outside the resolved row-group schema, if a projected column's stored type differs from
 * its output type, if a projected column contains nulls the materialization does not support, or
 * if its pruning result selects no partitions or extends past the row group's size.
 *
 * # Lifetime
 *
 * The table provider is non-owning and must outlive this task and every emitted state. Every
 * emitted deferred column must be materialized or dropped before the query completes. Like every
 * GQE read path, this task relies on the engine-wide contract that table contents are not replaced
 * or cleared while a query reads them, and the underlying storage must not be explicitly
 * deallocated while emitted state is alive.
 *
 * # Thread Safety
 *
 * `next()` is externally thread-safe. Each call operates only on its own input chunk and the
 * provider's thread-safe-for-readers lookup.
 */
class in_memory_resolve_task final : public qep::stateless_transform_task {
 public:
  /**
   * @brief Construct an in-memory resolve task.
   *
   * @param[in] table_provider Non-owning provider used to resolve referenced tables by name.
   * @param[in] column_indexes Source columns to emit, in output order. Empty for count-only
   * plans, whose chunks pass through.
   * @param[in] output_data_types Exact stored type for each requested column. Must have the same
   * length as `column_indexes`.
   */
  in_memory_resolve_task(table_provider const* table_provider,
                         std::vector<cudf::size_type> column_indexes,
                         std::vector<cudf::data_type> output_data_types);

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
  /** @brief Resolve and validate one row-group reference into a deferred-column container. */
  [[nodiscard]] qep::state_container resolve(in_memory_reference const& ref,
                                             context_reference ctx_ref) const;

  table_provider const* _table_provider;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace storage
}  // namespace gqe
