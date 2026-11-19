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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe::qep {

/**
 * @brief Incrementally select and order a single-key top-K result.
 *
 * Non-trivial ordering expressions must be evaluated by an upstream `project_task`; this task
 * references the resulting key by column index.
 *
 * # Candidate materialization
 *
 * `next()` eagerly materializes at most `offset + count` candidates from each input chunk. This
 * keeps candidate payloads small for the expected top-K requests and allows candidate gathering to
 * overlap with upstream input processing. Deferring materialization until `finalize()` could reduce
 * copying for large K, but would require splitting candidate selection into a separate stateful
 * transform task. TPC-H uses K values no greater than 100, so the current design favors early
 * materialization.
 *
 * # Input shape
 *
 * The task accepts every column-bearing `logical_table_view` shape. `next()` fully compacts each
 * chunk before candidate selection, so relation-mask entries that are false or null do not
 * participate.
 *
 * # Output shape
 *
 * `[indirect_column, ...]`. Every indirect column shares a gather source whose position list is in
 * final query order and already has `offset` removed. A downstream `gather_task` materializes the
 * result.
 */
class top_k_task : public fold_task {
 public:
  /**
   * @param[in] key_index Input column used as the ordering key.
   * @param[in] column_order Ascending or descending ordering.
   * @param[in] null_precedence Placement of null keys in the final ordering.
   * @param[in] offset Number of leading ordered rows to omit.
   * @param[in] count Maximum number of rows to emit after `offset`.
   * @param[in] output_data_types Input/output schema, used for an input with no chunks.
   */
  top_k_task(cudf::size_type key_index,
             cudf::order column_order,
             cudf::null_order null_precedence,
             cudf::size_type offset,
             cudf::size_type count,
             std::vector<cudf::data_type> output_data_types);

  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  cudf::size_type _key_index;
  cudf::order _column_order;
  cudf::null_order _null_precedence;
  cudf::size_type _offset;
  cudf::size_type _count;
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace gqe::qep
