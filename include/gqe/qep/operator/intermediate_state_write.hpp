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

namespace gqe {
namespace qep {

/**
 * @brief Materialize a stream of regular-table or count-only chunks.
 *
 * Input chunks have the following cases:
 *
 * - A regular-table chunk is a non-empty sequence of owned or borrowed cuDF columns.
 * - A count-only chunk contains one `row_count` slot.
 * - An empty container represents absent upstream output and is ignored.
 *
 * All chunks accumulated by one task invocation must have the same shape. Regular-table chunks
 * must also have the same column count and nested column types. Masked and indirect tables must
 * first pass through `gather_task`.
 *
 * `next` retains regular chunks or updates a checked host-side row count without launching CUDA
 * work. Concurrent `next` calls are supported, but the relative order of their chunks in the
 * finalized table is unspecified. The writer preserves an ordered result when normalization
 * supplies at most one present chunk. `finalize` handles the accumulated input as follows:
 *
 * - One regular chunk is returned directly.
 * - Multiple regular chunks are concatenated into newly allocated output using the supplied stream
 *   and memory resource.
 * - Count-only chunks produce one state containing their accumulated row count.
 * - If every input is absent, the output types supplied at construction produce a schema-correct
 *   empty result. An empty output-type list denotes count-only output.
 *
 * For multiple regular chunks, the concatenated result does not share ownership with its inputs.
 * The moved accumulator therefore remains the `shared_state` owner of those inputs until the caller
 * releases it. The caller must keep the accumulator alive until work on the supplied stream has
 * completed. The executor does this through its normal result handoff.
 */
class intermediate_state_write_task final : public fold_task {
 public:
  /**
   * @brief Construct an intermediate writer for the specified output types.
   *
   * @param[in] output_data_types Output column types in emission order. An empty list denotes
   *   count-only output.
   */
  explicit intermediate_state_write_task(std::vector<cudf::data_type> output_data_types);

  /** @copydoc fold_task::initialize */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc fold_task::next */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc fold_task::finalize */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  std::vector<cudf::data_type> _output_data_types;
};

}  // namespace qep
}  // namespace gqe
