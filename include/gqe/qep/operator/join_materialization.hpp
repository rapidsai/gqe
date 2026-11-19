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

namespace gqe::qep {

/**
 * @file
 * @brief Selective gathering of late-materialized join outputs.
 *
 * # Design
 *
 * A late-materialized join output references its inputs through gather sources. Each source pins
 * its whole backing relation until every consumer releases the output, even when only a few rows
 * are selected. A gather source is selective when its position-list length is strictly less than
 * `threshold` times its backing relation's row count. Duplicate positions count as output rows, and
 * a composed position list is compared against the original backing relation, so the decision is
 * cumulative across joins. Every projected column of a selective source is gathered in one gather
 * per source, which releases this output's reference to the backing relation. Other states keep
 * shared ownership. An empty selection always qualifies, even for an empty backing relation.
 *
 * Gathering every output would repeat the cost that late materialization avoids for dense outputs.
 * Gathering none leaves a small output pinning a large source until every consumer finishes, which
 * can hold tens of GiB for a few thousand rows.
 *
 * # Contracts
 *
 * - The input is either a count-only state or a logical relation with at least one column and no
 *   valid mask. Zero-row relations are accepted. A violation throws `std::logic_error`.
 * - Count-only states and direct columns pass through unchanged, and so does every column when
 *   `threshold` is zero. The output has no valid mask, and each column keeps its logical position.
 * - Apply it after residual join filtering and before a shared producer publishes its result to
 *   multiple consumers.
 * - Other owners can keep a backing relation alive, so a selective source does not guarantee that
 *   its memory is reclaimed. In particular, the build side of a broadcast join is shared by every
 *   probe partition while the build state lives, and each partition decides against the whole build
 *   relation. Several partitions that each select just under the threshold can together gather
 *   more than the threshold fraction of the build's projected columns.
 * - Variable-width columns can have a byte ratio that differs from their row ratio.
 * - The decision reads host metadata only. Gathering uses `stream_handle` and `mr`, can synchronize
 *   inside cuDF for variable-width columns as `gather_task` does, and orders the release of source
 *   storage after its reads on `stream_handle`.
 * - Concurrent calls are safe for immutable inputs and a thread-safe memory resource.
 */

/**
 * @brief Gather the columns of selective gather sources in a completed join output.
 *
 * See the file documentation for the selection rule and contracts.
 *
 * @param[in] inputs Completed join output with owning or externally owned source states.
 * @param[in] threshold Selection ratio in [0, 1]. Zero disables gathering.
 * @param[in] stream_handle CUDA stream lifetime capability for all device work.
 * @param[in] mr Memory resource for all device allocations.
 * @return Output states in their original projection order.
 * @throws std::invalid_argument If `threshold` is not finite or outside [0, 1].
 * @throws std::logic_error If the input has no columns, is not a logical relation, or carries a
 *   valid mask.
 */
[[nodiscard]] state_container materialize_selective_join_output(state_container_view inputs,
                                                                double threshold,
                                                                query_stream const& stream_handle,
                                                                rmm::device_async_resource_ref mr);

/**
 * @brief Apply `materialize_selective_join_output` to every partition of a completed join.
 *
 * Lowering inserts this transform after a late-materialized broadcast join's residual filtering
 * and before caching the relation's terminal task, so every consumer of the relation shares one
 * result per partition. The output can mix direct and indirect columns, so this task does not
 * establish an ordinary-table boundary.
 */
class join_materialization_task : public stateless_transform_task {
 public:
  /**
   * @brief Construct the task.
   *
   * @param[in] threshold Selection ratio in [0, 1]. Zero disables gathering.
   * @throws std::invalid_argument If `threshold` is not finite or outside [0, 1].
   */
  explicit join_materialization_task(double threshold);

  /** @copydoc stateless_transform_task::next */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /** @copydoc task::clone */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

  /** @brief Selection ratio below which a gather source is gathered. */
  [[nodiscard]] double threshold() const noexcept { return _threshold; }

 private:
  double _threshold;
};

}  // namespace gqe::qep
