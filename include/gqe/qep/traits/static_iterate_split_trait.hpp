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

#include <cstddef>
#include <memory>
#include <optional>

namespace gqe {
namespace qep {

/**
 * @brief Trait interface that turns a streaming QEP task into one-shot execution units for
 *        the static task graph.
 *
 * # Purpose
 *
 * Storage readers use `splittable_source_trait` and explicit source work ranges instead.
 *
 * Bridges a streaming QEP task (its methods may be called many times) to the static
 * task-graph executor (each adapter executes once). `split(idx, total)` returns a task that
 * processes one slice of the work. The QEP task-graph builder probes for this interface via
 * `dynamic_cast` and instantiates N parallel adapters, one per split. Tasks that don't
 * implement this interface run as a single execution unit unless they are storage sources.
 *
 * # Split granularity
 *
 * The split is over *work*, not over data keys or ranges — each split processes a
 * non-overlapping portion of the underlying work, but the choice of portion is
 * implementation-defined and may not correspond to any data property.
 *
 * # Split count selection
 *
 * `required_split_count_from_fold_input()` selects one of two static-executor contracts.
 *
 * - If it returns `std::nullopt`, the builder chooses the split count using the query
 *   configuration and `max_splits()`. Every count in `[1, max_splits()]` must be supported.
 * - Otherwise, the returned index identifies a fold task in this task's ordered predecessors.
 *   The builder must use that fold's input-adapter count as the split count, without applying
 *   the query partition cap again. `max_splits()` validates this required count rather than
 *   selecting or truncating it. Other counts need not be supported.
 *
 * `max_splits()` must return at least one in either mode. The builder validates the required
 * predecessor index, that the predecessor is a fold, and that its input-adapter count is in
 * `[1, max_splits()]`. For a supported count, the splits must be non-overlapping and exhaustive.
 *
 * This relationship is specific to the static executor. Each fold input adapter calls the
 * fold's `next()` at most once, so its adapter count bounds retained chunks when the fold
 * retains at most one chunk per call. Early exits may leave fewer chunks than adapters. It is
 * not a relationship between worker count and retained chunks in a streaming executor.
 *
 * @tparam TaskType The abstract QEP task kind that splitting produces.
 */
template <typename TaskType>
class static_split_trait {
 public:
  virtual ~static_split_trait() = default;

  /**
   * @brief Identify the fold predecessor that determines the required static split count.
   *
   * @return An index into the task's ordered predecessors, or `std::nullopt` to select the
   * count using the query configuration and `max_splits()`.
   */
  [[nodiscard]] virtual std::optional<std::size_t> required_split_count_from_fold_input()
    const noexcept
  {
    return std::nullopt;
  }

  /**
   * @brief Upper bound on the static split count.
   *
   * This caps a configuration-selected count or validates a required fold-input count,
   * according to the split count selection contract above.
   *
   * @return A positive upper bound on the supported split count.
   */
  [[nodiscard]] virtual cudf::size_type max_splits() const noexcept = 0;

  /**
   * @brief Construct a new task that processes one split of this task's work.
   *
   * For a supported `total_splits`, implementations must produce non-overlapping, exhaustive
   * splits when called with `split_idx` ranging over `[0, total_splits)`.
   *
   * @pre `0 <= split_idx < total_splits` and `1 <= total_splits <= max_splits()`.
   * Implementations validate these bounds when constructing a split.
   * @pre If `required_split_count_from_fold_input()` returns a predecessor index,
   * `total_splits` equals that fold's input-adapter count. The static builder selects and
   * validates this count. Other callers must supply the same count. Data-dependent violations
   * may be detected only when the returned task is initialized.
   *
   * @param[in] split_idx Index of the split to process, in `[0, total_splits)`.
   * @param[in] total_splits A supported count under the split count selection contract above.
   *
   * @return A new task that processes only the requested split.
   */
  [[nodiscard]] virtual std::unique_ptr<TaskType> split(cudf::size_type split_idx,
                                                        cudf::size_type total_splits) const = 0;
};

/**
 * @brief Static execution-unit splitting for `qep::iterate_task`s.
 *
 * Each split processes a non-overlapping portion of the operator state and yields exactly one
 * chunk: the rows belonging to its split index. The chunk may be empty (zero rows) when the
 * split contains no rows, but it must still carry the output schema — the iterator must not
 * yield zero chunks (immediate `std::nullopt`), because the adapter has no other source for
 * the schema.
 */
using static_iterate_split_trait = static_split_trait<iterate_task>;

}  // namespace qep
}  // namespace gqe
