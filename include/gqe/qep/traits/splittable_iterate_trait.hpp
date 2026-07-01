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

namespace gqe {
namespace qep {

/**
 * @brief Trait interface that turns a streaming `qep::iterate_task` into one-shot execution
 *        units for the static task graph.
 *
 * # Purpose
 *
 * Bridges streaming `qep::iterate_task` (`next()` may be called many times) to the static
 * task-graph executor (each adapter executes once). `split(idx, total)` returns an iterator
 * whose entire output is one slice — exactly one chunk, which may be empty (zero rows) but
 * still carries the output schema. The QEP task-graph builder probes for this interface via
 * `dynamic_cast` and instantiates N parallel adapters, one per split. Iterate tasks that don't
 * implement this interface run as a single execution unit.
 *
 * # Split granularity
 *
 * The split is over *work*, not over data keys or ranges — each split reads a non-overlapping
 * portion of the underlying source, but the choice of portion is implementation-defined and
 * may not correspond to any data property.
 */
class splittable_iterate_trait {
 public:
  virtual ~splittable_iterate_trait() = default;

  /**
   * @brief Maximum number of splits this iterator can be divided into.
   *
   * The builder caps its instantiated adapter count by this value (and by the query
   * configuration). Must return at least 1.
   *
   * @return Maximum useful split count.
   */
  [[nodiscard]] virtual cudf::size_type max_splits() const noexcept = 0;

  /**
   * @brief Construct a new iterator that reads one split of this iterator's work.
   *
   * The returned iterator yields exactly one chunk: the rows belonging to `split_idx`. The
   * chunk may be empty (zero rows) when the split contains no rows, but it must still carry the
   * output schema — the iterator must not yield zero chunks (immediate `std::nullopt`), because
   * the adapter has no other source for the schema. Implementations must produce
   * non-overlapping, exhaustive splits when called with `split_idx` ranging over
   * `[0, total_splits)`.
   *
   * @param[in] split_idx Index of the split to read, in `[0, total_splits)`.
   * @param[in] total_splits Total number of splits to divide the work into. Must be `>= 1`
   *   and `<= max_splits()`.
   *
   * @return A new iterator task that reads only the requested split.
   */
  [[nodiscard]] virtual std::unique_ptr<iterate_task> split(cudf::size_type split_idx,
                                                            cudf::size_type total_splits) const = 0;
};

}  // namespace qep
}  // namespace gqe
