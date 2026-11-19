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
#include <gqe/utility/error.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace gqe::qep {

/**
 * @brief Half-open range of stable source work-unit indices.
 *
 * An empty range has equal `begin` and `end` indices.
 */
struct source_work_range {
  std::size_t begin = 0;  ///< First included source work-unit index.

  std::size_t end = 0;  ///< First excluded source work-unit index.

  [[nodiscard]] bool operator==(source_work_range const&) const = default;
};

/**
 * @brief Divide a source work range into balanced, contiguous assignments.
 *
 * The first `range_size % count` assignments receive one extra work unit, where `range_size`
 * is `range.end - range.begin`. Empty assignments are valid when there are fewer work units
 * than assignments. Complexity is `O(1)`.
 *
 * @param[in] range Half-open range to divide. Must be ordered.
 * @param[in] index Assignment index. Must be less than `count`.
 * @param[in] count Positive number of assignments.
 * @return The half-open range for assignment `index`.
 * @throws std::invalid_argument If the range is reversed, `count` is zero, or `index` is invalid.
 */
[[nodiscard]] inline source_work_range balanced_source_work_range(source_work_range range,
                                                                  std::size_t index,
                                                                  std::size_t count)
{
  GQE_EXPECTS(
    range.begin <= range.end && count > 0 && index < count,
    "Source range must be ordered and assignment must satisfy index < count with count > 0",
    std::invalid_argument);
  auto const size      = range.end - range.begin;
  auto const base      = size / count;
  auto const remainder = size % count;
  auto const begin     = range.begin + base * index + std::min(index, remainder);
  return {begin, begin + base + (index < remainder ? 1 : 0)};
}

/**
 * @brief Construct independent readers for executor-selected source work ranges.
 *
 * # Design
 *
 * The source defines a stable, dense sequence of work units numbered `[0, work_unit_count())`.
 * Work units are source-specific, such as Parquet files or in-memory row groups. The executor
 * owns assignment policy and may use `balanced_source_work_range()` to calculate ranges.
 *
 * `split()` preserves the reader family and configuration. A one-shot source returns a one-shot
 * reader, a chunked source returns a chunked reader, and a reference-generating source returns a
 * reference-generating reader. Each returned task owns independent iteration state.
 * Non-overlapping ranges collectively covering the source read each work unit exactly once.
 *
 * Empty ranges are valid. Their readers emit one schema-carrying empty chunk, an empty reference,
 * or a zero row count, according to the source's output shape, before end-of-stream. Executors
 * may omit empty assignments when they support running downstream finalization without a reader.
 *
 * This interface describes storage input. Operator scans whose work depends on upstream state
 * use `static_iterate_split_trait` for static executor-specific execution units instead.
 */
class splittable_source_trait {
 public:
  virtual ~splittable_source_trait() = default;

  /**
   * @brief Return the number of source work units available for assignment.
   *
   * The count may be zero. Work-unit indices remain stable for the lifetime of this task.
   * Complexity is `O(1)`.
   *
   * @return Number of source work units.
   */
  [[nodiscard]] virtual std::size_t work_unit_count() const noexcept = 0;

  /**
   * @brief Construct an independent source reader for an explicit work range.
   *
   * Complexity is `O(K + B)`, where `K` is the number of selected work units and `B` is the total
   * size in bytes of source descriptors copied into the returned task.
   *
   * @param[in] range Half-open work range. Must satisfy
   *   `range.begin <= range.end <= work_unit_count()`.
   * @return An independent iterate task covering exactly `range`.
   *
   * @throws std::invalid_argument If the range is reversed or lies outside the source work
   * units.
   */
  [[nodiscard]] virtual std::unique_ptr<iterate_task> split(source_work_range range) const = 0;
};

}  // namespace gqe::qep
