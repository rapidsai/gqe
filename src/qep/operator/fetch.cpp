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

#include <gqe/qep/operator/fetch.hpp>

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gqe::qep {
namespace {

/**
 * @brief Per-execution state for a fetch, holding the number of stream rows fetched so far.
 *
 * The counter is a position in the input stream, not a fill level of the result. It also advances
 * for rows outside the window. Concurrent `next()` calls partition the stream into disjoint row
 * ranges by fetching from this counter. A 64-bit counter is required because the stream's total
 * row count may exceed `cudf::size_type` even though every individual chunk fits.
 */
struct fetch_accumulator : task_private_state {
  std::atomic<int64_t> fetched_rows{0};
};

/** @brief The window intersection of one consumed chunk, in chunk-local row coordinates. */
struct local_window {
  int64_t begin;
  int64_t end;

  [[nodiscard]] bool covers(int64_t rows) const noexcept { return begin == 0 && end == rows; }
};

}  // namespace

fetch_task::fetch_task(int64_t offset, int64_t count)
  : stateful_transform_task(task_ordering::preserve), _offset{offset}, _count{count}
{
  GQE_EXPECTS(_offset >= 0, "fetch_task: offset must be non-negative");
  GQE_EXPECTS(_count >= 0, "fetch_task: count must be non-negative");
  GQE_EXPECTS(_offset <= std::numeric_limits<int64_t>::max() - _count,
              "fetch_task: offset + count overflows int64_t",
              std::overflow_error);
}

std::unique_ptr<task> fetch_task::clone() const
{
  return std::make_unique<fetch_task>(_offset, _count);
}

state_container fetch_task::initialize(state_container_view inputs,
                                       context_reference,
                                       query_stream const&,
                                       rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::fetch_task::initialize"};
  GQE_EXPECTS(inputs.empty(), "fetch_task::initialize: fetch has no materialized inputs");
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<fetch_accumulator>()})
    .build();
}

std::optional<state_container> fetch_task::next(state_container_view inputs,
                                                state_container_view accumulator,
                                                context_reference,
                                                query_stream const& stream_handle,
                                                rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::fetch_task::next"};
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(!inputs.empty(), "fetch_task::next: inputs must not be empty");

  auto& counter    = get_task_private_state<fetch_accumulator>(accumulator, 0);
  auto const limit = _offset + _count;

  // Consume this chunk's stream range `[start, start + rows)` and intersect it with the window.
  // The order in which concurrent calls consume the stream is unspecified and only partitions the
  // window, so relaxed ordering suffices. A range past the window is the early exit. The
  // `start > 0` exception keeps the very first range of an empty window emitting a schema-correct
  // zero-row output.
  auto const intersect_fetch_window = [&](int64_t rows) -> std::optional<local_window> {
    auto const start = counter.fetched_rows.fetch_add(rows, std::memory_order_relaxed);
    if (start >= limit && start > 0) { return std::nullopt; }
    return local_window{std::clamp<int64_t>(_offset - start, 0, rows),
                        std::clamp<int64_t>(limit - start, 0, rows)};
  };

  if (auto const count_only = try_row_count(inputs); count_only.has_value()) {
    auto const window = intersect_fetch_window(*count_only);
    if (!window.has_value()) { return std::nullopt; }
    return make_row_count_container(static_cast<cudf::size_type>(window->end - window->begin));
  }

  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "fetch_task::next: input is not a logical relation or count-only container");
  column_materialization materialized =
    materialize_all_compacted_columns(*logical, stream_handle, mr);
  auto const table  = materialized.table_view();
  auto const rows   = static_cast<int64_t>(materialized.row_count());
  auto const window = intersect_fetch_window(rows);
  if (!window.has_value()) {
    return std::nullopt;
  } else if (window->covers(rows)) {
    return std::move(materialized).release_column_states();
  } else if (window->begin == window->end) {
    return state_container_builder()
      .add_state(std::move(*cudf::empty_like(table)), stream_handle)
      .build();
  } else {
    auto const sliced = cudf::slice(table,
                                    {static_cast<cudf::size_type>(window->begin),
                                     static_cast<cudf::size_type>(window->end)},
                                    stream)
                          .front();
    auto output =
      state_container_builder().add_state(cudf::table{sliced, stream, mr}, stream_handle).build();
    order_deallocation_after_stream_use(inputs, stream_handle);
    return output;
  }
}

}  // namespace gqe::qep
