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

#include <gqe/qep/operator/intermediate_state_write.hpp>

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {
namespace {

enum class input_mode { unset, regular, count_only };

/**
 * @brief Accumulated inputs for one intermediate-state writer invocation.
 *
 * `mutex` protects `mode`, `chunks`, `row_count`, and the transitions that update them. `next` may
 * access this state concurrently. The executor calls `finalize` only after every `next` call has
 * completed, so `finalize` accesses the fields without acquiring the mutex.
 */
struct intermediate_state_write_accumulator final : task_private_state {
  std::mutex mutex;                     ///< Guards all fields below.
  input_mode mode = input_mode::unset;  ///< Shape established by the first present input.
  std::vector<state_container> chunks;  ///< Retained regular-table chunks.
  std::int64_t row_count = 0;           ///< Checked sum of count-only chunks.
};

[[nodiscard]] intermediate_state_write_accumulator& get_accumulator(
  state_container_view accumulator)
{
  GQE_EXPECTS(accumulator.size() == 1,
              "intermediate_state_write_task: accumulator must contain one state slot");
  GQE_EXPECTS(accumulator.front() != nullptr,
              "intermediate_state_write_task: accumulator contains a null state slot");
  auto* private_state = std::get_if<state_kind::task_private>(accumulator.front().get());
  GQE_EXPECTS(private_state != nullptr,
              "intermediate_state_write_task: accumulator is not task-private state");
  GQE_EXPECTS(private_state->data != nullptr,
              "intermediate_state_write_task: accumulator task-private state is null");
  auto* state = dynamic_cast<intermediate_state_write_accumulator*>(private_state->data.get());
  GQE_EXPECTS(state != nullptr,
              "intermediate_state_write_task: accumulator belongs to another task");
  return *state;
}

[[nodiscard]] cudf::table_view validate_regular_input(state_container_view inputs)
{
  GQE_EXPECTS(!inputs.empty(),
              "intermediate_state_write_task::next: input container must not be empty");
  auto const table = to_table_view(inputs);
  GQE_EXPECTS(table.has_value() && static_cast<std::size_t>(table->num_columns()) == inputs.size(),
              "intermediate_state_write_task::next: input must be a regular table or count-only "
              "state; gather masked and indirect tables first");
  return *table;
}

}  // namespace

intermediate_state_write_task::intermediate_state_write_task(
  std::vector<cudf::data_type> output_data_types)
  : fold_task(task_ordering::preserve), _output_data_types(std::move(output_data_types))
{
}

state_container intermediate_state_write_task::initialize(context_reference,
                                                          query_stream const&,
                                                          rmm::device_async_resource_ref) const
{
  auto accumulator = std::make_unique<intermediate_state_write_accumulator>();
  return state_container_builder()
    .add_state(state_kind::task_private{std::move(accumulator)})
    .build();
}

void intermediate_state_write_task::next(state_container_view inputs,
                                         state_container_view accumulator,
                                         context_reference,
                                         query_stream const&,
                                         rmm::device_async_resource_ref) const
{
  if (inputs.empty()) { return; }

  auto& state = get_accumulator(accumulator);

  if (auto const count = try_row_count(inputs); count.has_value()) {
    GQE_EXPECTS(_output_data_types.empty(),
                "intermediate_state_write_task::next: configured for regular-table output but "
                "received count-only input");
    {
      std::scoped_lock const guard{state.mutex};
      GQE_EXPECTS(state.mode != input_mode::regular,
                  "intermediate_state_write_task::next: regular and count-only chunks cannot be "
                  "mixed");
      GQE_EXPECTS(
        state.row_count <=
          static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()) - *count,
        "intermediate_state_write_task::next: accumulated row count exceeds cudf::size_type",
        std::overflow_error);
      state.mode = input_mode::count_only;
      state.row_count += *count;
    }
  } else {
    auto const table = validate_regular_input(inputs);
    GQE_EXPECTS(!_output_data_types.empty(),
                "intermediate_state_write_task::next: configured for count-only output but "
                "received regular-table input");

    std::optional<cudf::table_view> reference_table;
    bool appended_first_chunk = false;
    {
      std::scoped_lock const guard{state.mutex};
      GQE_EXPECTS(state.mode != input_mode::count_only,
                  "intermediate_state_write_task::next: regular and count-only chunks cannot be "
                  "mixed");
      if (state.chunks.empty()) {
        state.chunks.push_back(make_mutable_state_copy(inputs));
        state.mode           = input_mode::regular;
        appended_first_chunk = true;
      } else {
        reference_table = to_table_view(state.chunks.front());
      }
    }

    if (!appended_first_chunk) {
      // `reference_table` borrows from the first retained chunk. `next` only appends chunks, and
      // the executor keeps the accumulator alive until `finalize`, so the view remains valid after
      // releasing the mutex.
      GQE_EXPECTS(reference_table.has_value(),
                  "intermediate_state_write_task::next: retained chunk is not a regular table");
      GQE_EXPECTS(cudf::have_same_types(*reference_table, table),
                  "intermediate_state_write_task::next: input chunk schemas do not match");
      {
        std::scoped_lock const guard{state.mutex};
        state.chunks.push_back(make_mutable_state_copy(inputs));
      }
    }
  }
}

state_container intermediate_state_write_task::finalize(state_container&& accumulator,
                                                        context_reference,
                                                        query_stream const& stream_handle,
                                                        rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  auto& state       = get_accumulator(accumulator);

  if (state.mode == input_mode::unset) {
    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  if (state.mode == input_mode::count_only) {
    return make_row_count_container(static_cast<cudf::size_type>(state.row_count));
  }

  auto& chunks = state.chunks;
  GQE_EXPECTS(!chunks.empty(),
              "intermediate_state_write_task::finalize: regular mode has no retained chunks");
  if (chunks.size() == 1) { return std::move(chunks.front()); }

  std::vector<cudf::table_view> tables;
  tables.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto const table = to_table_view(chunk);
    GQE_EXPECTS(table.has_value(),
                "intermediate_state_write_task::finalize: retained chunk is not a regular table");
    tables.push_back(*table);
  }
  auto result = cudf::concatenate(tables, stream, mr);
  // Keep `chunks` in the caller-owned accumulator. The executor holds the accumulator until its
  // result handoff has synchronized `stream`, so the asynchronous concatenate cannot outlive its
  // input columns.
  return state_container_builder().add_state(std::move(*result), stream_handle).build();
}

std::unique_ptr<task> intermediate_state_write_task::clone() const
{
  return std::make_unique<intermediate_state_write_task>(_output_data_types);
}

}  // namespace qep
}  // namespace gqe
