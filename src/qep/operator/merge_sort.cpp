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

#include <gqe/qep/operator/merge_sort.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {
namespace {

struct merge_sort_accumulator : public task_private_state {
  std::mutex mutex;
  std::vector<cudf::table> sorted_chunks;
};

[[nodiscard]] merge_sort_accumulator& get_accumulator(state_container_view accumulator)
{
  assert(accumulator.size() == 1);
  assert(accumulator.front() != nullptr);
  auto* private_state = std::get_if<state_kind::task_private>(accumulator.front().get());
  assert(private_state != nullptr);
  assert(private_state->data != nullptr);
  return *static_cast<merge_sort_accumulator*>(private_state->data.get());
}

}  // namespace

merge_sort_task::merge_sort_task(std::vector<cudf::size_type> key_column_indices,
                                 std::vector<cudf::order> column_orders,
                                 std::vector<cudf::null_order> null_precedences,
                                 std::vector<cudf::data_type> output_data_types)
  : fold_task(task_ordering::establish),
    _key_column_indices(std::move(key_column_indices)),
    _column_orders(std::move(column_orders)),
    _null_precedences(std::move(null_precedences)),
    _output_data_types(std::move(output_data_types))
{
  GQE_EXPECTS(!_output_data_types.empty(),
              "qep::merge_sort_task: output_data_types must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(!_key_column_indices.empty(),
              "qep::merge_sort_task: key_column_indices must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(_key_column_indices.size() == _column_orders.size(),
              "qep::merge_sort_task: key_column_indices and column_orders must have equal lengths",
              std::invalid_argument);
  GQE_EXPECTS(
    _key_column_indices.size() == _null_precedences.size(),
    "qep::merge_sort_task: key_column_indices and null_precedences must have equal lengths",
    std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_key_column_indices,
                                  [&](auto const index) {
                                    return index >= 0 && static_cast<std::size_t>(index) <
                                                           _output_data_types.size();
                                  }),
              "qep::merge_sort_task: key column index is out of range",
              std::invalid_argument);
}

std::unique_ptr<task> merge_sort_task::clone() const
{
  return std::make_unique<merge_sort_task>(
    _key_column_indices, _column_orders, _null_precedences, _output_data_types);
}

state_container merge_sort_task::initialize(context_reference,
                                            query_stream const&,
                                            rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::merge_sort_task::initialize"};
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<merge_sort_accumulator>()})
    .build();
}

void merge_sort_task::next(state_container_view inputs,
                           state_container_view accumulator,
                           context_reference,
                           query_stream const& stream_handle,
                           rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::merge_sort_task::next"};
  GQE_EXPECTS(!inputs.empty(), "qep::merge_sort_task::next: input container is empty");
  GQE_EXPECTS(inputs.size() == _output_data_types.size(),
              "qep::merge_sort_task::next: input column count does not match declared schema");

  cudf::size_type row_count = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto const& slot = inputs[index];
    GQE_EXPECTS(slot != nullptr, "qep::merge_sort_task::next: input contains a null state slot");
    GQE_EXPECTS(std::holds_alternative<state_kind::cudf_column>(*slot) ||
                  std::holds_alternative<state_kind::cudf_column_view>(*slot),
                "qep::merge_sort_task::next: input must have regular-table shape");
    auto const column = to_column_view(slot);
    assert(column.has_value());
    if (index == 0) {
      row_count = column->size();
    } else {
      GQE_EXPECTS(column->size() == row_count,
                  "qep::merge_sort_task::next: input column row counts do not match");
    }
    GQE_EXPECTS(column->type() == _output_data_types[index],
                "qep::merge_sort_task::next: input column type does not match declared schema");
  }

  auto const table = to_table_view(inputs);
  GQE_EXPECTS(table.has_value(), "qep::merge_sort_task::next: input must have regular-table shape");

  auto const key_table = table->select(_key_column_indices);
  auto sorted = cudf::sort_by_key(*table, key_table, _column_orders, _null_precedences, stream, mr);

  GQE_LOG_TRACE("Sort QEP merge-sort chunk: input_rows={}, input_cols={}",
                table->num_rows(),
                table->num_columns());

  auto& accumulator_data = get_accumulator(accumulator);
  std::lock_guard<std::mutex> const guard{accumulator_data.mutex};
  accumulator_data.sorted_chunks.emplace_back(std::move(*sorted));
}

state_container merge_sort_task::finalize(state_container&& accumulator,
                                          context_reference,
                                          query_stream const& stream_handle,
                                          rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::merge_sort_task::finalize"};
  auto& accumulator_data = get_accumulator(accumulator);
  if (accumulator_data.sorted_chunks.empty()) {
    return make_empty_state_container(_output_data_types);
  }

  std::vector<cudf::table_view> sorted_chunk_views;
  sorted_chunk_views.reserve(accumulator_data.sorted_chunks.size());
  for (auto const& chunk : accumulator_data.sorted_chunks) {
    sorted_chunk_views.push_back(chunk.view());
  }

  auto merged = cudf::merge(
    sorted_chunk_views, _key_column_indices, _column_orders, _null_precedences, stream, mr);

  GQE_LOG_TRACE("Finalize QEP merge-sort: output_rows={}, output_cols={}, chunks={}",
                merged->num_rows(),
                merged->num_columns(),
                sorted_chunk_views.size());
  return state_container_builder().add_state(std::move(*merged), stream_handle).build();
}

}  // namespace qep
}  // namespace gqe
