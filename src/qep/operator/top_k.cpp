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

#include <gqe/qep/operator/top_k.hpp>

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::qep {
namespace {

/**
 * @brief Per-execution state for a top-K fold.
 *
 * Concurrent `next()` calls publish one candidate table per input chunk. After every `next()` call
 * completes, `finalize()` consumes the collected tables without concurrent access.
 */
struct top_k_accumulator : task_private_state {
  // Protects host-side mutation of chunk_candidates while next() publishes entries. It does not
  // synchronize CUDA work or protect candidate column data.
  std::mutex mutex;
  std::vector<state_container> chunk_candidates;
};

top_k_accumulator* get_accumulator(state_container_view accumulator)
{
  GQE_EXPECTS(accumulator.size() == 1 && accumulator.front() != nullptr,
              "top_k_task: accumulator must contain one non-null task_private state");
  auto* private_state = std::get_if<state_kind::task_private>(accumulator.front().get());
  GQE_EXPECTS(private_state != nullptr && private_state->data != nullptr,
              "top_k_task: accumulator is not a task_private state");
  return static_cast<top_k_accumulator*>(private_state->data.get());
}

/**
 * @brief Select the positions of the first `k` rows under the requested key ordering.
 *
 * Non-null keys use `cudf::top_k_order`. Nullable keys fall back to `cudf::sorted_order` because
 * cuDF top-K does not support nulls. The returned positions index `key` and are not guaranteed to
 * be in final sorted order. Requests larger than `key.size()` are clamped to the column size.
 *
 * @param[in] key Column whose values determine selection.
 * @param[in] k Number of positions to select.
 * @param[in] order Ascending or descending key order.
 * @param[in] null_precedence Placement of null keys in the nullable fallback.
 * @param[in] stream CUDA stream on which to order selection work.
 * @param[in] mr Device memory resource used for allocations.
 * @return Owning INT32 column containing positions into `key`.
 */
std::unique_ptr<cudf::column> select_positions(cudf::column_view key,
                                               cudf::size_type k,
                                               cudf::order order,
                                               cudf::null_order null_precedence,
                                               query_stream const& stream_handle,
                                               rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  k                 = std::min(k, key.size());
  if (k == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  } else if (key.has_nulls()) {
    GQE_LOG_TRACE("Nullable top-K key uses full-sort fallback: rows={}, k={}", key.size(), k);
    auto order_column =
      cudf::sorted_order(cudf::table_view{{key}}, {order}, {null_precedence}, stream, mr);
    return std::make_unique<cudf::column>(
      cudf::slice(order_column->view(), {0, k}).front(), stream, mr);
  } else {
    return cudf::top_k_order(key, k, order, stream, mr);
  }
}

}  // namespace

top_k_task::top_k_task(cudf::size_type key_index,
                       cudf::order column_order,
                       cudf::null_order null_precedence,
                       cudf::size_type offset,
                       cudf::size_type count,
                       std::vector<cudf::data_type> output_data_types)
  : fold_task(task_ordering::establish),
    _key_index{key_index},
    _column_order{column_order},
    _null_precedence{null_precedence},
    _offset{offset},
    _count{count},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(_key_index >= 0, "top_k_task: key_index must be non-negative", std::invalid_argument);
  GQE_EXPECTS(_offset >= 0, "top_k_task: offset must be non-negative", std::invalid_argument);
  GQE_EXPECTS(_count >= 0, "top_k_task: count must be non-negative", std::invalid_argument);
  GQE_EXPECTS(!_output_data_types.empty(),
              "top_k_task: output_data_types must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(static_cast<std::size_t>(_key_index) < _output_data_types.size(),
              "top_k_task: key_index is outside output_data_types",
              std::out_of_range);
  GQE_EXPECTS(_offset <= std::numeric_limits<cudf::size_type>::max() - _count,
              "top_k_task: offset + count overflows cudf::size_type",
              std::overflow_error);
}

std::unique_ptr<task> top_k_task::clone() const
{
  return std::make_unique<top_k_task>(
    _key_index, _column_order, _null_precedence, _offset, _count, _output_data_types);
}

state_container top_k_task::initialize(context_reference,
                                       query_stream const&,
                                       rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::top_k_task::initialize"};
  auto accumulator = std::make_unique<top_k_accumulator>();
  return state_container_builder()
    .add_state(state_kind::task_private{std::move(accumulator)})
    .build();
}

void top_k_task::next(state_container_view inputs,
                      state_container_view accumulator,
                      context_reference,
                      query_stream const& stream_handle,
                      rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::top_k_task::next"};
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "top_k_task::next: input is not a logical relation");
  column_materialization materialized =
    materialize_all_compacted_columns(*logical, stream_handle, mr);
  auto const input = materialized.table_view();
  GQE_EXPECTS(input.num_columns() == static_cast<cudf::size_type>(_output_data_types.size()),
              "top_k_task::next: input column count does not match output_data_types");
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    auto const declared_type =
      cudf::column_view{_output_data_types[i], 0, nullptr, nullptr, 0, 0, {}};
    GQE_EXPECTS(cudf::have_same_types(input.column(i), declared_type),
                "top_k_task::next: input column type does not match output_data_types");
  }
  GQE_EXPECTS(_key_index < input.num_columns(),
              "top_k_task::next: key_index is outside the input columns",
              std::out_of_range);

  auto const k   = std::min(_offset + _count, input.num_rows());
  auto positions = select_positions(
    input.column(_key_index), k, _column_order, _null_precedence, stream_handle, mr);
  auto candidates = cudf::gather(input,
                                 positions->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::NOT_ALLOWED,
                                 stream,
                                 mr);
  auto candidate_states =
    state_container_builder().add_state(std::move(*candidates), stream_handle).build();
  order_deallocation_after_stream_use(inputs, stream_handle);

  auto* accumulator_data = get_accumulator(accumulator);
  {
    std::lock_guard<std::mutex> const guard{accumulator_data->mutex};
    accumulator_data->chunk_candidates.push_back(std::move(candidate_states));
  }
}

state_container top_k_task::finalize(state_container&& accumulator,
                                     context_reference,
                                     query_stream const& stream_handle,
                                     rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::top_k_task::finalize"};
  auto* accumulator_data = get_accumulator(accumulator);
  auto const num_chunks  = accumulator_data->chunk_candidates.size();

  state_container candidate_states;
  if (num_chunks == 0) {
    candidate_states = make_empty_state_container(_output_data_types);
  } else if (num_chunks == 1) {
    candidate_states = std::move(accumulator_data->chunk_candidates.front());
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(accumulator_data->chunk_candidates.size());
    for (auto const& chunk : accumulator_data->chunk_candidates) {
      auto const view = to_table_view(chunk);
      GQE_EXPECTS(view.has_value(), "top_k_task::finalize: candidate is not a plain table");
      views.push_back(*view);
    }
    auto candidates = cudf::concatenate(views, stream, mr);
    candidate_states =
      state_container_builder().add_state(std::move(*candidates), stream_handle).build();
  }
  auto const candidates = to_table_view(candidate_states);
  GQE_EXPECTS(candidates.has_value(), "top_k_task::finalize: candidates are not a plain table");

  auto selected_count = candidates->num_rows();
  std::unique_ptr<cudf::column> ordered;
  if (num_chunks == 1) {
    // next() already reduced the only chunk to at most offset + count candidates.
    ordered = cudf::sorted_order(cudf::table_view{{candidates->column(_key_index)}},
                                 {_column_order},
                                 {_null_precedence},
                                 stream,
                                 mr);
  } else {
    selected_count = std::min(_offset + _count, candidates->num_rows());
    auto selected  = select_positions(candidates->column(_key_index),
                                     selected_count,
                                     _column_order,
                                     _null_precedence,
                                     stream_handle,
                                     mr);

    // `top_k_order` returns the correct set but does not promise order. Sort only those K keys,
    // then compose that order with the positions into `candidates`.
    auto selected_keys = cudf::gather(cudf::table_view{{candidates->column(_key_index)}},
                                      selected->view(),
                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                      cudf::negative_index_policy::NOT_ALLOWED,
                                      stream,
                                      mr);
    auto within_selected =
      cudf::sorted_order(selected_keys->view(), {_column_order}, {_null_precedence}, stream, mr);
    auto ordered_table   = cudf::gather(cudf::table_view{{selected->view()}},
                                      within_selected->view(),
                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                      cudf::negative_index_policy::NOT_ALLOWED,
                                      stream,
                                      mr);
    auto ordered_columns = ordered_table->release();
    ordered              = std::move(ordered_columns.front());
  }

  auto const begin = std::min(_offset, ordered->size());
  auto const end   = std::min(begin + _count, ordered->size());
  auto positions =
    std::make_unique<cudf::column>(cudf::slice(ordered->view(), {begin, end}).front(), stream, mr);

  GQE_LOG_TRACE("Finalize QEP top-k task: candidates={}, selected={}, offset={}, output={}",
                candidates->num_rows(),
                selected_count,
                _offset,
                positions->size());

  auto const num_columns = candidates->num_columns();
  auto relation_source   = indirect_table_source_state::make(std::move(candidate_states));
  auto gather_source     = indirect_gather_source_state::make(
    std::move(relation_source), make_position_list(std::move(positions), stream_handle));

  state_container output;
  output.reserve(static_cast<std::size_t>(num_columns));
  for (cudf::size_type i = 0; i < num_columns; ++i) {
    output.push_back(indirect_column_state::make(gather_source, i));
  }
  return output;
}

}  // namespace gqe::qep
