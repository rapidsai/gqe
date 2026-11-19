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

#include <gqe/qep/operator/reduce_aggregate.hpp>

#include <gqe/aggregation.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <algorithm>
#include <cassert>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Host-side running total for COUNT_ALL.
 *
 * The active alternative matches the user-supplied `output_data_types[i]` (selected at
 * `initialize` time). Caller is responsible for picking a width that won't overflow at the
 * expected row count.
 */
using count_running_total = std::variant<int32_t, int64_t>;

/**
 * @brief Task-private accumulator for `reduce_aggregate_task`.
 *
 * Non-COUNT_ALL kinds produce a 1-row partial column in `next`, appended to the per-value
 * `partials` vector; `finalize` concatenates each value's partials and reduces them with
 * the second-pass kind (apply-concat-apply). COUNT_VALID uses this path with first-pass
 * kind COUNT_VALID and second-pass kind SUM.
 *
 * COUNT_ALL short-circuits: each chunk contributes `get_row_count(inputs)` -- a host-side
 * row count -- which we sum into the per-value running total under the mutex. No column
 * data is required, so `SELECT COUNT(*) FROM t` works over the count-only chunks emitted
 * by the read task.
 */
struct reduce_aggregate_accumulator : public task_private_state {
  std::mutex mutex;  ///< Must be acquired by every `next` call before reading or mutating
                     ///< the rest of this state. Not required in `initialize` or `finalize` --
                     ///< the executor guarantees they run single-threaded.

  std::vector<count_running_total>
    count_running_totals;  ///< COUNT_ALL running total per value, typed to the user-supplied
                           ///< output type. Unmasked chunks contribute their row count. Masked
                           ///< chunks contribute their surviving row count.
  std::vector<std::vector<std::unique_ptr<cudf::column>>>
    partials;  ///< One 1-row column per `next` call, per value, for the apply-concat-apply
               ///< path. Used by every non-COUNT_ALL kind.

  std::size_t chunk_count = 0;  ///< Number of chunks accepted by `next`.
};

[[nodiscard]] count_running_total make_count_zero(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT32: return count_running_total{int32_t{0}};
    case cudf::type_id::INT64: return count_running_total{int64_t{0}};
    default:
      throw std::invalid_argument(
        std::format("reduce_aggregate_task: COUNT output type must be INT32 or INT64; got id {}",
                    static_cast<int>(type.id())));
  }
}

void validate_value_indices(
  std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> const&
    value_indices,
  cudf::size_type num_columns)
{
  for (auto const& [kind, value_idx] : value_indices) {
    if (kind != cudf::aggregation::COUNT_ALL && (*value_idx < 0 || *value_idx >= num_columns)) {
      throw std::out_of_range(std::format(
        "reduce_aggregate_task: value column index {} is out of range for {} input columns",
        *value_idx,
        num_columns));
    }
  }
}

// Ported from `executor/aggregate.cpp::get_reduce_aggregation`.
std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::ANY: return cudf::make_any_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::ALL: return cudf::make_all_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::SUM_OF_SQUARES:
      return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
    default:
      throw std::runtime_error(std::format(
        "reduce_aggregate_task: unsupported reduce aggregation kind {}", static_cast<int>(kind)));
  }
}

}  // namespace

reduce_aggregate_task::reduce_aggregate_task(
  std::vector<std::pair<cudf::aggregation::Kind, std::optional<cudf::size_type>>> value_indices,
  std::vector<cudf::data_type> output_data_types)
  : _value_indices(std::move(value_indices)), _output_data_types(std::move(output_data_types))
{
  GQE_EXPECTS(_value_indices.size() == _output_data_types.size(),
              "reduce_aggregate_task: value_indices and output_data_types must have the same "
              "length",
              std::invalid_argument);
  for (auto const& [kind, value_idx] : _value_indices) {
    GQE_EXPECTS((kind == cudf::aggregation::COUNT_ALL) == !value_idx.has_value(),
                "reduce_aggregate_task: COUNT_ALL must not select a value column; every other "
                "aggregation kind must select one",
                std::invalid_argument);
    if (kind == cudf::aggregation::COUNT_ALL) {
      _materialized_positions.push_back(std::nullopt);
      continue;
    }
    auto const position = std::ranges::find(_materialized_indices, *value_idx);
    auto const index =
      static_cast<cudf::size_type>(std::distance(_materialized_indices.begin(), position));
    if (position == _materialized_indices.end()) { _materialized_indices.push_back(*value_idx); }
    _materialized_positions.push_back(index);
  }
}

std::unique_ptr<task> reduce_aggregate_task::clone() const
{
  return std::make_unique<reduce_aggregate_task>(_value_indices, _output_data_types);
}

state_container reduce_aggregate_task::initialize(context_reference /* ctx_ref */,
                                                  query_stream const&,
                                                  rmm::device_async_resource_ref /* mr */) const
{
  gqe::utility::nvtx_scoped_range range{"reduce_aggregate_task::initialize"};

  auto accumulator_data = std::make_unique<reduce_aggregate_accumulator>();
  accumulator_data->count_running_totals.reserve(_value_indices.size());
  for (std::size_t i = 0; i < _value_indices.size(); ++i) {
    auto const kind = _value_indices[i].first;
    accumulator_data->count_running_totals.push_back(
      kind == cudf::aggregation::COUNT_ALL
        ? make_count_zero(_output_data_types[i])
        // Keep this vector index-aligned with _value_indices; non-COUNT_ALL entries are unused.
        : count_running_total{int64_t{0}});
  }
  accumulator_data->partials.resize(_value_indices.size());

  GQE_LOG_TRACE("Initialize QEP reduce aggregate task: aggregate_count={}", _value_indices.size());

  state_container_builder builder;
  builder.add_state(state_kind::task_private{std::move(accumulator_data)});
  return builder.build();
}

void reduce_aggregate_task::next(state_container_view inputs,
                                 state_container_view accumulator,
                                 context_reference /* ctx_ref */,
                                 query_stream const& stream_handle,
                                 rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range range{"reduce_aggregate_task::next"};

  auto* private_state = std::get_if<state_kind::task_private>(accumulator[0].get());
  assert(private_state != nullptr);
  // The `task_private` slot was populated in `initialize` with this concrete type, so a
  // `static_cast` is sufficient.
  auto* accumulator_data = static_cast<reduce_aggregate_accumulator*>(private_state->data.get());

  auto const count_only = try_row_count(inputs);
  std::optional<logical_table_view> logical;
  if (!count_only.has_value()) { logical = logical_table_view::try_from(inputs); }
  GQE_EXPECTS(count_only.has_value() || logical.has_value(),
              "reduce_aggregate_task::next: input is not a logical relation or count-only shape");

  if (count_only.has_value()) {
    GQE_EXPECTS(std::ranges::all_of(_value_indices,
                                    [](auto const& value_index) {
                                      return value_index.first == cudf::aggregation::COUNT_ALL;
                                    }),
                "reduce_aggregate_task::next: count-only input requires only COUNT_ALL values");
  } else {
    validate_value_indices(_value_indices, logical->num_columns());
  }

  bool const chunk_is_masked = logical.has_value() && logical->valid_mask().has_value();
  auto const chunk_rows      = count_only.has_value() ? *count_only : logical->row_count();

  std::optional<column_materialization> materialized_values;
  if (logical.has_value() && !_materialized_indices.empty()) {
    materialized_values.emplace(
      materialize_columns(*logical, _materialized_indices, stream_handle, mr));
  }

  GQE_LOG_TRACE(
    "Fold QEP reduce aggregate chunk: chunk_rows={}, chunk_value_cols={}, "
    "chunk_has_valid_mask={}, valid_mask_null_count={}",
    chunk_rows,
    logical.has_value() ? logical->num_columns() : 0,
    chunk_is_masked,
    chunk_is_masked ? logical->valid_mask()->null_count() : 0);

  std::vector<std::unique_ptr<cudf::column>> chunk_partials(_value_indices.size());
  std::vector<int64_t> chunk_count_increments(_value_indices.size(), 0);

  std::optional<null_mask_injection> injection;
  cudf::table_view values_table;
  if (materialized_values.has_value()) {
    values_table = materialized_values->table_view();
    if (chunk_is_masked) {
      injection.emplace(
        inject_valid_mask_as_nulls(*logical->valid_mask(), values_table, stream_handle, mr));
      values_table = injection->as_table_view();
    }
  }

  auto surviving_rows = chunk_rows;
  if (chunk_is_masked && std::ranges::any_of(_value_indices, [](auto const& value_index) {
        return value_index.first == cudf::aggregation::COUNT_ALL;
      })) {
    // False and null mask entries both reject a row.
    auto [mask, rejected_rows] = cudf::bools_to_mask(*logical->valid_mask(), stream, mr);
    surviving_rows             = chunk_rows - rejected_rows;
  }

  for (std::size_t i = 0; i < _value_indices.size(); ++i) {
    auto const kind = _value_indices[i].first;
    if (kind == cudf::aggregation::COUNT_ALL) {
      chunk_count_increments[i] = static_cast<int64_t>(surviving_rows);
      continue;
    }

    auto const materialized_index = *_materialized_positions[i];
    auto const reduce_op          = make_reduce_aggregation(kind);
    auto scalar                   = cudf::reduce(
      values_table.column(materialized_index), *reduce_op, _output_data_types[i], stream, mr);
    chunk_partials[i] = cudf::make_column_from_scalar(*scalar, 1, stream, mr);
  }

  order_deallocation_after_stream_use(inputs, stream_handle);

  std::lock_guard<std::mutex> const guard{accumulator_data->mutex};

  ++accumulator_data->chunk_count;

  for (std::size_t i = 0; i < _value_indices.size(); ++i) {
    auto const kind = _value_indices[i].first;
    if (kind == cudf::aggregation::COUNT_ALL) {
      auto const increment = chunk_count_increments[i];
      std::visit(
        [increment](auto& total) {
          using T = std::decay_t<decltype(total)>;
          total += static_cast<T>(increment);
        },
        accumulator_data->count_running_totals[i]);
    } else {
      accumulator_data->partials[i].push_back(std::move(chunk_partials[i]));
    }
  }
}

state_container reduce_aggregate_task::finalize(state_container&& accumulator,
                                                context_reference /* ctx_ref */,
                                                query_stream const& stream_handle,
                                                rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  gqe::utility::nvtx_scoped_range range{"reduce_aggregate_task::finalize"};

  auto* private_state = std::get_if<state_kind::task_private>(accumulator[0].get());
  assert(private_state != nullptr);
  auto* accumulator_data = static_cast<reduce_aggregate_accumulator*>(private_state->data.get());

  GQE_LOG_TRACE("Finalize QEP reduce aggregate task: num_aggregates={}, accumulated_chunk_count={}",
                _value_indices.size(),
                accumulator_data->chunk_count);

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(_value_indices.size());

  for (std::size_t i = 0; i < _value_indices.size(); ++i) {
    auto const kind = _value_indices[i].first;

    if (kind == cudf::aggregation::COUNT_ALL) {
      // Host-side fast path: emit the running total as a 1-row column whose type matches
      // the user-supplied output type (the active variant alternative).
      auto column = std::visit(
        [&](auto total) {
          using T     = std::decay_t<decltype(total)>;
          auto scalar = cudf::make_fixed_width_scalar<T>(total, stream, mr);
          return cudf::make_column_from_scalar(*scalar, 1, stream, mr);
        },
        accumulator_data->count_running_totals[i]);
      result_columns.push_back(std::move(column));
      continue;
    }

    auto& partials = accumulator_data->partials[i];
    if (partials.empty()) {
      // No chunks delivered. SQL semantics: COUNT returns 0, every other reduction returns NULL.
      if (kind == cudf::aggregation::COUNT_VALID) {
        auto column = std::visit(
          [&](auto value) {
            using T     = std::decay_t<decltype(value)>;
            auto scalar = cudf::make_fixed_width_scalar<T>(value, stream, mr);
            return cudf::make_column_from_scalar(*scalar, 1, stream, mr);
          },
          make_count_zero(_output_data_types[i]));
        result_columns.push_back(std::move(column));
      } else {
        result_columns.push_back(cudf::make_fixed_width_column(
          _output_data_types[i], 1, cudf::mask_state::ALL_NULL, stream, mr));
      }
      continue;
    }

    // Apply-concat-apply: concatenate per-chunk partials, then reduce with the second-pass
    // kind to a final scalar.
    std::vector<cudf::column_view> views;
    views.reserve(partials.size());
    for (auto& p : partials) {
      views.push_back(p->view());
    }
    auto concatenated    = cudf::concatenate(views, stream, mr);
    auto const reduce_op = make_reduce_aggregation(get_second_aggregation_kind(kind));
    auto scalar = cudf::reduce(concatenated->view(), *reduce_op, _output_data_types[i], stream, mr);
    result_columns.push_back(cudf::make_column_from_scalar(*scalar, 1, stream, mr));
  }

  cudf::table result_table{std::move(result_columns)};

  return state_container_builder().add_state(std::move(result_table), stream_handle).build();
}

}  // namespace qep
}  // namespace gqe
