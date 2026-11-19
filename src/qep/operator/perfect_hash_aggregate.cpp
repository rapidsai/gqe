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

#include <gqe/qep/operator/perfect_hash_aggregate.hpp>

#include "apply_concat_apply_aggregate_common.hpp"
#include "materialized_chunks.hpp"

#include <gqe/executor/perfect_hash/scatter_aggregate.hpp>
#include <gqe/executor/perfect_hash/unique_indices.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cassert>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

struct perfect_hash_aggregate_accumulator : public task_private_state {
  std::mutex mutex;
  std::vector<state_container> chunks;
  bool has_mask = false;
  int64_t input_row_count{0};
};

// Mirrors the key-type dispatch in the buffer accessor of
// `gqe/executor/perfect_hash/detail/query_common.hpp`.
[[nodiscard]] bool is_supported_key_type(cudf::type_id type)
{
  switch (type) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::TIMESTAMP_DAYS: return true;
    default: return false;
  }
}

// Mirrors the input/output type dispatch in `executor/perfect_hash/scatter_aggregate.cpp`.
[[nodiscard]] bool is_supported_value_type(cudf::type_id type)
{
  return type == cudf::type_id::INT32 || type == cudf::type_id::INT64 ||
         type == cudf::type_id::FLOAT32 || type == cudf::type_id::FLOAT64;
}

// Mirrors the aggregation-kind dispatch in `executor/perfect_hash/scatter_aggregate.cpp`.
[[nodiscard]] bool is_supported_aggregation(cudf::aggregation::Kind kind)
{
  return kind == cudf::aggregation::SUM || kind == cudf::aggregation::PRODUCT ||
         kind == cudf::aggregation::MIN || kind == cudf::aggregation::COUNT_VALID ||
         kind == cudf::aggregation::COUNT_ALL;
}

void validate_configuration(
  std::vector<cudf::size_type> const& key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> const& value_indices,
  std::vector<cudf::data_type> const& output_data_types)
{
  GQE_EXPECTS(!key_indices.empty(),
              "perfect_hash_aggregate_task: at least one grouping key is required",
              std::invalid_argument);
  GQE_EXPECTS(output_data_types.size() == key_indices.size() + value_indices.size(),
              "perfect_hash_aggregate_task: output type count must equal key count plus aggregate "
              "count",
              std::invalid_argument);

  for (std::size_t i = 0; i < value_indices.size(); ++i) {
    auto const kind        = value_indices[i].first;
    auto const output_type = output_data_types[key_indices.size() + i].id();
    GQE_EXPECTS(is_supported_aggregation(kind),
                std::format("perfect_hash_aggregate_task: unsupported aggregation kind {}",
                            static_cast<int>(kind)),
                std::invalid_argument);
    GQE_EXPECTS(is_supported_value_type(output_type),
                std::format("perfect_hash_aggregate_task: unsupported aggregate output type {}",
                            static_cast<int>(output_type)),
                std::invalid_argument);
  }
}

void validate_input(
  detail::aggregate_input_view const& input,
  std::vector<cudf::size_type> const& key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> const& value_indices,
  std::vector<cudf::data_type> const& output_data_types)
{
  auto const num_columns = input.table.num_columns();
  for (std::size_t i = 0; i < key_indices.size(); ++i) {
    auto const index = key_indices[i];
    GQE_EXPECTS(index >= 0 && index < num_columns,
                std::format("perfect_hash_aggregate_task: key column index {} is out of range for "
                            "{} input columns",
                            index,
                            num_columns),
                std::out_of_range);
    auto const column = input.table.column(index);
    GQE_EXPECTS(cudf::is_fixed_width(column.type()) && is_supported_key_type(column.type().id()),
                "perfect_hash_aggregate_task: grouping keys must use perfect-hash-supported "
                "fixed-width types",
                std::invalid_argument);
    GQE_EXPECTS(column.null_count() == 0,
                "perfect_hash_aggregate_task: grouping key columns must not contain nulls, "
                "including rows excluded by valid_mask",
                std::invalid_argument);
    auto const declared_type =
      cudf::column_view{output_data_types[i], 0, nullptr, nullptr, 0, 0, {}};
    GQE_EXPECTS(cudf::have_same_types(column, declared_type),
                "perfect_hash_aggregate_task: declared key output type does not match the input "
                "key type",
                std::invalid_argument);
  }

  for (auto const [kind, index] : value_indices) {
    GQE_EXPECTS(index >= 0 && index < num_columns,
                std::format("perfect_hash_aggregate_task: value column index {} is out of range "
                            "for {} input columns",
                            index,
                            num_columns),
                std::out_of_range);
    auto const column = input.table.column(index);
    GQE_EXPECTS(is_supported_value_type(column.type().id()),
                "perfect_hash_aggregate_task: aggregate values must use a perfect-hash-supported "
                "numeric type",
                std::invalid_argument);
    GQE_EXPECTS(kind == cudf::aggregation::COUNT_ALL || column.null_count() == 0,
                "perfect_hash_aggregate_task: aggregate value columns must not contain nulls, "
                "including rows excluded by valid_mask, except for COUNT_ALL",
                std::invalid_argument);
  }
}

[[nodiscard]] std::unique_ptr<cudf::table> aggregate(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> const& value_indices,
  std::vector<cudf::data_type> const& output_data_types,
  cudf::column_view active_mask,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  std::vector<cudf::column_view> key_columns;
  key_columns.reserve(key_indices.size());
  for (auto const index : key_indices) {
    key_columns.push_back(table.column(index));
  }

  auto [unique_key_indices, group_indices] =
    perfect_hash::unique_indices(key_columns, active_mask, stream);
  auto const group_count = static_cast<cudf::size_type>(unique_key_indices.size());
  auto const gather_map  = cudf::column_view{
    cudf::data_type{cudf::type_id::INT32}, group_count, unique_key_indices.data(), nullptr, 0};
  auto gathered_keys = cudf::gather(
    cudf::table_view{key_columns}, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto result_columns = gathered_keys->release();

  if (group_count == 0) {
    for (std::size_t i = 0; i < value_indices.size(); ++i) {
      result_columns.push_back(cudf::make_empty_column(output_data_types[key_indices.size() + i]));
    }
    return std::make_unique<cudf::table>(std::move(result_columns));
  }

  for (std::size_t i = 0; i < value_indices.size(); ++i) {
    auto const [kind, value_index] = value_indices[i];
    auto aggregate_result =
      perfect_hash::scatter_aggregate(table.column(value_index),
                                      group_indices,
                                      active_mask,
                                      std::nullopt,
                                      kind,
                                      group_count,
                                      output_data_types[key_indices.size() + i].id(),
                                      stream);
    result_columns.push_back(std::make_unique<cudf::column>(std::move(aggregate_result)));
  }

  return std::make_unique<cudf::table>(std::move(result_columns));
}

[[nodiscard]] cudf::table_view stored_table_view(state_container_view chunk, bool has_mask)
{
  if (has_mask) {
    auto const masked = masked_table_view::try_from(chunk);
    assert(masked.has_value());
    return masked->columns;
  }
  auto const table = to_table_view(chunk);
  assert(table.has_value());
  return *table;
}

}  // namespace

perfect_hash_partial_aggregate_task::perfect_hash_partial_aggregate_task(
  std::vector<cudf::size_type> key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
  std::vector<cudf::data_type> output_data_types)
  : _key_indices(std::move(key_indices)),
    _value_indices(std::move(value_indices)),
    _output_data_types(std::move(output_data_types))
{
  validate_configuration(_key_indices, _value_indices, _output_data_types);

  auto layout                 = detail::make_aggregate_input_layout(_key_indices, _value_indices);
  _materialized_indices       = std::move(layout.materialized_indices);
  _materialized_key_indices   = std::move(layout.materialized_key_indices);
  _materialized_value_indices = std::move(layout.materialized_value_indices);
}

std::unique_ptr<task> perfect_hash_partial_aggregate_task::clone() const
{
  return std::make_unique<perfect_hash_partial_aggregate_task>(
    _key_indices, _value_indices, _output_data_types);
}

std::optional<state_container> perfect_hash_partial_aggregate_task::next(
  state_container_view inputs,
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"perfect_hash_partial_aggregate_task::next"};

  auto materialized =
    detail::materialize_aggregate_input(inputs, _materialized_indices, stream_handle, mr);
  auto const input = materialized.view();
  validate_input(input, _materialized_key_indices, _materialized_value_indices, _output_data_types);
  auto normalized = detail::normalize_active_mask(input.mask, stream_handle, mr);
  auto partial    = aggregate(input.table,
                           _materialized_key_indices,
                           _materialized_value_indices,
                           _output_data_types,
                           normalized.view,
                           stream_handle,
                           mr);

  GQE_LOG_TRACE(
    "Apply QEP perfect-hash partial aggregate: input_rows={}, input_columns={}, "
    "input_has_valid_mask={}, output_groups={}",
    input.table.num_rows(),
    materialized.logical_column_count,
    input.has_mask,
    partial->num_rows());

  auto output = state_container_builder().add_state(std::move(*partial), stream_handle).build();
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

perfect_hash_aggregate_task::perfect_hash_aggregate_task(
  std::vector<cudf::size_type> key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
  std::vector<cudf::data_type> output_data_types)
  : _key_indices(std::move(key_indices)),
    _value_indices(std::move(value_indices)),
    _output_data_types(std::move(output_data_types))
{
  validate_configuration(_key_indices, _value_indices, _output_data_types);
}

std::unique_ptr<task> perfect_hash_aggregate_task::clone() const
{
  return std::make_unique<perfect_hash_aggregate_task>(
    _key_indices, _value_indices, _output_data_types);
}

state_container perfect_hash_aggregate_task::initialize(
  context_reference /* ctx_ref */,
  query_stream const&,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"perfect_hash_aggregate_task::initialize"};

  GQE_LOG_TRACE("Initialize QEP perfect-hash aggregate: key_count={}, aggregate_count={}",
                _key_indices.size(),
                _value_indices.size());
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<perfect_hash_aggregate_accumulator>()})
    .build();
}

void perfect_hash_aggregate_task::next(state_container_view inputs,
                                       state_container_view accumulator,
                                       context_reference /* ctx_ref */,
                                       query_stream const&,
                                       rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"perfect_hash_aggregate_task::next"};

  auto const input = detail::resolve_aggregate_input(inputs, /*is_mask_allowed=*/true);
  validate_input(input, _key_indices, _value_indices, _output_data_types);

  auto* private_state = std::get_if<state_kind::task_private>(accumulator[0].get());
  assert(private_state != nullptr);
  auto* accumulator_data =
    static_cast<perfect_hash_aggregate_accumulator*>(private_state->data.get());

  {
    std::lock_guard<std::mutex> const guard{accumulator_data->mutex};
    if (accumulator_data->chunks.empty()) {
      accumulator_data->has_mask = input.has_mask;
    } else {
      GQE_EXPECTS(accumulator_data->has_mask == input.has_mask,
                  "perfect_hash_aggregate_task::next: mixed masked/unmasked input chunks are not "
                  "supported");
      auto const first_table =
        stored_table_view(accumulator_data->chunks.front(), accumulator_data->has_mask);
      GQE_EXPECTS(cudf::have_same_types(first_table, input.table),
                  "perfect_hash_aggregate_task::next: input chunk schemas do not match",
                  std::invalid_argument);
    }

    accumulator_data->chunks.push_back(make_mutable_state_copy(inputs));
    accumulator_data->input_row_count += input.table.num_rows();
  }

  GQE_LOG_TRACE(
    "Fold QEP perfect-hash aggregate chunk: chunk_rows={}, chunk_columns={}, "
    "chunk_has_valid_mask={}",
    input.table.num_rows(),
    input.table.num_columns(),
    input.has_mask);
}

state_container perfect_hash_aggregate_task::finalize(state_container&& accumulator,
                                                      context_reference /* ctx_ref */,
                                                      query_stream const& stream_handle,
                                                      rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"perfect_hash_aggregate_task::finalize"};

  auto* private_state = std::get_if<state_kind::task_private>(accumulator[0].get());
  assert(private_state != nullptr);
  auto* accumulator_data =
    static_cast<perfect_hash_aggregate_accumulator*>(private_state->data.get());

  if (accumulator_data->chunks.empty()) {
    GQE_LOG_TRACE(
      "Finalize QEP perfect-hash aggregate: chunk_count=0, input_rows=0, "
      "output_groups=0");
    return make_empty_state_container(_output_data_types);
  }

  auto const chunk_count = accumulator_data->chunks.size();
  auto concatenated =
    detail::concatenate_materialized_chunks(accumulator_data->chunks, stream_handle, mr);
  auto const active_mask =
    concatenated.mask.has_value() ? concatenated.mask->view() : cudf::column_view{};
  auto normalized = detail::normalize_active_mask(active_mask, stream_handle, mr);
  auto result     = aggregate(concatenated.table->view(),
                          _key_indices,
                          _value_indices,
                          _output_data_types,
                          normalized.view,
                          stream_handle,
                          mr);

  GQE_LOG_TRACE(
    "Finalize QEP perfect-hash aggregate: chunk_count={}, input_rows={}, "
    "output_groups={}",
    chunk_count,
    accumulator_data->input_row_count,
    result->num_rows());

  return state_container_builder().add_state(std::move(*result), stream_handle).build();
}

}  // namespace qep
}  // namespace gqe
