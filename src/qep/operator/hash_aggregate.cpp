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

#include <gqe/qep/operator/hash_aggregate.hpp>

#include "apply_concat_apply_aggregate_common.hpp"
#include "materialized_chunks.hpp"

#include <gqe/aggregation.hpp>
#include <gqe/executor/groupby.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cassert>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Retained partial chunks for one hash-aggregate fold.
 *
 * `mutex` protects `chunks`, `input_row_count`, and their joint update in `next`. The executor
 * calls `finalize` only after all concurrent `next` calls finish, so finalization needs no latch.
 */
struct hash_aggregate_accumulator : public task_private_state {
  std::mutex mutex;  ///< Protects `chunks`, `input_row_count`, and their joint update in `next`.
  std::vector<state_container> chunks;
  int64_t input_row_count{0};
};

[[nodiscard]] bool is_supported_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::SUM:
    case cudf::aggregation::COUNT_VALID:
    case cudf::aggregation::COUNT_ALL:
    case cudf::aggregation::MIN:
    case cudf::aggregation::MAX: return true;
    default: return false;
  }
}

void validate_configuration(
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices,
  bool allow_count_aggregations,
  std::optional<std::size_t> output_type_count = std::nullopt)
{
  GQE_EXPECTS(!key_indices.empty(),
              "hash_aggregate_task: at least one grouping key is required",
              std::invalid_argument);
  if (output_type_count.has_value()) {
    GQE_EXPECTS(*output_type_count == key_indices.size() + value_indices.size(),
                "hash_aggregate_task: output type count must equal key count plus aggregate count",
                std::invalid_argument);
  }
  for (auto const& [kind, index] : value_indices) {
    static_cast<void>(index);
    auto const is_count =
      kind == cudf::aggregation::COUNT_VALID || kind == cudf::aggregation::COUNT_ALL;
    GQE_EXPECTS(
      is_supported_aggregation(kind) && (allow_count_aggregations || !is_count),
      std::format("hash_aggregate_task: unsupported aggregation kind {}", static_cast<int>(kind)),
      std::invalid_argument);
  }
}

void validate_indices(
  cudf::table_view table,
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices)
{
  auto const num_columns = table.num_columns();
  for (auto const index : key_indices) {
    GQE_EXPECTS(index >= 0 && index < num_columns,
                std::format("hash_aggregate_task: key column index {} is out of range for {} input "
                            "columns",
                            index,
                            num_columns),
                std::out_of_range);
  }
  for (auto const& [kind, index] : value_indices) {
    static_cast<void>(kind);
    GQE_EXPECTS(index >= 0 && index < num_columns,
                std::format("hash_aggregate_task: value column index {} is out of range for {} "
                            "input columns",
                            index,
                            num_columns),
                std::out_of_range);
  }
}

void validate_fold_schema(
  cudf::table_view table,
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices,
  std::span<cudf::data_type const> final_output_types)
{
  GQE_EXPECTS(
    table.num_columns() == static_cast<cudf::size_type>(key_indices.size() + value_indices.size()),
    "hash_aggregate_task: partial column count must equal key count plus aggregate count",
    std::invalid_argument);
  validate_indices(table, key_indices, value_indices);

  for (std::size_t i = 0; i < key_indices.size(); ++i) {
    auto const expected = cudf::column_view{final_output_types[i], 0, nullptr, nullptr, 0, 0, {}};
    GQE_EXPECTS(cudf::have_same_types(table.column(key_indices[i]), expected),
                "hash_aggregate_task: partial key type does not match final key type",
                std::invalid_argument);
  }
  for (std::size_t i = 0; i < value_indices.size(); ++i) {
    auto const [kind, index] = value_indices[i];
    auto const result_type   = gqe::detail::compute_target_type(table.column(index).type(), kind);
    auto const actual        = cudf::column_view{result_type, 0, nullptr, nullptr, 0, 0, {}};
    auto const expected =
      cudf::column_view{final_output_types[key_indices.size() + i], 0, nullptr, nullptr, 0, 0, {}};
    GQE_EXPECTS(cudf::have_same_types(actual, expected),
                "hash_aggregate_task: combine result type does not match declared final type",
                std::invalid_argument);
  }
}

[[nodiscard]] std::vector<cudf::groupby::aggregation_request> make_requests(
  cudf::table_view table,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(value_indices.size());
  for (auto const& [kind, index] : value_indices) {
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(make_groupby_aggregation(kind));
    requests.push_back(
      cudf::groupby::aggregation_request{table.column(index), std::move(aggregations)});
  }
  return requests;
}

[[nodiscard]] cudf::table aggregate(
  cudf::table_view table,
  std::span<cudf::size_type const> key_indices,
  std::span<std::pair<cudf::aggregation::Kind, cudf::size_type> const> value_indices,
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
  auto requests = make_requests(table, value_indices);

  groupby::groupby grouped{cudf::table_view{key_columns}};
  auto [key_result, aggregate_results] = grouped.aggregate(requests, active_mask, stream, mr);
  auto result_columns                  = key_result->release();
  for (auto& result : aggregate_results) {
    GQE_EXPECTS(result.results.size() == 1,
                "hash_aggregate_task: group-by aggregation returned an unexpected result count");
    result_columns.push_back(std::move(result.results.front()));
  }
  return cudf::table(std::move(result_columns));
}

}  // namespace

hash_partial_aggregate_task::hash_partial_aggregate_task(
  std::vector<cudf::size_type> key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices)
  : _key_indices(std::move(key_indices)), _value_indices(std::move(value_indices))
{
  validate_configuration(_key_indices, _value_indices, /*allow_count_aggregations=*/true);

  auto layout                 = detail::make_aggregate_input_layout(_key_indices, _value_indices);
  _materialized_indices       = std::move(layout.materialized_indices);
  _materialized_key_indices   = std::move(layout.materialized_key_indices);
  _materialized_value_indices = std::move(layout.materialized_value_indices);
}

std::unique_ptr<task> hash_partial_aggregate_task::clone() const
{
  return std::make_unique<hash_partial_aggregate_task>(_key_indices, _value_indices);
}

std::optional<state_container> hash_partial_aggregate_task::next(
  state_container_view inputs,
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"hash_partial_aggregate_task::next"};

  auto materialized =
    detail::materialize_aggregate_input(inputs, _materialized_indices, stream_handle, mr);
  auto const input = materialized.view();
  validate_indices(input.table, _materialized_key_indices, _materialized_value_indices);
  auto normalized = detail::normalize_active_mask(input.mask, stream_handle, mr);
  auto result     = aggregate(input.table,
                          _materialized_key_indices,
                          _materialized_value_indices,
                          normalized.view,
                          stream_handle,
                          mr);

  GQE_LOG_TRACE(
    "Apply QEP hash partial aggregate: input_rows={}, input_columns={}, "
    "input_has_valid_mask={}, output_groups={}",
    input.table.num_rows(),
    materialized.logical_column_count,
    input.has_mask,
    result.num_rows());
  auto output = state_container_builder().add_state(std::move(result), stream_handle).build();
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

hash_aggregate_task::hash_aggregate_task(
  std::vector<cudf::size_type> key_indices,
  std::vector<std::pair<cudf::aggregation::Kind, cudf::size_type>> value_indices,
  std::vector<cudf::data_type> final_output_types)
  : _key_indices(std::move(key_indices)),
    _value_indices(std::move(value_indices)),
    _final_output_types(std::move(final_output_types))
{
  validate_configuration(_key_indices,
                         _value_indices,
                         /*allow_count_aggregations=*/false,
                         _final_output_types.size());
}

std::unique_ptr<task> hash_aggregate_task::clone() const
{
  return std::make_unique<hash_aggregate_task>(_key_indices, _value_indices, _final_output_types);
}

state_container hash_aggregate_task::initialize(context_reference /* ctx_ref */,
                                                query_stream const&,
                                                rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"hash_aggregate_task::initialize"};
  GQE_LOG_TRACE("Initialize QEP hash aggregate: key_count={}, aggregate_count={}",
                _key_indices.size(),
                _value_indices.size());
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<hash_aggregate_accumulator>()})
    .build();
}

void hash_aggregate_task::next(state_container_view inputs,
                               state_container_view accumulator,
                               context_reference /* ctx_ref */,
                               query_stream const&,
                               rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"hash_aggregate_task::next"};
  auto const input = detail::resolve_aggregate_input(inputs, /*is_mask_allowed=*/false);
  validate_fold_schema(input.table, _key_indices, _value_indices, _final_output_types);
  auto& accumulator_data = get_task_private_state<hash_aggregate_accumulator>(accumulator, 0);

  {
    std::scoped_lock const guard{accumulator_data.mutex};
    if (!accumulator_data.chunks.empty()) {
      auto const first = to_table_view(accumulator_data.chunks.front());
      assert(first.has_value());
      GQE_EXPECTS(cudf::have_same_types(*first, input.table),
                  "hash_aggregate_task: partial chunk schemas do not match",
                  std::invalid_argument);
    }
    accumulator_data.chunks.push_back(make_mutable_state_copy(inputs));
    accumulator_data.input_row_count += input.table.num_rows();
  }

  GQE_LOG_TRACE("Fold QEP hash aggregate chunk: chunk_rows={}, chunk_columns={}",
                input.table.num_rows(),
                input.table.num_columns());
}

state_container hash_aggregate_task::finalize(state_container&& accumulator,
                                              context_reference /* ctx_ref */,
                                              query_stream const& stream_handle,
                                              rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"hash_aggregate_task::finalize"};
  auto& accumulator_data = get_task_private_state<hash_aggregate_accumulator>(accumulator, 0);
  if (accumulator_data.chunks.empty()) {
    GQE_LOG_TRACE("Finalize QEP hash aggregate: chunk_count=0, input_rows=0, output_groups=0");
    return make_empty_state_container(_final_output_types);
  }

  auto const chunk_count = accumulator_data.chunks.size();
  auto concatenated =
    detail::concatenate_materialized_chunks(accumulator_data.chunks, stream_handle, mr);

  auto result = aggregate(concatenated.table->view(),
                          _key_indices,
                          _value_indices,
                          cudf::column_view{},
                          stream_handle,
                          mr);

  GQE_LOG_TRACE("Finalize QEP hash aggregate: chunk_count={}, input_rows={}, output_groups={}",
                chunk_count,
                accumulator_data.input_row_count,
                result.num_rows());
  return state_container_builder().add_state(std::move(result), stream_handle).build();
}

}  // namespace qep
}  // namespace gqe
