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

#include <gqe/qep/operator/equality_semi_anti_join.hpp>

#include "join_output.hpp"

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::qep {

namespace {

// ----- Build state -----

/**
 * @brief Retain materialized build keys until finalization.
 *
 * `mutex` protects every mutation of `chunks`. Each `next` call materializes its selected key
 * states before locking, then appends them as one atomic transition. The fold
 * lifecycle guarantees that finalization begins only after all `next` calls return.
 */
struct build_accumulator : public task_private_state {
  std::mutex mutex;                     ///< Protects every mutation of `chunks`.
  std::vector<state_container> chunks;  ///< Retained key chunks.
};

/**
 * @brief Own the finalized build keys and the filtered join that views them.
 *
 * An empty state container means that the fold received no chunks. A schema-bearing zero-row
 * build owns one state per configured key and constructs `_join`. Declaration order ensures that
 * `_join` is destroyed before the retained build key columns.
 *
 * When `has_filtered_join()` is true, `_build_key_types` contains exactly `_build_key_count`
 * entries.
 */
class build_state : public task_private_state {
 public:
  build_state(state_container build_key_states,
              std::size_t build_key_count,
              cudf::null_equality null_equality,
              query_stream const& stream_handle,
              rmm::device_async_resource_ref mr)
    : _build_key_states{std::move(build_key_states)}, _build_key_count{build_key_count}
  {
    auto const stream = stream_handle.ref();
    if (_build_key_states.empty()) { return; }

    auto const keys = to_table_view(state_container_view(_build_key_states));
    GQE_EXPECTS(keys.has_value(),
                "equality_semi_anti_join build_state: states must form a plain table");
    GQE_EXPECTS(static_cast<std::size_t>(keys->num_columns()) == _build_key_count,
                "equality_semi_anti_join build_state: key count differs from configured count");
    _build_key_types.reserve(_build_key_count);
    for (auto const& column : *keys) {
      _build_key_types.push_back(column.type());
    }
    _join = std::make_unique<cudf::filtered_join>(*keys, null_equality, stream, mr);
  }

  void order_deallocation_after_stream_use(query_stream const& stream_handle) const
  {
    qep::order_deallocation_after_stream_use(_build_key_states, stream_handle);
  }

  [[nodiscard]] bool has_filtered_join() const noexcept { return _join != nullptr; }

  [[nodiscard]] std::size_t build_key_count() const noexcept { return _build_key_count; }

  [[nodiscard]] std::span<cudf::data_type const> build_key_types() const noexcept
  {
    return _build_key_types;
  }

  [[nodiscard]] cudf::filtered_join const& join() const
  {
    GQE_EXPECTS(_join != nullptr,
                "equality_semi_anti_join build_state: no filtered join is available");
    return *_join;
  }

 private:
  // Declaration order is intentional. `_join` must be destroyed before its backing columns.
  state_container _build_key_states;
  std::size_t _build_key_count;
  std::vector<cudf::data_type> _build_key_types;
  std::unique_ptr<cudf::filtered_join> _join;
};

// ----- Probing -----

[[nodiscard]] std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_filtered_join(
  cudf::filtered_join const& join,
  cudf::table_view probe_keys,
  join_type_type join_type,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  switch (join_type) {
    case join_type_type::left_semi: return join.semi_join(probe_keys, stream, mr);
    case join_type_type::left_anti: return join.anti_join(probe_keys, stream, mr);
    default: throw std::logic_error("equality_semi_anti_join_probe_task: unsupported join type");
  }
}

[[nodiscard]] std::unique_ptr<cudf::column> make_no_build_positions(
  cudf::size_type compacted_probe_row_count,
  join_type_type join_type,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  switch (join_type) {
    case join_type_type::left_semi:
      return std::make_unique<cudf::column>(
        rmm::device_uvector<cudf::size_type>(0, stream, mr), rmm::device_buffer(), 0);
    case join_type_type::left_anti: {
      cudf::numeric_scalar<cudf::size_type> const zero(0, true, stream, mr);
      return cudf::sequence(compacted_probe_row_count, zero, stream, mr);
    }
    default: throw std::logic_error("equality_semi_anti_join_probe_task: unsupported join type");
  }
}

}  // namespace

equality_semi_anti_join_build_task::equality_semi_anti_join_build_task(
  std::vector<cudf::size_type> build_key_indices, cudf::null_equality null_equality)
  : _build_key_indices{std::move(build_key_indices)}, _null_equality{null_equality}
{
  GQE_EXPECTS(!_build_key_indices.empty(),
              "equality_semi_anti_join_build_task: build_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_build_key_indices, [](auto index) { return index >= 0; }),
              "equality_semi_anti_join_build_task: key indices must be non-negative",
              std::invalid_argument);
}

std::unique_ptr<task> equality_semi_anti_join_build_task::clone() const
{
  return std::make_unique<equality_semi_anti_join_build_task>(_build_key_indices, _null_equality);
}

state_container equality_semi_anti_join_build_task::initialize(context_reference,
                                                               query_stream const&,
                                                               rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::equality_semi_anti_join_build_task::initialize"};
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<build_accumulator>()})
    .build();
}

void equality_semi_anti_join_build_task::next(state_container_view inputs,
                                              state_container_view accumulator,
                                              context_reference,
                                              query_stream const& stream_handle,
                                              rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::equality_semi_anti_join_build_task::next"};
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "equality_semi_anti_join_build_task::next: input must be a logical relation");
  GQE_EXPECTS(accumulator.size() == 1,
              "equality_semi_anti_join_build_task::next: expected exactly one accumulator state");
  GQE_EXPECTS(std::ranges::all_of(_build_key_indices,
                                  [&](auto index) { return index < logical->num_columns(); }),
              "equality_semi_anti_join_build_task::next: key index is out of range",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::all_of(_build_key_indices,
                        [&](auto index) { return !cudf::is_nested(logical->column_type(index)); }),
    "equality_semi_anti_join_build_task::next: nested key types are not supported",
    std::invalid_argument);
  auto materialized =
    materialize_compacted_columns(*logical, _build_key_indices, stream_handle, mr);
  auto key_chunk = std::move(materialized).release_column_states();
  auto& state    = get_task_private_state<build_accumulator>(accumulator, 0);
  std::scoped_lock const guard{state.mutex};
  state.chunks.push_back(std::move(key_chunk));
}

state_container equality_semi_anti_join_build_task::finalize(
  state_container&& accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::equality_semi_anti_join_build_task::finalize"};
  GQE_EXPECTS(
    accumulator.size() == 1,
    "equality_semi_anti_join_build_task::finalize: expected exactly one accumulator state");
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);

  state_container build_key_states;
  if (!state.chunks.empty()) {
    std::vector<cudf::table_view> key_views;
    key_views.reserve(state.chunks.size());
    for (auto const& chunk : state.chunks) {
      auto const plain = to_table_view(state_container_view(chunk));
      GQE_EXPECTS(plain.has_value(),
                  "equality_semi_anti_join_build_task::finalize: retained chunk is not a table");
      key_views.push_back(*plain);
    }
    auto concatenated = cudf::concatenate(key_views, stream, mr);
    build_key_states =
      state_container_builder().add_state(std::move(*concatenated), stream_handle).build();
  }
  state_container release_batch;
  for (auto& chunk : state.chunks) {
    release_batch.insert(release_batch.end(),
                         std::make_move_iterator(chunk.begin()),
                         std::make_move_iterator(chunk.end()));
  }
  state.chunks.clear();
  order_deallocation_after_stream_use(release_batch, stream_handle);
  release_batch.clear();

  auto finalized = std::make_unique<build_state>(
    std::move(build_key_states), _build_key_indices.size(), _null_equality, stream_handle, mr);
  return state_container_builder()
    .add_state(state_kind::task_private{std::move(finalized)})
    .build();
}

equality_semi_anti_join_probe_task::equality_semi_anti_join_probe_task(
  std::vector<cudf::size_type> probe_key_indices,
  join_type_type join_type,
  std::vector<cudf::size_type> projection_indices,
  std::vector<cudf::data_type> output_data_types)
  : _probe_key_indices{std::move(probe_key_indices)},
    _join_type{join_type},
    _projection_indices{std::move(projection_indices)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(!_probe_key_indices.empty(),
              "equality_semi_anti_join_probe_task: probe_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_probe_key_indices, [](auto index) { return index >= 0; }),
              "equality_semi_anti_join_probe_task: key indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(
    _join_type == join_type_type::left_semi || _join_type == join_type_type::left_anti,
    "equality_semi_anti_join_probe_task: only left-semi and left-anti joins are supported",
    std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_projection_indices, [](auto index) { return index >= 0; }),
              "equality_semi_anti_join_probe_task: projection indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(_projection_indices.size() == _output_data_types.size(),
              "equality_semi_anti_join_probe_task: projection and output type counts must match",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_output_data_types,
                                  [](cudf::data_type type) {
                                    return cudf::is_fixed_width(type) ||
                                           type.id() == cudf::type_id::STRING;
                                  }),
              "equality_semi_anti_join_probe_task: output types must be fixed-width or STRING",
              std::invalid_argument);
}

std::unique_ptr<task> equality_semi_anti_join_probe_task::clone() const
{
  return std::make_unique<equality_semi_anti_join_probe_task>(
    _probe_key_indices, _join_type, _projection_indices, _output_data_types);
}

state_container equality_semi_anti_join_probe_task::initialize(state_container_view inputs,
                                                               context_reference,
                                                               query_stream const&,
                                                               rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::equality_semi_anti_join_probe_task::initialize"};
  GQE_EXPECTS(inputs.size() == 1,
              "equality_semi_anti_join_probe_task::initialize: expected exactly one build state");
  std::ignore = get_task_private_state<build_state>(inputs, 0);
  return make_mutable_state_copy(inputs);
}

std::optional<state_container> equality_semi_anti_join_probe_task::next(
  state_container_view inputs,
  state_container_view accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::equality_semi_anti_join_probe_task::next"};
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "equality_semi_anti_join_probe_task::next: input must be a logical relation");
  GQE_EXPECTS(accumulator.size() == 1,
              "equality_semi_anti_join_probe_task::next: expected exactly one build state");
  auto const& build = get_task_private_state<build_state>(accumulator, 0);
  GQE_EXPECTS(std::ranges::all_of(_probe_key_indices,
                                  [&](auto index) { return index < logical->num_columns(); }),
              "equality_semi_anti_join_probe_task::next: key index is out of range",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::all_of(_probe_key_indices,
                        [&](auto index) { return !cudf::is_nested(logical->column_type(index)); }),
    "equality_semi_anti_join_probe_task::next: nested key types are not supported",
    std::invalid_argument);
  detail::join_materialized_column_layout const empty_build_layout(0, {}, {});
  detail::validate_join_projection_types(
    _projection_indices, _output_data_types, cudf::table_view{}, empty_build_layout, *logical);
  auto materialized = materialize_compacted_columns(
    *logical, _probe_key_indices, !_projection_indices.empty(), stream_handle, mr);
  auto const input_keys = materialized.table_view();
  GQE_EXPECTS(build.build_key_count() == static_cast<std::size_t>(input_keys.num_columns()),
              "equality_semi_anti_join_probe_task::next: build/probe key counts differ");
  if (build.has_filtered_join()) {
    // Nested keys are rejected above, so comparing `cudf::data_type` values is sufficient.
    GQE_EXPECTS(
      std::ranges::all_of(std::views::iota(cudf::size_type{0}, input_keys.num_columns()),
                          [&](auto index) {
                            return build.build_key_types()[static_cast<std::size_t>(index)] ==
                                   input_keys.column(index).type();
                          }),
      "equality_semi_anti_join_probe_task::next: build/probe key types differ");
  }

  std::unique_ptr<cudf::column> compacted_probe_positions;
  if (build.has_filtered_join()) {
    auto positions = probe_filtered_join(build.join(), input_keys, _join_type, stream_handle, mr);
    GQE_EXPECTS(positions != nullptr,
                "equality_semi_anti_join_probe_task::next: cuDF returned null positions");
    compacted_probe_positions = std::make_unique<cudf::column>(
      std::move(*positions), rmm::device_buffer(), cudf::size_type{0});
  } else {
    compacted_probe_positions =
      make_no_build_positions(input_keys.num_rows(), _join_type, stream_handle, mr);
  }

  build.order_deallocation_after_stream_use(stream_handle);
  order_deallocation_after_stream_use(materialized.states(), stream_handle);
  if (_projection_indices.empty()) {
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(compacted_probe_positions->size());
  }

  auto positions = make_position_list(std::move(compacted_probe_positions), stream_handle);
  // Match positions address compacted probe keys. Restore the original logical row coordinates
  // before composing them with existing payload indirection.
  if (auto const& rows = materialized.row_positions(); rows != nullptr) {
    positions = compose_position_lists(rows, positions, stream_handle, mr);
  }
  auto output = detail::make_join_indirect_output(_projection_indices,
                                                  {},
                                                  empty_build_layout,
                                                  *logical,
                                                  positions,
                                                  positions,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream_handle,
                                                  mr);
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace gqe::qep
