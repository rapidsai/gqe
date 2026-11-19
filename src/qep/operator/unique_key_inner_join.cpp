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

#include <gqe/qep/operator/unique_key_inner_join.hpp>

#include "join_output.hpp"
#include "materialized_chunks.hpp"

#include <gqe/executor/unique_key_inner_join.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

// ----- Build state -----

/**
 * @brief Retains build chunks until the fold is finalized.
 *
 * `mutex` protects `chunks`. Each `next` call prepares and validates one complete chunk before
 * locking, then atomically appends that chunk. The fold lifecycle guarantees that `finalize` runs
 * only after all `next` calls return, so finalization reads `chunks` without the mutex.
 */
struct build_accumulator : public task_private_state {
  std::mutex mutex;                     ///< Protects every mutation of `chunks`.
  std::vector<state_container> chunks;  ///< Complete retained chunks in append order.
};

/**
 * @brief Owns the finalized build payload and the hash join that views its selected keys.
 *
 * The payload states precede the cached join in declaration order, ensuring
 * that destruction releases the viewing hash join before its backing columns. After construction
 * the state is immutable, so multiple probe tasks may share it concurrently.
 */
class build_state : public task_private_state {
 public:
  build_state(state_container build_column_states,
              detail::join_materialized_column_layout column_layout,
              std::span<cudf::size_type const> build_key_indices,
              cudf::null_equality null_equality,
              bool is_hash_map_cache_enabled,
              query_stream const& stream_handle,
              rmm::device_async_resource_ref mr)
    : _build_column_states{std::move(build_column_states)},
      _column_layout{std::move(column_layout)},
      _null_equality{null_equality}
  {
    auto const stream = stream_handle.ref();
    if (_build_column_states.empty()) { return; }

    auto const relation = to_table_view(state_container_view(_build_column_states));
    GQE_EXPECTS(relation.has_value(),
                "unique_key_inner_join build_state: states must form a plain table");
    _build_keys = relation->select(_column_layout.remap(build_key_indices));

    if (!is_hash_map_cache_enabled) { return; }

    _cached_join = std::make_unique<gqe::unique_key_join>(
      _build_keys, cudf::column_view{}, _null_equality, stream, mr);
  }

  [[nodiscard]] bool has_build_relation() const noexcept { return !_build_column_states.empty(); }

  [[nodiscard]] cudf::table_view build_relation_view() const
  {
    auto const relation = to_table_view(state_container_view(_build_column_states));
    GQE_EXPECTS(relation.has_value(),
                "unique_key_inner_join build_state: no build relation is available");
    return *relation;
  }

  [[nodiscard]] cudf::table_view build_keys() const noexcept { return _build_keys; }

  [[nodiscard]] detail::join_materialized_column_layout const& column_layout() const noexcept
  {
    return _column_layout;
  }

  [[nodiscard]] cudf::null_equality null_equality() const noexcept { return _null_equality; }

  [[nodiscard]] auto const* cached_join() const noexcept { return _cached_join.get(); }

  [[nodiscard]] state_container_view build_column_states() const noexcept
  {
    return state_container_view(_build_column_states);
  }

  void order_deallocation_after_stream_use(query_stream const& stream_handle) const
  {
    qep::order_deallocation_after_stream_use(_build_column_states, stream_handle);
  }

 private:
  // Declaration order is intentional. The cached join must be destroyed before the columns
  // that back `_build_keys`.
  state_container _build_column_states;
  detail::join_materialized_column_layout _column_layout;
  cudf::table_view _build_keys;
  cudf::null_equality _null_equality;
  std::unique_ptr<gqe::unique_key_join> _cached_join;
};

}  // namespace

unique_key_inner_join_build_task::unique_key_inner_join_build_task(
  cudf::size_type build_logical_column_count,
  std::vector<cudf::size_type> build_key_indices,
  std::vector<cudf::size_type> projected_payload_indices,
  cudf::null_equality null_equality,
  bool is_hash_map_cache_enabled,
  std::vector<cudf::data_type> build_key_types)
  : _build_key_indices{std::move(build_key_indices)},
    _column_layout{build_logical_column_count, _build_key_indices, projected_payload_indices},
    _null_equality{null_equality},
    _is_hash_map_cache_enabled{is_hash_map_cache_enabled},
    _build_key_types{std::move(build_key_types)}
{
  GQE_EXPECTS(_build_key_types.size() == _build_key_indices.size(),
              "unique_key_inner_join_build_task: one type is required per key",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_build_key_types,
                                  [](auto type) { return gqe::unique_key_join_supported(type); }),
              "unique_key_inner_join_build_task: unsupported key type",
              std::invalid_argument);

  GQE_EXPECTS(!_build_key_indices.empty(),
              "unique_key_inner_join_build_task: build_key_indices must be non-empty",
              std::invalid_argument);
}

std::unique_ptr<task> unique_key_inner_join_build_task::clone() const
{
  return std::make_unique<unique_key_inner_join_build_task>(*this);
}

state_container unique_key_inner_join_build_task::initialize(context_reference,
                                                             query_stream const&,
                                                             rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::unique_key_inner_join_build_task::initialize"};
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<build_accumulator>()})
    .build();
}

void unique_key_inner_join_build_task::next(state_container_view inputs,
                                            state_container_view accumulator,
                                            context_reference,
                                            query_stream const& stream_handle,
                                            rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::unique_key_inner_join_build_task::next"};
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "unique_key_inner_join_build_task::next: input must be a logical relation");
  GQE_EXPECTS(
    logical->num_columns() == _column_layout.logical_column_count(),
    "unique_key_inner_join_build_task::next: input logical width differs from configuration");
  GQE_EXPECTS(accumulator.size() == 1,
              "unique_key_inner_join_build_task::next: expected exactly one accumulator state");

  GQE_EXPECTS(std::ranges::equal(_build_key_indices,
                                 _build_key_types,
                                 {},
                                 [&](auto index) { return logical->column_type(index); }),
              "unique_key_inner_join_build_task: input key types differ from configuration");

  column_materialization materialized =
    materialize_compacted_columns(*logical, _column_layout.logical_indices(), stream_handle, mr);
  auto chunk  = std::move(materialized).release_column_states();
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);
  {
    std::scoped_lock const guard{state.mutex};
    state.chunks.push_back(std::move(chunk));
  }
}

state_container unique_key_inner_join_build_task::finalize(state_container&& accumulator,
                                                           context_reference,
                                                           query_stream const& stream_handle,
                                                           rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::unique_key_inner_join_build_task::finalize"};
  GQE_EXPECTS(accumulator.size() == 1,
              "unique_key_inner_join_build_task::finalize: expected exactly one accumulator state");
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);

  auto concatenated = detail::concatenate_materialized_chunks(state.chunks, stream_handle, mr);
  state_container build_column_states;
  if (concatenated.table.has_value()) {
    build_column_states =
      state_container_builder().add_state(std::move(*concatenated.table), stream_handle).build();
  }

  auto finalized = std::make_unique<build_state>(std::move(build_column_states),
                                                 _column_layout,
                                                 _build_key_indices,
                                                 _null_equality,
                                                 _is_hash_map_cache_enabled,
                                                 stream_handle,
                                                 mr);
  return state_container_builder()
    .add_state(state_kind::task_private{std::move(finalized)})
    .build();
}

unique_key_inner_join_probe_task::unique_key_inner_join_probe_task(
  std::vector<cudf::size_type> probe_key_indices,
  std::vector<cudf::size_type> projection_indices,
  std::vector<cudf::data_type> output_data_types)
  : _probe_key_indices{std::move(probe_key_indices)},
    _projection_indices{std::move(projection_indices)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(!_probe_key_indices.empty(),
              "unique_key_inner_join_probe_task: probe_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_probe_key_indices, [](auto index) { return index >= 0; }),
              "unique_key_inner_join_probe_task: key indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_projection_indices, [](auto index) { return index >= 0; }),
              "unique_key_inner_join_probe_task: projection indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(_projection_indices.size() == _output_data_types.size(),
              "unique_key_inner_join_probe_task: projection and output type counts must match",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_output_data_types,
                                  [](cudf::data_type type) {
                                    return cudf::is_fixed_width(type) ||
                                           type.id() == cudf::type_id::STRING;
                                  }),
              "unique_key_inner_join_probe_task: output types must be fixed-width or STRING",
              std::invalid_argument);
}

std::unique_ptr<task> unique_key_inner_join_probe_task::clone() const
{
  return std::make_unique<unique_key_inner_join_probe_task>(
    _probe_key_indices, _projection_indices, _output_data_types);
}

state_container unique_key_inner_join_probe_task::initialize(state_container_view inputs,
                                                             context_reference,
                                                             query_stream const&,
                                                             rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::unique_key_inner_join_probe_task::initialize"};
  GQE_EXPECTS(inputs.size() == 1,
              "unique_key_inner_join_probe_task::initialize: expected exactly one build state");
  std::ignore = get_task_private_state<build_state>(inputs, 0);
  return make_mutable_state_copy(inputs);
}

std::optional<state_container> unique_key_inner_join_probe_task::next(
  state_container_view inputs,
  state_container_view accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::unique_key_inner_join_probe_task::next"};
  auto const logical_probe = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical_probe.has_value(),
              "unique_key_inner_join_probe_task::next: input must be a logical relation");
  GQE_EXPECTS(accumulator.size() == 1,
              "unique_key_inner_join_probe_task::next: expected exactly one build state");
  auto const& build = get_task_private_state<build_state>(accumulator, 0);
  if (!build.has_build_relation()) {
    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  auto const probe_column_layout = detail::join_materialized_column_layout{
    logical_probe->num_columns(), _probe_key_indices, std::span<cudf::size_type const>{}};
  auto const build_columns = build.build_relation_view();
  detail::validate_join_projection_types(
    _projection_indices, _output_data_types, build_columns, build.column_layout(), *logical_probe);
  auto const has_probe_projection = std::ranges::any_of(_projection_indices, [&](auto index) {
    return index >= build.column_layout().logical_column_count();
  });
  column_materialization materialized_probe = materialize_compacted_columns(
    *logical_probe, probe_column_layout.logical_indices(), has_probe_projection, stream_handle, mr);
  auto const probe_columns = materialized_probe.table_view();
  auto const probe_keys    = probe_columns.select(probe_column_layout.remap(_probe_key_indices));
  GQE_EXPECTS(build.build_keys().num_columns() == probe_keys.num_columns(),
              "unique_key_inner_join_probe_task::next: build/probe key counts differ");
  GQE_EXPECTS(cudf::have_same_types(build.build_keys(), probe_keys),
              "unique_key_inner_join_probe_task::next: build/probe key types differ");

  std::unique_ptr<gqe::unique_key_join> local_join;
  auto const* join = build.cached_join();
  if (join == nullptr) {
    local_join = std::make_unique<gqe::unique_key_join>(
      build.build_keys(), cudf::column_view{}, build.null_equality(), stream, mr);

    join = local_join.get();
  }

  // The join returns [probe, build] positions in compacted key coordinates.
  auto [probe_matches, build_matches] =
    join->inner_join(probe_keys, cudf::column_view{}, stream, mr);
  GQE_EXPECTS(probe_matches != nullptr && build_matches != nullptr,
              "unique_key_inner_join_probe_task::next: hash join returned null match vectors");
  GQE_EXPECTS(probe_matches->size() == build_matches->size(),
              "unique_key_inner_join_probe_task::next: match-vector sizes differ");

  if (_projection_indices.empty()) {
    auto const match_count = static_cast<cudf::size_type>(probe_matches->size());
    build.order_deallocation_after_stream_use(stream_handle);
    order_deallocation_after_stream_use(materialized_probe.states(), stream_handle);
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(match_count);
  }

  auto build_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*build_matches), rmm::device_buffer{}, 0),
    stream_handle);
  auto probe_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*probe_matches), rmm::device_buffer{}, 0),
    stream_handle);

  auto const build_policy = cudf::out_of_bounds_policy::DONT_CHECK;
  // Join match positions address compacted keys. Restore the original logical row coordinates
  // before composing them with existing payload indirection.
  if (auto const& rows = materialized_probe.row_positions(); rows != nullptr) {
    probe_positions = compose_position_lists(rows, probe_positions, stream_handle, mr);
  }
  auto output = detail::make_join_indirect_output(_projection_indices,
                                                  build.build_column_states(),
                                                  build.column_layout(),
                                                  *logical_probe,
                                                  build_positions,
                                                  probe_positions,
                                                  build_policy,
                                                  stream_handle,
                                                  mr);
  build.order_deallocation_after_stream_use(stream_handle);
  order_deallocation_after_stream_use(materialized_probe.states(), stream_handle);
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace qep
}  // namespace gqe
