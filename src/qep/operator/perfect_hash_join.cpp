/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/qep/operator/perfect_hash_join.hpp>

#include "join_output.hpp"
#include "materialized_chunks.hpp"

#include <gqe/executor/perfect_hash/masked_join.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>
#include <cudf/replace.hpp>
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
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

bool is_supported_perfect_hash_join_key_type(cudf::type_id type) noexcept
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

namespace {

/**
 * @brief Retains compacted build chunks until the build relation is finalized.
 *
 * `next` may append chunks concurrently from multiple host threads, each under `mutex`. `finalize`
 * runs after all `next` calls complete and therefore reads `chunks` without locking.
 */
struct build_accumulator : public task_private_state {
  std::mutex mutex;                     ///< Protects every mutation of `chunks`.
  std::vector<state_container> chunks;  ///< Complete retained chunks in append order.
};

/**
 * @brief Owns the finalized build relation and equality-key indices.
 *
 * An empty build-column container means that the build fold received no chunks. A schema-bearing
 * zero-row build retains one owning state per concatenated build column and is therefore non-empty.
 * The build relation never has a valid mask because `next` compacts masked input.
 */
class build_state : public task_private_state {
 public:
  build_state(state_container build_column_states,
              detail::join_materialized_column_layout column_layout,
              std::vector<cudf::size_type> build_key_indices)
    : _build_column_states{std::move(build_column_states)},
      _column_layout{std::move(column_layout)},
      _build_key_indices{std::move(build_key_indices)}
  {
    GQE_EXPECTS(_build_column_states.empty() ||
                  to_table_view(state_container_view(_build_column_states)).has_value(),
                "build_state: build column states must form a plain table");
  }

  /** @brief Whether the build fold received at least one chunk. */
  [[nodiscard]] bool has_build_relation() const noexcept { return !_build_column_states.empty(); }

  /** @brief Return a zero-copy view of the finalized regular build relation. */
  [[nodiscard]] cudf::table_view build_relation_view() const
  {
    auto const view = to_table_view(state_container_view(_build_column_states));
    GQE_EXPECTS(view.has_value(), "build_state: no regular build relation is available");
    return *view;
  }

  /** @brief Borrow all finalized regular build-column states in input order. */
  [[nodiscard]] state_container_view build_column_states() const noexcept
  {
    return state_container_view(_build_column_states);
  }

  [[nodiscard]] std::vector<cudf::size_type> const& build_key_indices() const noexcept
  {
    return _build_key_indices;
  }

  [[nodiscard]] detail::join_materialized_column_layout const& column_layout() const noexcept
  {
    return _column_layout;
  }

  void order_deallocation_after_stream_use(query_stream const& stream_handle) const
  {
    qep::order_deallocation_after_stream_use(_build_column_states, stream_handle);
  }

 private:
  state_container _build_column_states;
  detail::join_materialized_column_layout _column_layout;
  std::vector<cudf::size_type> _build_key_indices;
};

}  // namespace

perfect_hash_join_build_task::perfect_hash_join_build_task(
  cudf::size_type build_logical_column_count,
  std::vector<cudf::size_type> build_key_indices,
  std::vector<cudf::size_type> projected_payload_indices)
  : _build_key_indices{std::move(build_key_indices)},
    _column_layout{build_logical_column_count, _build_key_indices, projected_payload_indices}
{
  GQE_EXPECTS(!_build_key_indices.empty(),
              "perfect_hash_join_build_task: build_key_indices must be non-empty",
              std::invalid_argument);
}

std::unique_ptr<task> perfect_hash_join_build_task::clone() const
{
  return std::make_unique<perfect_hash_join_build_task>(*this);
}

state_container perfect_hash_join_build_task::initialize(context_reference,
                                                         query_stream const&,
                                                         rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::perfect_hash_join_build_task::initialize"};
  return state_container_builder()
    .add_state(state_kind::task_private{std::make_unique<build_accumulator>()})
    .build();
}

void perfect_hash_join_build_task::next(state_container_view inputs,
                                        state_container_view accumulator,
                                        context_reference,
                                        query_stream const& stream_handle,
                                        rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::perfect_hash_join_build_task::next"};
  GQE_EXPECTS(accumulator.size() == 1,
              "perfect_hash_join_build_task::next: expected exactly one accumulator state");
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "perfect_hash_join_build_task::next: input must be a logical relation");
  GQE_EXPECTS(logical->num_columns() == _column_layout.logical_column_count(),
              "perfect_hash_join_build_task::next: input logical width differs from configuration");

  // Compacting here keeps masked-out rows out of the broadcast input that every probe reads.
  column_materialization materialized =
    materialize_compacted_columns(*logical, _column_layout.logical_indices(), stream_handle, mr);
  auto chunk  = std::move(materialized).release_column_states();
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);
  std::scoped_lock const guard{state.mutex};
  state.chunks.push_back(std::move(chunk));
}

state_container perfect_hash_join_build_task::finalize(state_container&& accumulator,
                                                       context_reference,
                                                       query_stream const& stream_handle,
                                                       rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::perfect_hash_join_build_task::finalize"};
  GQE_EXPECTS(accumulator.size() == 1,
              "perfect_hash_join_build_task::finalize: expected exactly one accumulator state");
  auto& state       = get_task_private_state<build_accumulator>(accumulator, 0);
  auto concatenated = detail::concatenate_materialized_chunks(state.chunks, stream_handle, mr);
  state_container build_column_states;
  if (concatenated.table.has_value()) {
    build_column_states =
      state_container_builder().add_state(std::move(*concatenated.table), stream_handle).build();
  }

  auto finalized_state = std::make_unique<build_state>(
    std::move(build_column_states), _column_layout, _build_key_indices);
  return state_container_builder()
    .add_state(state_kind::task_private{std::move(finalized_state)})
    .build();
}

perfect_hash_join_probe_task::perfect_hash_join_probe_task(
  std::vector<cudf::size_type> probe_key_indices,
  perfect_hash_join_build_input build_input,
  std::vector<cudf::size_type> projection_indices,
  std::vector<cudf::data_type> output_data_types)
  : _probe_key_indices{std::move(probe_key_indices)},
    _build_input{build_input},
    _projection_indices{std::move(projection_indices)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(!_probe_key_indices.empty(),
              "perfect_hash_join_probe_task: probe_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_probe_key_indices, [](auto index) { return index >= 0; }),
              "perfect_hash_join_probe_task: key indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_projection_indices, [](auto index) { return index >= 0; }),
              "perfect_hash_join_probe_task: projection indices must be non-negative",
              std::invalid_argument);
  GQE_EXPECTS(_projection_indices.size() == _output_data_types.size(),
              "perfect_hash_join_probe_task: projection and output type counts must match",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_output_data_types,
                                  [](cudf::data_type type) {
                                    return cudf::is_fixed_width(type) ||
                                           type.id() == cudf::type_id::STRING;
                                  }),
              "perfect_hash_join_probe_task: output types must be fixed-width or STRING",
              std::invalid_argument);
}

std::unique_ptr<task> perfect_hash_join_probe_task::clone() const
{
  return std::make_unique<perfect_hash_join_probe_task>(
    _probe_key_indices, _build_input, _projection_indices, _output_data_types);
}

state_container perfect_hash_join_probe_task::initialize(state_container_view inputs,
                                                         context_reference,
                                                         query_stream const&,
                                                         rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::perfect_hash_join_probe_task::initialize"};
  GQE_EXPECTS(inputs.size() == 1,
              "perfect_hash_join_probe_task::initialize: expected exactly one build state");
  std::ignore = get_task_private_state<build_state>(inputs, 0);
  return make_mutable_state_copy(inputs);
}

std::optional<state_container> perfect_hash_join_probe_task::next(
  state_container_view inputs,
  state_container_view accumulator,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::perfect_hash_join_probe_task::next"};
  GQE_EXPECTS(accumulator.size() == 1,
              "perfect_hash_join_probe_task::next: expected exactly one build state");
  auto const logical_probe = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical_probe.has_value(),
              "perfect_hash_join_probe_task::next: input must be a logical relation");
  auto const& build = get_task_private_state<build_state>(accumulator, 0);
  if (!build.has_build_relation()) {
    return _output_data_types.empty() ? make_row_count_container(0)
                                      : make_empty_state_container(_output_data_types);
  }

  auto const build_view          = build.build_relation_view();
  auto const probe_column_layout = detail::join_materialized_column_layout{
    logical_probe->num_columns(), _probe_key_indices, std::span<cudf::size_type const>{}};
  GQE_EXPECTS(build.build_key_indices().size() == _probe_key_indices.size(),
              "perfect_hash_join_probe_task::next: build/probe key counts differ");
  detail::validate_join_projection_types(
    _projection_indices, _output_data_types, build_view, build.column_layout(), *logical_probe);

  column_materialization materialized_probe_keys =
    materialize_columns(*logical_probe, probe_column_layout.logical_indices(), stream_handle, mr);
  auto const build_keys = build_view.select(build.column_layout().remap(build.build_key_indices()));
  auto const probe_keys =
    materialized_probe_keys.table_view().select(probe_column_layout.remap(_probe_key_indices));
  auto const perfect_hash_build_keys =
    _build_input == perfect_hash_join_build_input::materialized ? build_keys : probe_keys;
  GQE_EXPECTS(std::ranges::all_of(perfect_hash_build_keys,
                                  [](cudf::column_view const& key) {
                                    return is_supported_perfect_hash_join_key_type(key.type().id());
                                  }),
              "perfect_hash_join_probe_task::next: build key type is unsupported",
              std::invalid_argument);
  GQE_EXPECTS(cudf::have_same_types(build_keys, probe_keys),
              "perfect_hash_join_probe_task::next: build/probe key types differ");

  shared_state normalized_probe_mask_state;
  auto probe_mask = logical_probe->valid_mask().value_or(cudf::column_view{});
  if (logical_probe->valid_mask().has_value() && probe_mask.nullable()) {
    cudf::numeric_scalar<bool> const false_value{false, true, stream, mr};
    normalized_probe_mask_state = make_shared_state(
      state_kind::valid_mask{cudf::replace_nulls(probe_mask, false_value, stream, mr)},
      stream_handle);
    probe_mask = *to_valid_mask_view(normalized_probe_mask_state);
  }
  // The build relation is compacted, so only the probe side can carry a mask.
  cudf::column_view const build_mask{};
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_matches;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_matches;
  if (_build_input == perfect_hash_join_build_input::materialized) {
    std::tie(build_matches, probe_matches) =
      perfect_hash::perfect_join(build_keys, probe_keys, build_mask, probe_mask, stream, mr);
  } else {
    std::tie(probe_matches, build_matches) =
      perfect_hash::perfect_join(probe_keys, build_keys, probe_mask, build_mask, stream, mr);
  }
  GQE_EXPECTS(build_matches != nullptr,
              "perfect_hash_join_probe_task::next: the perfect hash returned null build "
              "matches");
  GQE_EXPECTS(probe_matches != nullptr,
              "perfect_hash_join_probe_task::next: the perfect hash returned null probe "
              "matches");

  GQE_EXPECTS(probe_matches->size() == build_matches->size(),
              "perfect_hash_join_probe_task::next: match-vector sizes differ");
  GQE_EXPECTS(
    build_matches->size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "perfect_hash_join_probe_task::next: match count exceeds cudf::size_type",
    std::overflow_error);
  auto const match_count = static_cast<cudf::size_type>(build_matches->size());
  if (_projection_indices.empty()) {
    build.order_deallocation_after_stream_use(stream_handle);
    order_deallocation_after_stream_use(materialized_probe_keys.states(), stream_handle);
    if (normalized_probe_mask_state != nullptr) {
      state_container const normalized_mask{normalized_probe_mask_state};
      order_deallocation_after_stream_use(normalized_mask, stream_handle);
    }
    order_deallocation_after_stream_use(inputs, stream_handle);
    return make_row_count_container(match_count);
  }

  auto build_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*build_matches), rmm::device_buffer{}, 0),
    stream_handle);
  auto probe_positions = make_position_list(
    std::make_unique<cudf::column>(std::move(*probe_matches), rmm::device_buffer{}, 0),
    stream_handle);

  auto output = detail::make_join_indirect_output(_projection_indices,
                                                  build.build_column_states(),
                                                  build.column_layout(),
                                                  *logical_probe,
                                                  build_positions,
                                                  probe_positions,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream_handle,
                                                  mr);
  build.order_deallocation_after_stream_use(stream_handle);
  order_deallocation_after_stream_use(materialized_probe_keys.states(), stream_handle);
  if (normalized_probe_mask_state != nullptr) {
    state_container const normalized_mask{normalized_probe_mask_state};
    order_deallocation_after_stream_use(normalized_mask, stream_handle);
  }
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace qep
}  // namespace gqe
