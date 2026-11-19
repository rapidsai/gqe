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

#include "../join_output.hpp"
#include "../materialized_chunks.hpp"
#include "launchers.hpp"

#include <gqe/expression/utility.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/operator/mark_join.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {
namespace {

namespace core = mark_join_detail;

// ----- Input validation -----

/** @brief Validate and select numeric equality-key columns. */
[[nodiscard]] cudf::table_view select_keys(cudf::table_view table,
                                           std::span<cudf::size_type const> keys)
{
  for (auto i : keys) {
    GQE_EXPECTS(i >= 0 && i < table.num_columns(),
                "mark_join select_keys: key index is out of range",
                std::invalid_argument);
    GQE_EXPECTS(cudf::is_numeric(table.column(i).type()),
                "mark_join select_keys: keys must be numeric",
                std::invalid_argument);
  }
  return table.select(keys.begin(), keys.end());
}

/** @brief Validate keys using metadata, without materializing unrelated payload columns. */
void validate_keys(logical_table_view input, std::span<cudf::size_type const> keys)
{
  for (auto index : keys) {
    GQE_EXPECTS(index >= 0 && index < input.num_columns(),
                "mark join key index is out of range",
                std::invalid_argument);
    GQE_EXPECTS(cudf::is_numeric(input.column_type(index)),
                "mark join keys must be numeric",
                std::invalid_argument);
  }
}

// ----- Build state -----

/** @brief Pair original payload owners with the selectively materialized join columns. */
struct build_chunk {
  state_container source;        ///< Original logical columns and optional mask.
  state_container materialized;  ///< Keys and residual operands, preserving row geometry.
};

/**
 * @brief Retain complete chunks until finalization.
 *
 * The fold lifecycle orders all appends before finalization reads the chunks without locking.
 */
struct build_accumulator : public task_private_state {
  std::mutex mutex;  ///< Protects atomic appends of complete chunks.
  std::vector<build_chunk> chunks;
};

/** @brief Retain original payload chunks and selected build columns for probing and scanning. */
class build_state : public task_private_state {
 public:
  build_state(std::vector<state_container> chunks,
              state_container columns,
              std::vector<cudf::size_type> keys,
              std::vector<cudf::size_type> predicate_columns,
              cudf::null_equality null_equality)
    : _chunks{std::move(chunks)},
      _build_column_states{std::move(columns)},
      _build_key_indices{std::move(keys)},
      _null_equality{null_equality}
  {
    cudf::size_type offset = 0;
    for (auto const& chunk : _chunks) {
      _chunk_offsets.push_back(offset);
      offset += logical_table_view::try_from(chunk)->row_count();
    }
    if (!has_build_relation()) { return; }
    auto const source = logical_table_view::try_from(_chunks.front());
    _layout.emplace(source->num_columns(), _build_key_indices, predicate_columns);
  }

  [[nodiscard]] bool has_build_relation() const noexcept { return !_chunks.empty(); }

  [[nodiscard]] cudf::table_view build_relation_view() const
  {
    auto const masked = masked_table_view::try_from(_build_column_states);
    if (masked) { return masked->columns; }
    auto const relation = to_table_view(_build_column_states);
    GQE_EXPECTS(relation.has_value(), "mark join build relation is unavailable");
    return *relation;
  }

  [[nodiscard]] std::optional<cudf::column_view> mask() const
  {
    return logical_table_view::try_from(_build_column_states)->valid_mask();
  }

  [[nodiscard]] detail::join_materialized_column_layout const& layout() const { return *_layout; }
  [[nodiscard]] cudf::table_view build_keys() const
  {
    return select_keys(build_relation_view(), layout().remap(_build_key_indices));
  }
  [[nodiscard]] std::span<state_container const> chunks() const { return _chunks; }
  [[nodiscard]] cudf::size_type chunk_offset(std::size_t index) const
  {
    return _chunk_offsets.at(index);
  }
  [[nodiscard]] state_container_view build_column_states() const { return _build_column_states; }
  [[nodiscard]] std::span<cudf::size_type const> build_key_indices() const
  {
    return _build_key_indices;
  }
  [[nodiscard]] cudf::null_equality null_equality() const noexcept { return _null_equality; }
  [[nodiscard]] cudf::size_type rows() const
  {
    return has_build_relation() ? build_relation_view().num_rows() : 0;
  }

 private:
  std::vector<state_container> _chunks;
  std::vector<cudf::size_type> _chunk_offsets;
  state_container _build_column_states;
  std::vector<cudf::size_type> _build_key_indices;
  std::optional<detail::join_materialized_column_layout> _layout;
  cudf::null_equality _null_equality;
};

// ----- Probe and completion state -----

/**
 * @brief Own mutable marks for one probe execution while retaining its immutable build.
 *
 * Concurrent probes atomically mark one map constructed during probe initialization.
 * Completion tokens retain this state without reading its marks.
 */
struct probe_data {
  state_container build;  ///< Retained immutable build dependency.

  core::hash_index_handle index;  ///< Mutable map shared by this accumulator's probe calls.

  probe_data(state_container_view input,
             query_stream const& stream_handle,
             rmm::device_async_resource_ref mr)
    : build{make_mutable_state_copy(input)}
  {
    auto const stream  = stream_handle.ref();
    auto const& source = get_task_private_state<build_state>(input, 0);
    if (source.has_build_relation()) {
      index = core::make_hash_index(source.build_keys(), source.mask(), stream, mr);
      order_deallocation_after_stream_use(source.build_column_states(), stream_handle);
    }
  }
};

/** @brief Share execution-owned probe state across concurrent calls to `next`. */
struct probe_accumulator : public task_private_state {
  std::shared_ptr<probe_data> value;  ///< Mutable execution state shared by concurrent probes.

  explicit probe_accumulator(std::shared_ptr<probe_data> state) : value{std::move(state)} {}
};

/** @brief Retain probe state until the completion barrier without reading pending device marks. */
struct completion_token : public task_private_state {
  std::shared_ptr<probe_data const> value;  ///< Read-only handle retained until scanning.

  explicit completion_token(std::shared_ptr<probe_data const> state) : value{std::move(state)} {}
};

/** @brief Retain one completion token per probe accumulator under a mutex. */
struct completion_state : public task_private_state {
  std::mutex mutex;  ///< Protects insertion into `probes`.

  std::unordered_map<probe_data const*, std::shared_ptr<probe_data const>>
    probes;  ///< Unique tokens, released after finalization merges their marks.

  state_container build;  ///< Build identity retained by the completed bitmap.

  std::optional<core::mark_bitmap> marks;  ///< Absent only when no probe tokens arrived.
};

// ----- Scan state -----

/** @brief Retain scan dependencies and claim original chunks without copying the completed bitmap.
 */
struct scan_state : public task_private_state {
  state_container inputs;  ///< Original build and read-only completion states.

  std::mutex mutex;  ///< Protects the cursor shared by concurrent unsplit calls.

  std::size_t next_chunk;
  std::size_t end_chunk;

  scan_state(state_container input, std::size_t begin, std::size_t end)
    : inputs{std::move(input)}, next_chunk{begin}, end_chunk{end}
  {
  }
};

}  // namespace

// ----- Build task -----

mark_join_build_task::mark_join_build_task(std::vector<cudf::size_type> build_key_indices,
                                           cudf::null_equality null_equality,
                                           std::vector<cudf::size_type> predicate_column_indices)
  : _build_key_indices{std::move(build_key_indices)},
    _null_equality{null_equality},
    _predicate_column_indices{std::move(predicate_column_indices)}
{
  GQE_EXPECTS(!_build_key_indices.empty(),
              "mark_join_build_task: build_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_build_key_indices, [](auto index) { return index >= 0; }),
              "mark_join_build_task: key indices must be non-negative",
              std::invalid_argument);
}

std::unique_ptr<task> mark_join_build_task::clone() const
{
  return std::make_unique<mark_join_build_task>(
    _build_key_indices, _null_equality, _predicate_column_indices);
}

state_container mark_join_build_task::initialize(context_reference,
                                                 query_stream const&,
                                                 rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_build_task::initialize"};
  return make_private_state_container<build_accumulator>();
}

void mark_join_build_task::next(state_container_view inputs,
                                state_container_view accumulator,
                                context_reference,
                                query_stream const& stream_handle,
                                rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_build_task::next"};
  GQE_EXPECTS(accumulator.size() == 1, "mark join build expects one accumulator");
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "mark join build input must be a logical table");
  validate_keys(*logical, _build_key_indices);
  detail::join_materialized_column_layout const layout{
    logical->num_columns(), _build_key_indices, _predicate_column_indices};
  auto materialized = materialize_columns(*logical, layout.logical_indices(), stream_handle, mr);
  auto columns      = std::move(materialized).release_column_states();
  if (auto const* mask = logical->valid_mask_state()) { columns.insert(columns.begin(), *mask); }
  build_chunk chunk{make_mutable_state_copy(inputs), std::move(columns)};
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);
  std::scoped_lock const guard{state.mutex};
  if (!state.chunks.empty()) {
    GQE_EXPECTS(logical_table_view::try_from(state.chunks.front().source)->num_columns() ==
                  logical->num_columns(),
                "mark join build chunk widths differ");
  }
  state.chunks.push_back(std::move(chunk));
}

state_container mark_join_build_task::finalize(state_container&& accumulator,
                                               context_reference,
                                               query_stream const& stream_handle,
                                               rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_build_task::finalize"};
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(accumulator.size() == 1, "mark join build expects one accumulator");
  auto& state = get_task_private_state<build_accumulator>(accumulator, 0);
  std::vector<state_container> sources;
  std::vector<state_container> columns;
  sources.reserve(state.chunks.size());
  columns.reserve(state.chunks.size());
  bool const has_mask = std::ranges::any_of(state.chunks, [](auto const& chunk) {
    return logical_table_view::try_from(chunk.source)->valid_mask().has_value();
  });
  for (auto& chunk : state.chunks) {
    // The shared concatenation helper expects consistent shapes. An unmasked chunk contributes
    // all-true entries only when another chunk carries a mask.
    if (has_mask && !logical_table_view::try_from(chunk.materialized)->valid_mask()) {
      auto mask =
        cudf::make_column_from_scalar(cudf::numeric_scalar<bool>{true, true, stream, mr},
                                      logical_table_view::try_from(chunk.source)->row_count(),
                                      stream,
                                      mr);
      auto mask_state = state_container_builder()
                          .add_state(state_kind::valid_mask{std::move(mask)}, stream_handle)
                          .build();
      chunk.materialized.insert(chunk.materialized.begin(), std::move(mask_state.front()));
    }
    sources.push_back(std::move(chunk.source));
    columns.push_back(std::move(chunk.materialized));
  }
  state_container payload;
  if (columns.size() == 1) {
    payload = std::move(columns.front());
  } else {
    auto concatenated = detail::concatenate_materialized_chunks(columns, stream_handle, mr);
    if (concatenated.table) {
      state_container_builder builder;
      if (concatenated.mask) {
        builder.add_state(
          state_kind::valid_mask{std::make_unique<cudf::column>(std::move(*concatenated.mask))},
          stream_handle);
      }
      payload = builder.add_state(std::move(*concatenated.table), stream_handle).build();
    }
  }
  return make_private_state_container<build_state>(std::move(sources),
                                                   std::move(payload),
                                                   _build_key_indices,
                                                   _predicate_column_indices,
                                                   _null_equality);
}

// ----- Probe task -----

mark_join_probe_task::mark_join_probe_task(std::vector<cudf::size_type> probe_key_indices,
                                           std::unique_ptr<expression> predicate)
  : _probe_key_indices{std::move(probe_key_indices)}, _predicate{std::move(predicate)}
{
  GQE_EXPECTS(!_probe_key_indices.empty(),
              "mark_join_probe_task: probe_key_indices must be non-empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_probe_key_indices, [](auto index) { return index >= 0; }),
              "mark_join_probe_task: key indices must be non-negative",
              std::invalid_argument);
}

std::unique_ptr<task> mark_join_probe_task::clone() const
{
  return std::make_unique<mark_join_probe_task>(_probe_key_indices,
                                                _predicate ? _predicate->clone() : nullptr);
}

state_container mark_join_probe_task::initialize(state_container_view inputs,
                                                 context_reference,
                                                 query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_probe_task::initialize"};
  GQE_EXPECTS(inputs.size() == 1,
              "mark_join_probe_task::initialize: expected exactly one build state");
  auto const& build = get_task_private_state<build_state>(inputs, 0);
  GQE_EXPECTS(build.build_key_indices().size() == _probe_key_indices.size(),
              "mark_join_probe_task::initialize: build/probe key counts differ");
  return make_private_state_container<probe_accumulator>(
    std::make_shared<probe_data>(inputs, stream_handle, mr));
}

std::optional<state_container> mark_join_probe_task::next(state_container_view inputs,
                                                          state_container_view accumulator,
                                                          context_reference,
                                                          query_stream const& stream_handle,
                                                          rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_probe_task::next"};
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(accumulator.size() == 1,
              "mark_join_probe_task::next: expected exactly one accumulator state");
  auto const& state  = get_task_private_state<probe_accumulator>(accumulator, 0).value;
  auto const& build  = get_task_private_state<build_state>(state->build, 0);
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(), "mark join probe input must be a logical table");
  validate_keys(*logical, _probe_key_indices);
  if (build.has_build_relation()) {
    auto const build_table = build.build_relation_view();
    std::vector<cudf::size_type> predicate_columns;
    if (_predicate) {
      auto const left_width = build.layout().logical_column_count();
      std::vector<expression const*> const roots{_predicate.get()};
      for (auto index : collect_column_references(roots)) {
        GQE_EXPECTS(index >= 0 && index < left_width + logical->num_columns(),
                    "mark join residual column index out of range",
                    std::invalid_argument);
        if (index >= left_width) { predicate_columns.push_back(index - left_width); }
      }
      std::vector<cudf::data_type> types;
      auto const left = logical_table_view::try_from(build.chunks().front());
      for (cudf::size_type i = 0; i < left_width; ++i) {
        types.push_back(left->column_type(i));
      }
      for (cudf::size_type i = 0; i < logical->num_columns(); ++i) {
        types.push_back(logical->column_type(i));
      }
      GQE_EXPECTS(_predicate->data_type(types).id() == cudf::type_id::BOOL8,
                  "mark join residual predicate must be Boolean",
                  std::invalid_argument);
    }
    detail::join_materialized_column_layout const layout{
      logical->num_columns(), _probe_key_indices, predicate_columns};
    auto materialized = materialize_columns(*logical, layout.logical_indices(), stream_handle, mr);
    auto const left   = build.build_keys();
    auto const right  = select_keys(materialized.table_view(), layout.remap(_probe_key_indices));
    GQE_EXPECTS(cudf::have_same_types(left, right),
                "mark join build/probe key types differ",
                std::invalid_argument);
    if (left.num_rows() > 0 && right.num_rows() > 0) {
      core::probe_hash_index(*state->index,
                             left,
                             right,
                             build_table,
                             materialized.table_view(),
                             build.layout().logical_column_count(),
                             build.layout().logical_indices(),
                             layout.logical_indices(),
                             logical->valid_mask(),
                             _predicate.get(),
                             build.null_equality(),
                             stream,
                             mr);
      order_deallocation_after_stream_use(build.build_column_states(), stream_handle);
    }
    order_deallocation_after_stream_use(materialized.states(), stream_handle);
  }
  order_deallocation_after_stream_use(inputs, stream_handle);
  return make_private_state_container<completion_token>(state);
}

// ----- Completion task -----

state_container mark_join_complete_task::initialize(context_reference,
                                                    query_stream const&,
                                                    rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_complete_task::initialize"};
  return make_private_state_container<completion_state>();
}

void mark_join_complete_task::next(state_container_view inputs,
                                   state_container_view accumulator,
                                   context_reference,
                                   query_stream const&,
                                   rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_complete_task::next"};
  GQE_EXPECTS(accumulator.size() == 1,
              "mark_join_complete_task::next: expected exactly one accumulator state");
  GQE_EXPECTS(inputs.size() == 1,
              "mark_join_complete_task::next: expected exactly one completion token");
  auto const& token = get_task_private_state<completion_token>(inputs, 0).value;
  auto& state       = get_task_private_state<completion_state>(accumulator, 0);
  std::scoped_lock const guard{state.mutex};
  state.probes.try_emplace(token.get(), token);
}

state_container mark_join_complete_task::finalize(state_container&& accumulator,
                                                  context_reference,
                                                  query_stream const& stream_handle,
                                                  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  auto& state       = get_task_private_state<completion_state>(accumulator, 0);
  if (!state.probes.empty()) {
    state.build       = state.probes.begin()->second->build;
    auto const& build = get_task_private_state<build_state>(state.build, 0);
    state.marks.emplace(build.rows(), stream, mr);
    for (auto const& [identity, probe] : state.probes) {
      GQE_EXPECTS(&get_task_private_state<build_state>(probe->build, 0) == &build,
                  "mark_join_complete_task: tokens belong to different builds");
      if (probe->index) { core::scan_hash_index(*probe->index, state.marks->data(), stream, mr); }
    }
    state.probes.clear();
  }
  return std::move(accumulator);
}

std::unique_ptr<task> mark_join_complete_task::clone() const
{
  return std::make_unique<mark_join_complete_task>();
}

// ----- Scan task -----

mark_join_scan_task::mark_join_scan_task(join_type_type join_type,
                                         std::vector<cudf::size_type> projection_indices,
                                         std::vector<cudf::data_type> output_data_types)
  : _join_type{join_type},
    _projection_indices{std::move(projection_indices)},
    _output_data_types{std::move(output_data_types)}
{
  GQE_EXPECTS(join_type == join_type_type::left_semi || join_type == join_type_type::left_anti,
              "mark_join_scan_task: only left semi and left anti joins are supported",
              std::invalid_argument);
  GQE_EXPECTS(_projection_indices.size() == _output_data_types.size() &&
                std::ranges::all_of(_projection_indices, [](auto i) { return i >= 0; }),
              "mark_join_scan_task: invalid output projection",
              std::invalid_argument);
  GQE_EXPECTS(
    std::ranges::all_of(
      _output_data_types,
      [](auto type) { return cudf::is_fixed_width(type) || type.id() == cudf::type_id::STRING; }),
    "mark_join_scan_task: output types must be fixed-width or STRING",
    std::invalid_argument);
}

std::unique_ptr<task> mark_join_scan_task::clone() const
{
  auto result =
    std::make_unique<mark_join_scan_task>(_join_type, _projection_indices, _output_data_types);
  result->_split_index = _split_index;
  result->_split_count = _split_count;
  return result;
}

std::optional<std::size_t> mark_join_scan_task::required_split_count_from_fold_input()
  const noexcept
{
  return _projection_indices.empty() ? std::nullopt : std::optional<std::size_t>{0};
}

cudf::size_type mark_join_scan_task::max_splits() const noexcept
{
  return _projection_indices.empty() ? 1 : std::numeric_limits<cudf::size_type>::max();
}

std::unique_ptr<iterate_task> mark_join_scan_task::split(cudf::size_type split_idx,
                                                         cudf::size_type total_splits) const
{
  GQE_EXPECTS(
    total_splits > 0 && total_splits <= max_splits() && split_idx >= 0 && split_idx < total_splits,
    "mark_join_scan_task: invalid split index or count",
    std::invalid_argument);
  auto result =
    std::make_unique<mark_join_scan_task>(_join_type, _projection_indices, _output_data_types);
  result->_split_index = split_idx;
  result->_split_count = total_splits;
  return result;
}

state_container mark_join_scan_task::initialize(state_container_view inputs,
                                                context_reference,
                                                query_stream const&,
                                                rmm::device_async_resource_ref) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_scan_task::initialize"};
  GQE_EXPECTS(inputs.size() == 2,
              "mark_join_scan_task::initialize: expected [build, completion] states");
  auto const& build    = get_task_private_state<build_state>(inputs, 0);
  auto const& complete = get_task_private_state<completion_state>(inputs, 1);
  if (!complete.build.empty()) {
    GQE_EXPECTS(&get_task_private_state<build_state>(complete.build, 0) == &build,
                "mark_join_scan_task::initialize: completion belongs to another build");
  }
  if (_split_index && !_projection_indices.empty()) {
    GQE_EXPECTS(build.chunks().size() <= static_cast<std::size_t>(_split_count),
                "mark_join_scan_task::initialize: split count does not cover retained chunks");
  }
  auto const begin = static_cast<std::size_t>(_split_index.value_or(0));
  auto const end   = _split_index                  ? begin + 1
                     : _projection_indices.empty() ? std::size_t{1}
                                                   : std::max(build.chunks().size(), std::size_t{1});
  return make_private_state_container<scan_state>(make_mutable_state_copy(inputs), begin, end);
}

std::optional<state_container> mark_join_scan_task::next(state_container_view iterator,
                                                         context_reference,
                                                         query_stream const& stream_handle,
                                                         rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::mark_join_scan_task::next"};
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(iterator.size() == 1,
              "mark_join_scan_task::next: expected exactly one iterator state");
  auto& state          = get_task_private_state<scan_state>(iterator, 0);
  auto const& build    = get_task_private_state<build_state>(state.inputs, 0);
  auto const& complete = get_task_private_state<completion_state>(state.inputs, 1);
  auto const chunks    = build.chunks();
  std::size_t chunk_index;
  {
    std::scoped_lock const guard{state.mutex};
    if (state.next_chunk == state.end_chunk) { return std::nullopt; }
    chunk_index = state.next_chunk++;
  }
  if (chunk_index >= chunks.size()) {
    return _projection_indices.empty() ? make_row_count_container(0)
                                       : make_empty_state_container(_output_data_types);
  }

  // Count-only output scans all chunks in one call to preserve a single total.
  auto const end_chunk        = _projection_indices.empty() ? chunks.size() : chunk_index + 1;
  cudf::size_type total_count = 0;
  detail::join_materialized_column_layout const empty_layout{0, {}, {}};
  for (; chunk_index < end_chunk; ++chunk_index) {
    auto const& chunk  = chunks[chunk_index];
    auto const logical = *logical_table_view::try_from(chunk);
    detail::validate_join_projection_types(
      _projection_indices, _output_data_types, {}, empty_layout, logical);
    auto const chunk_rows = logical.row_count();
    // No probe tokens means all active rows are unmatched. Allocate zeros only for this chunk.
    std::optional<core::mark_bitmap> unmarked;
    if (!complete.marks) { unmarked.emplace(chunk_rows, stream, mr); }
    auto const* marks = complete.marks ? complete.marks->data() : unmarked->data();
    if (complete.marks && chunk_rows > 0) { marks += build.chunk_offset(chunk_index); }
    auto selected = core::select_marked_rows(
      marks, chunk_rows, logical.valid_mask(), _join_type == join_type_type::left_anti, stream, mr);
    auto const count = static_cast<cudf::size_type>(selected.size());
    total_count += count;
    if (complete.marks) { complete.marks->order_after(stream); }
    if (!_projection_indices.empty()) {
      auto positions = make_position_list(
        std::make_unique<cudf::column>(std::move(selected), rmm::device_buffer{}, 0),
        stream_handle);
      auto output = detail::make_join_indirect_output(_projection_indices,
                                                      {},
                                                      empty_layout,
                                                      logical,
                                                      positions,
                                                      positions,
                                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                                      stream_handle,
                                                      mr);
      order_deallocation_after_stream_use(chunk, stream_handle);
      return output;
    }
    order_deallocation_after_stream_use(chunk, stream_handle);
  }
  return make_row_count_container(total_count);
}

}  // namespace qep
}  // namespace gqe
