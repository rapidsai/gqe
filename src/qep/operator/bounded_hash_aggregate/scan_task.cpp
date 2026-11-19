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

#include "launchers.hpp"
#include "map_layout.hpp"
#include "map_state_access.hpp"
#include "type_dispatch.hpp"

#include <gqe/qep/operator/bounded_hash_aggregate.hpp>
#include <gqe/qep/shapes/keyed_accumulator.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

namespace bha = bounded_hash_aggregate::detail;

// -----------------------------------------------------------------------------
// Scan state and chunk construction
// -----------------------------------------------------------------------------

/**
 * @brief Iterator-private scan cursor. Concurrent `next` calls acquire windows through `next_slot`.
 */
struct bounded_hash_aggregate_scan_cursor final : public task_private_state {
  std::atomic<int64_t> next_slot{0};

  /** @brief Slot count per acquired window. */
  int64_t window_slots{0};

  /** @brief First slot of this instance's split slice. */
  int64_t slice_begin{0};

  /** @brief One past the last slot of this instance's split slice. */
  int64_t slice_end{0};

  /**
   * @brief Host copies of the out-of-band seen flags, read once at `initialize` (the map is final).
   * Cleared on every split but the last, which carries the out-of-band rows.
   */
  bool null_key_seen{false};
  bool sentinel_key_seen{false};

  /**
   * @brief Host copy of the whole map's occupancy histogram, read once at `initialize` (the map is
   * final). Read-only afterwards, so concurrent `next` calls share it safely.
   */
  std::vector<int32_t> segment_counts;

  /**
   * @brief Exact row count of the window `[begin_slot, end_slot)`, when the histogram knows it.
   *
   * The occupancy histogram counts whole segments, so the count is exact when both window
   * boundaries land on segment boundaries. The map's end counts as a boundary, because the
   * last segment is simply short. Windows and slices are segment-aligned whenever the row
   * bound allows, so this is the common case. The sub-segment windows of a small row bound
   * return `std::nullopt` and fall back to the device-count readback.
   */
  [[nodiscard]] std::optional<cudf::size_type> exact_window_rows(int64_t begin_slot,
                                                                 int64_t end_slot,
                                                                 int64_t capacity) const
  {
    if (begin_slot % bha::histogram_segment_slots != 0) { return std::nullopt; }
    if (end_slot % bha::histogram_segment_slots != 0 && end_slot != capacity) {
      return std::nullopt;
    }
    auto const begin_segment = begin_slot >> bha::histogram_segment_log2;
    auto const end_segment   = bha::histogram_segments(end_slot);
    int64_t rows             = 0;
    for (auto segment = begin_segment; segment < end_segment; ++segment) {
      rows += segment_counts[segment];
    }
    // A window never spans more slots than a chunk's 32-bit row count holds.
    return static_cast<cudf::size_type>(rows);
  }
};

[[nodiscard]] bounded_hash_aggregate_scan_cursor& get_cursor(state_container_view iterator)
{
  GQE_EXPECTS(iterator.size() == 2 && iterator[1] != nullptr,
              "bounded_hash_aggregate_scan_task: malformed iterator state");
  auto* private_state = std::get_if<state_kind::task_private>(iterator[1].get());
  GQE_EXPECTS(private_state != nullptr && private_state->data != nullptr,
              "bounded_hash_aggregate_scan_task: iterator slot 1 is not the scan cursor");
  // The slot was populated in `initialize` with this concrete type, so a `static_cast` suffices.
  return *static_cast<bounded_hash_aggregate_scan_cursor*>(private_state->data.get());
}

template <typename Key, typename Value>
[[nodiscard]] state_container make_scan_chunk(keyed_accumulator_state& map,
                                              int64_t begin_slot,
                                              int64_t end_slot,
                                              std::optional<cudf::size_type> known_rows,
                                              bool append_null_key,
                                              bool append_sentinel_key,
                                              query_stream const& stream_handle,
                                              rmm::device_async_resource_ref mr)
{
  auto const stream           = stream_handle.ref();
  auto const window_slots     = end_slot - begin_slot;
  auto const out_of_band_rows = (append_null_key ? 1 : 0) + (append_sentinel_key ? 1 : 0);
  bool const track_validity   = map.tracks_value_validity();

  // With the histogram's exact count the output allocations are exact. The fallback
  // allocates for a fully filled window.
  auto const max_rows =
    static_cast<std::size_t>(known_rows.value_or(window_slots)) + out_of_band_rows;

  rmm::device_uvector<Key> keys(max_rows, stream, mr);
  rmm::device_uvector<bha::scan_output_t<Value>> values(max_rows, stream, mr);
  rmm::device_uvector<uint8_t> value_valid(track_validity ? max_rows : 0, stream, mr);
  rmm::device_scalar<cudf::size_type> counter(stream, mr);
  counter.set_value_to_zero_async(stream);

  auto const* elements =
    static_cast<typename bha::slot_layout<Key, Value>::storage_element const*>(map.slots());
  // A window known to be empty skips the kernel launch entirely.
  if (window_slots > 0 && known_rows.value_or(1) != 0) {
    bha::launch_scan<Key, Value>(elements,
                                 begin_slot,
                                 end_slot,
                                 map.value_validity(),
                                 map.wrap_counters(),
                                 keys.data(),
                                 values.data(),
                                 track_validity ? value_valid.data() : nullptr,
                                 counter.data(),
                                 stream);
  }
  // The chunk's columns need an exact host row count. The occupancy histogram provides it for
  // segment-aligned windows without touching the device. The sub-segment fallback must wait
  // for the window scan and read the device count back. That synchronizes only the calling
  // thread's stream, and workers on other streams overlap the wait.
  auto const compacted_rows = known_rows.has_value() ? *known_rows : counter.value(stream);

  auto total_rows = compacted_rows;
  if (out_of_band_rows > 0) {
    bha::launch_append_out_of_band<Key, Value>(
      static_cast<bha::control_block const*>(map.control()),
      append_null_key,
      append_sentinel_key,
      compacted_rows,
      keys.data(),
      values.data(),
      track_validity ? value_valid.data() : nullptr,
      stream);
    total_rows += out_of_band_rows;
  }

  // Key column: nullable only when the NULL-key group row is present (the row right after the
  // compacted window rows).
  rmm::device_buffer key_null_mask{};
  cudf::size_type key_null_count = 0;
  if (append_null_key) {
    key_null_mask = cudf::create_null_mask(total_rows, cudf::mask_state::ALL_VALID, stream, mr);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(key_null_mask.data()),
                        compacted_rows,
                        compacted_rows + 1,
                        false,
                        stream);
    key_null_count = 1;
  }
  auto key_column = std::make_unique<cudf::column>(
    map.key_type(), total_rows, keys.release(), std::move(key_null_mask), key_null_count);

  // Value column: nullable when validity tracking recorded a group without a non-null value.
  rmm::device_buffer value_null_mask{};
  cudf::size_type value_null_count = 0;
  if (track_validity && total_rows > 0) {
    cudf::column_view const valid_view{
      cudf::data_type{cudf::type_id::BOOL8}, total_rows, value_valid.data(), nullptr, 0};
    auto [mask, null_count] = cudf::bools_to_mask(valid_view, stream, mr);
    if (null_count > 0) {
      value_null_mask  = std::move(*mask);
      value_null_count = null_count;
    }
  }
  auto value_column = std::make_unique<cudf::column>(
    map.value_type(), total_rows, values.release(), std::move(value_null_mask), value_null_count);

  return state_container_builder()
    .add_state(state_kind::cudf_column{std::move(key_column)}, stream_handle)
    .add_state(state_kind::cudf_column{std::move(value_column)}, stream_handle)
    .build();
}

/** @brief Typed call target of `dispatch_map_types` compacting one slot window into a chunk. */
struct scan_chunk_fn {
  template <typename Key, typename Value>
  [[nodiscard]] state_container operator()(keyed_accumulator_state& map,
                                           int64_t begin_slot,
                                           int64_t end_slot,
                                           std::optional<cudf::size_type> known_rows,
                                           bool append_null_key,
                                           bool append_sentinel_key,
                                           query_stream const& stream_handle,
                                           rmm::device_async_resource_ref mr) const
  {
    // ARGMIN/ARGMAX dispatch on the ordering type. Their accumulator is the pair.
    auto const kind = map.aggregation_kind();
    if constexpr (std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>) {
      if (kind == cudf::aggregation::ARGMIN || kind == cudf::aggregation::ARGMAX) {
        return make_scan_chunk<Key, bha::arg_pair<Value>>(map,
                                                          begin_slot,
                                                          end_slot,
                                                          known_rows,
                                                          append_null_key,
                                                          append_sentinel_key,
                                                          stream_handle,
                                                          mr);
      }
    }
    // The DECIMAL128 SUM uses the split accumulator.
    if constexpr (std::is_same_v<Value, bha::two_limb>) {
      if (kind == cudf::aggregation::SUM) {
        return make_scan_chunk<Key, bha::split_sum128>(map,
                                                       begin_slot,
                                                       end_slot,
                                                       known_rows,
                                                       append_null_key,
                                                       append_sentinel_key,
                                                       stream_handle,
                                                       mr);
      }
    }
    return make_scan_chunk<Key, Value>(map,
                                       begin_slot,
                                       end_slot,
                                       known_rows,
                                       append_null_key,
                                       append_sentinel_key,
                                       stream_handle,
                                       mr);
  }
};

}  // namespace

// -----------------------------------------------------------------------------
// bounded_hash_aggregate_scan_task
// -----------------------------------------------------------------------------

bounded_hash_aggregate_scan_task::bounded_hash_aggregate_scan_task(single_output_t)
  : bounded_hash_aggregate_scan_task{std::nullopt, slice{0, 1}}
{
}

bounded_hash_aggregate_scan_task::bounded_hash_aggregate_scan_task(
  cudf::size_type max_rows_per_output)
  : bounded_hash_aggregate_scan_task{max_rows_per_output, slice{0, 1}}
{
}

bounded_hash_aggregate_scan_task::bounded_hash_aggregate_scan_task(
  std::optional<cudf::size_type> max_rows_per_output, slice scan_slice)
  : _max_rows_per_output{max_rows_per_output}, _slice{scan_slice}
{
  // Every public constructor, `split`, and `clone` funnels through here, so these checks make
  // an invalid task state unrepresentable.
  GQE_EXPECTS(!_max_rows_per_output.has_value() || *_max_rows_per_output > 0,
              "bounded_hash_aggregate_scan_task: max_rows_per_output must be positive",
              std::logic_error);
  GQE_EXPECTS(_slice.count > 0 && _slice.index >= 0 && _slice.index < _slice.count,
              "bounded_hash_aggregate_scan_task: the slice must satisfy 0 <= index < count",
              std::logic_error);
}

cudf::size_type bounded_hash_aggregate_scan_task::max_splits() const noexcept
{
  // Every count degenerates gracefully: an empty slice still emits its schema-carrying empty
  // chunk, so the builder's configuration is the only effective limit.
  return std::numeric_limits<cudf::size_type>::max();
}

std::unique_ptr<iterate_task> bounded_hash_aggregate_scan_task::split(
  cudf::size_type split_idx, cudf::size_type total_splits) const
{
  // The delegating constructor enforces `0 <= split_idx < total_splits`. A split product
  // emits its whole slice as one chunk, per the trait's one-shot contract.
  return std::unique_ptr<iterate_task>{
    new bounded_hash_aggregate_scan_task{std::nullopt, slice{split_idx, total_splits}}};
}

state_container bounded_hash_aggregate_scan_task::initialize(
  state_container_view inputs,
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref /* mr */) const
{
  utility::nvtx_scoped_range range{"bounded_hash_aggregate_scan_task::initialize"};
  auto const stream = stream_handle.ref();

  GQE_EXPECTS(inputs.size() == 1,
              "bounded_hash_aggregate_scan_task: expected exactly the build task's map state as "
              "input",
              std::logic_error);
  auto const& handle = bha::get_map_handle(inputs.front(), "bounded_hash_aggregate_scan_task");

  // The map is final once this task initializes (the executor orders the predecessor's
  // finalize first), so the out-of-band seen flags and the occupancy histogram are read once here,
  // in one transfer under one synchronize.
  auto region = bha::read_control_region(*handle, stream);

  auto cursor            = std::make_unique<bounded_hash_aggregate_scan_cursor>();
  cursor->segment_counts = std::move(region.segment_counts);

  // A balanced, segment-aligned slice of the map: the first `segments % count` slices get one
  // extra segment. Segment alignment keeps the slices' window row counts on the occupancy
  // histogram, and slices past the last segment are empty. The last split carries the
  // out-of-band rows, while every other split leaves the flags cleared.
  auto const capacity      = handle->capacity();
  auto const num_segments  = bha::histogram_segments(capacity);
  auto const split_count   = static_cast<int64_t>(_slice.count);
  auto const split_index   = static_cast<int64_t>(_slice.index);
  auto const base_segments = num_segments / split_count;
  auto const remainder     = num_segments % split_count;
  auto const begin_segment = split_index * base_segments + std::min(split_index, remainder);
  auto const end_segment   = begin_segment + base_segments + (split_index < remainder ? 1 : 0);
  cursor->slice_begin      = std::min(begin_segment * bha::histogram_segment_slots, capacity);
  cursor->slice_end        = std::min(end_segment * bha::histogram_segment_slots, capacity);
  cursor->next_slot        = cursor->slice_begin;
  if (split_index == split_count - 1) {
    cursor->null_key_seen     = region.control.null_key.seen != 0;
    cursor->sentinel_key_seen = region.control.sentinel_key.seen != 0;
  }

  // The window is the whole slice in the single-output mode, or `max_rows_per_output` slots
  // otherwise, so a window's compacted rows never exceed the bound.
  //
  //  - A row bound of at least one histogram segment rounds down to whole segments, keeping
  //    the windows segment-aligned so their exact row counts come from the occupancy histogram.
  //  - Smaller bounds keep slot-granular windows and fall back to the device-count readback.
  //  - Both clamp so a fully filled window plus the two out-of-band rows always fits a
  //    `cudf::size_type` chunk. The clamp is itself a segment multiple, so clamping never
  //    breaks the alignment.
  constexpr int64_t max_window_slots =
    (static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) - 2) /
    bha::histogram_segment_slots * bha::histogram_segment_slots;
  auto requested_window = _max_rows_per_output.has_value()
                            ? static_cast<int64_t>(*_max_rows_per_output)
                            : cursor->slice_end - cursor->slice_begin;
  if (_max_rows_per_output.has_value() && requested_window >= bha::histogram_segment_slots) {
    requested_window =
      requested_window / bha::histogram_segment_slots * bha::histogram_segment_slots;
  }
  cursor->window_slots = std::clamp<int64_t>(requested_window, bha::bucket_slots, max_window_slots);

  GQE_LOG_TRACE(
    "Initialize QEP bounded hash aggregate scan: capacity={}, window={}, "
    "slice=[{}, {})",
    capacity,
    cursor->window_slots,
    cursor->slice_begin,
    cursor->slice_end);

  return state_container_builder()
    .add_state(state_kind::keyed_accumulator{handle})
    .add_state(state_kind::task_private{std::move(cursor)})
    .build();
}

std::optional<state_container> bounded_hash_aggregate_scan_task::next(
  state_container_view iterator,
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"bounded_hash_aggregate_scan_task::next"};

  GQE_EXPECTS(iterator.size() == 2, "bounded_hash_aggregate_scan_task: malformed iterator state");
  auto const& handle = bha::get_map_handle(iterator.front(), "bounded_hash_aggregate_scan_task");
  auto& cursor       = get_cursor(iterator);

  auto const begin_slot = cursor.next_slot.fetch_add(cursor.window_slots);
  // An empty slice still emits its one schema-correct chunk: the first window acquisition lands
  // exactly on `slice_begin == slice_end` once, and every later acquisition is past it.
  bool const emits_empty_slice_chunk =
    begin_slot == cursor.slice_begin && cursor.slice_begin == cursor.slice_end;
  if (begin_slot >= cursor.slice_end && !emits_empty_slice_chunk) { return std::nullopt; }
  auto const end_slot = std::min(begin_slot + cursor.window_slots, cursor.slice_end);

  // The slice's final window carries the out-of-band group rows. Only the last split does
  // this, because the other splits' flags are cleared at initialize.
  bool const is_final_window     = end_slot == cursor.slice_end;
  bool const append_null_key     = is_final_window && cursor.null_key_seen;
  bool const append_sentinel_key = is_final_window && cursor.sentinel_key_seen;

  auto const known_rows = cursor.exact_window_rows(begin_slot, end_slot, handle->capacity());

  // The ordering type selects the accumulator layout (it equals the value type except for
  // ARGMIN/ARGMAX).
  return bha::dispatch_map_types(handle->key_type(),
                                 handle->ordering_type(),
                                 scan_chunk_fn{},
                                 *handle,
                                 begin_slot,
                                 end_slot,
                                 known_rows,
                                 append_null_key,
                                 append_sentinel_key,
                                 stream_handle,
                                 mr);
}

std::unique_ptr<task> bounded_hash_aggregate_scan_task::clone() const
{
  return std::unique_ptr<task>{new bounded_hash_aggregate_scan_task{_max_rows_per_output, _slice}};
}

}  // namespace qep
}  // namespace gqe
