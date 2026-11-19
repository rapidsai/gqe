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

#pragma once

#include "device_helpers.cuh"
#include "launchers.hpp"
#include "map_layout.hpp"
#include "map_storage.cuh"

#include <gqe/qep/shapes/keyed_accumulator.hpp>
#include <gqe/utility/cuda.hpp>

#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/stream>
#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cooperative_groups.h>

#include <cstdint>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

namespace cg = cooperative_groups;

/** @brief Reduce an accumulator to its emitted `scan_output_t` (the pair emits its payload). */
template <typename Value>
__device__ __forceinline__ scan_output_t<Value> extract_scan_output(Value value)
{
  if constexpr (is_arg_pair<Value>) {
    return value.payload;
  } else {
    return value;
  }
}

/**
 * @brief Scan-path and map-initialization kernels of the bounded hash aggregate.
 *
 * Template definitions only. Explicit instantiations live in one translation unit per key type
 * (`scan_kernels_*.cu`), which also own the storage-creation and map-initialization
 * instantiations to balance device compile time across the translation units.
 */

/**
 * @brief Initialize the control block: out-of-band identities and cleared flags.
 *
 * Single-thread launch, issued together with the cuco storage's sentinel fill.
 */
template <typename Value>
__global__ void initialize_control_kernel(control_block* control, Value identity)
{
  out_of_band_group group{};
  *reinterpret_cast<Value*>(group.value) = identity;
  control->null_key                      = group;
  control->sentinel_key                  = group;
  control->bound_violation               = 0;
  control->not_null_violation            = 0;
}

/**
 * @brief Compact the filled slots of one window into output arrays.
 *
 * Each warp tile acquires output rows cooperatively. The tile ballots its filled slots, the
 * leader issues one counter atomic for all of them, and lanes write at their rank offset, so
 * the writes of one acquisition are contiguous and coalesce. Output order within the window is
 * otherwise unspecified. `out_value_valid` is written only when value validity is tracked.
 */
template <typename Key, typename Value>
__global__ void scan_kernel(map_element<Key, Value> const* elements,
                            int64_t begin_slot,
                            int64_t end_slot,
                            uint8_t const* value_validity,
                            int64_t const* wrap_counters,
                            Key* out_keys,
                            scan_output_t<Value>* out_values,
                            uint8_t* out_value_valid,
                            cudf::size_type* out_count)
{
  using layout            = slot_layout<Key, Value>;
  constexpr Key empty_key = cuda::std::numeric_limits<Key>::max();

  auto const tile      = cg::tiled_partition<32>(cg::this_thread_block());
  auto index           = begin_slot + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  unsigned const lanemask_lt = (1u << tile.thread_rank()) - 1u;

  for (; tile.any(index < end_slot); index += stride) {
    Key key              = empty_key;
    bool const is_filled = [&]() {
      if (index >= end_slot) { return false; }
      key = *layout::key_ptr(elements, index);
      return key != empty_key;
    }();

    unsigned const writers = tile.ballot(is_filled);
    if (writers == 0u) { continue; }

    cudf::size_type base = 0;
    auto const leader    = cuda::std::countr_zero(writers);
    if (tile.thread_rank() == static_cast<unsigned>(leader)) {
      base = device_atomic_ref<cudf::size_type>{*out_count}.fetch_add(
        static_cast<cudf::size_type>(cuda::std::popcount(writers)), cuda::memory_order_relaxed);
    }
    base = tile.shfl(base, leader);

    if (is_filled) {
      auto const position = base + cuda::std::popcount(writers & lanemask_lt);
      out_keys[position]  = key;
      if constexpr (std::is_same_v<Value, split_sum128>) {
        out_values[position] =
          split_sum_value(*layout::value_ptr(elements, index), wrap_counters[index]);
      } else {
        out_values[position] = extract_scan_output<Value>(*layout::value_ptr(elements, index));
      }
      if (out_value_valid != nullptr) { out_value_valid[position] = value_validity[index]; }
    }
  }
}

/**
 * @brief Append the populated out-of-band groups after the compacted window rows.
 *
 * Single-thread launch. The NULL-key row precedes the sentinel-key row. The NULL-key row's key
 * output is zero-initialized, and its bit in the host-built key null mask is cleared.
 */
template <typename Key, typename Value>
__global__ void append_out_of_band_kernel(control_block const* control,
                                          bool append_null_key,
                                          bool append_sentinel_key,
                                          cudf::size_type base,
                                          Key* out_keys,
                                          scan_output_t<Value>* out_values,
                                          uint8_t* out_value_valid)
{
  cudf::size_type row = base;
  if (append_null_key) {
    out_keys[row]   = Key{};
    out_values[row] = extract_scan_output<accumulation_t<Value>>(
      *reinterpret_cast<accumulation_t<Value> const*>(control->null_key.value));
    if (out_value_valid != nullptr) {
      out_value_valid[row] = control->null_key.value_valid != 0 ? uint8_t{1} : uint8_t{0};
    }
    ++row;
  }
  if (append_sentinel_key) {
    out_keys[row]   = cuda::std::numeric_limits<Key>::max();
    out_values[row] = extract_scan_output<accumulation_t<Value>>(
      *reinterpret_cast<accumulation_t<Value> const*>(control->sentinel_key.value));
    if (out_value_valid != nullptr) {
      out_value_valid[row] = control->sentinel_key.value_valid != 0 ? uint8_t{1} : uint8_t{0};
    }
  }
}

template <typename Key, typename Value>
void launch_initialize_map(keyed_accumulator_state& map,
                           cudf::aggregation::Kind kind,
                           cuda::stream_ref stream)
{
  constexpr Key empty_key = std::numeric_limits<Key>::max();
  auto const identity     = [&]() {
    switch (kind) {
      case cudf::aggregation::MIN:
        return identity_value<accumulation_t<Value>, cudf::aggregation::MIN>();
      case cudf::aggregation::MAX:
        return identity_value<accumulation_t<Value>, cudf::aggregation::MAX>();
      case cudf::aggregation::PRODUCT:
        return identity_value<accumulation_t<Value>, cudf::aggregation::PRODUCT>();
      case cudf::aggregation::ARGMIN:
        return identity_value<accumulation_t<Value>, cudf::aggregation::ARGMIN>();
      case cudf::aggregation::ARGMAX:
        return identity_value<accumulation_t<Value>, cudf::aggregation::ARGMAX>();
      default: return identity_value<accumulation_t<Value>, cudf::aggregation::SUM>();
    }
  }();

  auto& storage = static_cast<typed_map_storage<Key, Value>&>(map.storage()).storage();
  // The split DECIMAL128 SUM starts its inline part at the (zero) identity's low limb.
  Value slot_identity{};
  if constexpr (std::is_same_v<Value, split_sum128>) {
    slot_identity = split_sum128{static_cast<int64_t>(identity.lo)};
  } else {
    slot_identity = identity;
  }
  storage.initialize_async(slot_layout<Key, Value>::fill_element(empty_key, slot_identity),
                           stream.get());

  // Define the control block's padding bytes. The struct assignments in
  // `initialize_control_kernel` write only the members, while `read_control_region` copies
  // the whole block, so unwritten padding would surface as an uninitialized-read report
  // under compute-sanitizer's initcheck.
  GQE_CUDA_TRY(cudaMemsetAsync(map.control(), 0, sizeof(control_block), stream.get()));

  initialize_control_kernel<accumulation_t<Value>>
    <<<1, 1, 0, stream.get()>>>(static_cast<control_block*>(map.control()), identity);
  GQE_CUDA_TRY(cudaPeekAtLastError());

  // Zero the occupancy histogram behind the control block.
  GQE_CUDA_TRY(
    cudaMemsetAsync(static_cast<control_block*>(map.control()) + 1,
                    0,
                    static_cast<std::size_t>(histogram_segments(map.capacity())) * sizeof(int32_t),
                    stream.get()));

  if (map.wrap_counter_bytes() != 0) {
    GQE_CUDA_TRY(cudaMemsetAsync(map.wrap_counters(), 0, map.wrap_counter_bytes(), stream.get()));
  }

  if (map.tracks_value_validity()) {
    GQE_CUDA_TRY(cudaMemsetAsync(
      map.value_validity(), 0, static_cast<std::size_t>(map.capacity()), stream.get()));
  }
}

template <typename Key, typename Value>
void launch_scan(map_element<Key, Value> const* elements,
                 int64_t begin_slot,
                 int64_t end_slot,
                 uint8_t const* value_validity,
                 int64_t const* wrap_counters,
                 Key* out_keys,
                 scan_output_t<Value>* out_values,
                 uint8_t* out_value_valid,
                 cudf::size_type* out_count,
                 cuda::stream_ref stream)
{
  auto const grid_size =
    utility::detect_launch_grid_size(scan_kernel<Key, Value>, launch_block_size);
  scan_kernel<Key, Value><<<grid_size, launch_block_size, 0, stream.get()>>>(elements,
                                                                             begin_slot,
                                                                             end_slot,
                                                                             value_validity,
                                                                             wrap_counters,
                                                                             out_keys,
                                                                             out_values,
                                                                             out_value_valid,
                                                                             out_count);
  GQE_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Key, typename Value>
void launch_append_out_of_band(control_block const* control,
                               bool append_null_key,
                               bool append_sentinel_key,
                               cudf::size_type base,
                               Key* out_keys,
                               scan_output_t<Value>* out_values,
                               uint8_t* out_value_valid,
                               cuda::stream_ref stream)
{
  append_out_of_band_kernel<Key, Value><<<1, 1, 0, stream.get()>>>(
    control, append_null_key, append_sentinel_key, base, out_keys, out_values, out_value_valid);
  GQE_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
