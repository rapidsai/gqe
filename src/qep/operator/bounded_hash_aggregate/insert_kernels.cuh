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
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/stream>
#include <cudf/aggregation.hpp>
#include <cudf/utilities/bit.hpp>

#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

namespace cg = cooperative_groups;

/**
 * @brief Insert-path kernels of the bounded hash aggregate.
 *
 * Template definitions only. Explicit instantiations live in one translation unit per key type
 * (`insert_kernels_*.cu`) so the device compilations run in parallel and host-side operator
 * edits never rebuild them.
 */

/**
 * @brief Combine `value` over the lanes named in `peers` (a `match_any` group).
 *
 * All lanes of the tile must execute this convergently. The fixed 5-step rank-based
 * Hillis-Steele runs over the peer group, so no shuffle count diverges. On return the rank-0
 * lane of each group holds the group's combined value, while other lanes hold partial
 * combinations.
 *
 * @param[in] tile The calling warp tile.
 * @param[in] peers Peer mask from the tile's `match_any`; must contain the calling lane.
 * @param[in] value This lane's contribution.
 * @param[in] rank This lane's rank within `peers`.
 */
template <typename Value, cudf::aggregation::Kind K>
__device__ __forceinline__ Value
warp_group_reduce(cg::thread_block_tile<32> const& tile, unsigned peers, Value value, unsigned rank)
{
  Value combined = value;
#pragma unroll
  for (int step = 1; step < 32; step <<= 1) {
    // The lane holding rank (rank + step) within my peer group, or >= 32 if none.
    unsigned const source = __fns(peers, 0, static_cast<int>(rank + step + 1));
    Value const other     = tile.shfl(combined, static_cast<int>(source & 31u));
    if (source < 32) { combined = combine_values<Value, K>(combined, other); }
  }
  return combined;
}

/**
 * @brief Relaxed device-scope vector load of `Count` adjacent `ElementBytes`-byte elements.
 *
 * The vector issues as one load instruction, except four 8-byte elements, which take two
 * 128-bit loads before sm_100.
 *
 * # Memory Consistency
 *
 * PTX models a vector access as one scalar access per element, so each element is a relaxed
 * load of its own width (PTX ISA 9.4, section 8.2.4).
 *
 * Callers pick the width of the field that other threads update atomically, so that a key read
 * never merges two keys into one wider element. A load that partially overlaps a concurrent
 * atomic is a mixed-size data race. The PTX memory consistency axioms do not apply to a program
 * with such a race. The exception for overlapping atomics covers only read-modify-write
 * operations, not loads (PTX ISA 9.4, section 8.7.2).
 *
 * References:
 *
 *  - Section 8.2.4, "Memory Operations on Vector Data Types",
 *    https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-operations-on-vector-data-types
 *  - Section 8.7.2, "Limitations on Mixed-size Data-races",
 *    https://docs.nvidia.com/cuda/parallel-thread-execution/#mixed-size-limitations
 *
 * # Address Space
 *
 * The PTX `ld.global` loads take an address in the global state space, while C++ pointers are
 * generic addresses. `__cvta_generic_to_global` (PTX `cvta.to.global`) converts one into the
 * other. The two forms coincide on current GPUs, but PTX does not guarantee that.
 *
 * @tparam ElementBytes Element width in bytes (1, 2, 4, or 8).
 * @tparam Count Number of elements (2 or 4).
 * @param[in] address Global-window address of the first element, aligned to the vector's width.
 * @return The elements as raw bits in the matching CUDA vector type.
 */
template <std::size_t ElementBytes, int32_t Count>
__device__ __forceinline__ auto load_relaxed_vector(std::size_t address)
{
  static_assert(Count == 2 || Count == 4, "unsupported vector length");
  if constexpr (ElementBytes == 8 && Count == 4) {
    ulonglong4_32a vector;
#if __CUDA_ARCH__ >= 1000
    asm volatile("ld.relaxed.gpu.global.v4.u64 {%0, %1, %2, %3}, [%4];"
                 : "=l"(vector.x), "=l"(vector.y), "=l"(vector.z), "=l"(vector.w)
                 : "l"(address)
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
                 : "=l"(vector.x), "=l"(vector.y)
                 : "l"(address)
                 : "memory");
    asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
                 : "=l"(vector.z), "=l"(vector.w)
                 : "l"(address + 16)
                 : "memory");
#endif
    return vector;
  } else if constexpr (ElementBytes == 8) {
    ulonglong2 vector;
    asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];"
                 : "=l"(vector.x), "=l"(vector.y)
                 : "l"(address)
                 : "memory");
    return vector;
  } else if constexpr (ElementBytes == 4 && Count == 4) {
    uint4 vector;
    asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(vector.x), "=r"(vector.y), "=r"(vector.z), "=r"(vector.w)
                 : "l"(address)
                 : "memory");
    return vector;
  } else if constexpr (ElementBytes == 4) {
    uint2 vector;
    asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(vector.x), "=r"(vector.y)
                 : "l"(address)
                 : "memory");
    return vector;
  } else if constexpr (ElementBytes == 2 && Count == 4) {
    ushort4 vector;
    asm volatile("ld.relaxed.gpu.global.v4.u16 {%0, %1, %2, %3}, [%4];"
                 : "=h"(vector.x), "=h"(vector.y), "=h"(vector.z), "=h"(vector.w)
                 : "l"(address)
                 : "memory");
    return vector;
  } else if constexpr (ElementBytes == 2) {
    ushort2 vector;
    asm volatile("ld.relaxed.gpu.global.v2.u16 {%0, %1}, [%2];"
                 : "=h"(vector.x), "=h"(vector.y)
                 : "l"(address)
                 : "memory");
    return vector;
  } else {
    static_assert(ElementBytes == 1, "unsupported element width");
    // PTX has no 8-bit registers, so the bytes load zero-extended into 16-bit registers.
    unsigned short x;
    unsigned short y;
    if constexpr (Count == 4) {
      unsigned short z;
      unsigned short w;
      asm volatile("ld.relaxed.gpu.global.v4.u8 {%0, %1, %2, %3}, [%4];"
                   : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
                   : "l"(address)
                   : "memory");
      return make_uchar4(static_cast<unsigned char>(x),
                         static_cast<unsigned char>(y),
                         static_cast<unsigned char>(z),
                         static_cast<unsigned char>(w));
    } else {
      asm volatile("ld.relaxed.gpu.global.v2.u8 {%0, %1}, [%2];"
                   : "=h"(x), "=h"(y)
                   : "l"(address)
                   : "memory");
      return make_uchar2(static_cast<unsigned char>(x), static_cast<unsigned char>(y));
    }
  }
}

/**
 * @brief Load all keys of the probing bucket that starts at slot `base` with vector loads.
 *
 * A bucket's keys share one fetch unit, but scalar atomic loads issue one L2 request each, and
 * the later ones wait behind the first one's miss. Both bucket layouts store all of a bucket's
 * keys adjacently, starting at its first slot, in a span aligned to its size, so one relaxed
 * device-scope vector load reads the whole span. Its elements are the keys, so each key read
 * has the width of the key CAS in `probe_and_fill`. Any key width from 8 to 64 bits works:
 *
 *  - Four 64-bit keys: one 256-bit load on sm_100 and newer, two 128-bit loads before.
 *  - Two 64-bit keys, or four 32-bit keys: one 128-bit load.
 *  - Narrower spans: one load of the span's width.
 *
 * @param[in] storage The map storage.
 * @param[in] base First slot index of the bucket.
 * @return The bucket's keys in slot order.
 */
template <typename Key, typename Value>
__device__ __forceinline__ cuda::std::array<Key, slot_layout<Key, Value>::slots_per_element>
load_bucket_keys(map_storage_ref<Key, Value> const& storage, int64_t base)
{
  using layout            = slot_layout<Key, Value>;
  using element           = typename layout::storage_element;
  constexpr int32_t slots = layout::slots_per_element;
  static_assert(alignof(element) % (slots * sizeof(Key)) == 0,
                "a bucket's key span must be aligned for its vector load");
  auto const address = __cvta_generic_to_global(layout::key_ptr(storage.data(), base));
  return cuda::std::bit_cast<cuda::std::array<Key, slots>>(
    load_relaxed_vector<sizeof(Key), slots>(address));
}

/**
 * @brief Probe the map for `key`, filling an empty slot on first miss.
 *
 * The probe sequence comes from the cuco probing scheme. `make_iterator` yields
 * storage-element indices, each covering one probing bucket, and all of the bucket's keys are
 * checked for a match before the lowest-index empty slot is filled with a CAS.
 *
 * Every thread inserting the same key targets the same candidate slot, so a lost CAS re-reads
 * the bucket and either matches the winner or continues probing. Re-reads are bounded because
 * each slot transitions empty-to-filled at most once.
 *
 * The loop terminates when the probing iterator wraps to its initial bucket: the probe sequence
 * then visited every slot, so the map is saturated and the key cannot be placed.
 *
 * A CAS that fills a slot additionally counts into `histogram`, the per-segment occupancy
 * counts behind the control block. The same at-most-one transition bounds the increments to
 * exactly one per filled slot.
 *
 * @return The filled or matched slot index, or -1 when the map is saturated.
 */
template <typename Key, typename Value>
__device__ __forceinline__ int64_t probe_and_fill(map_storage_ref<Key, Value> const& storage,
                                                  map_probing_scheme<Key> const& probing_scheme,
                                                  Key key,
                                                  int32_t* histogram)
{
  using layout            = slot_layout<Key, Value>;
  constexpr Key empty_key = cuda::std::numeric_limits<Key>::max();
  auto const count_fill   = [&](int64_t slot) {
    device_atomic_ref<int32_t>{histogram[slot >> histogram_segment_log2]}.fetch_add(
      1, cuda::memory_order_relaxed);
  };
  auto probing_iter =
    probing_scheme.template make_iterator<layout::elements_per_probe>(key, storage.extent());
  auto const init_idx = *probing_iter;
  while (true) {
    int64_t const base = static_cast<int64_t>(*probing_iter) * layout::slots_per_element;
    auto const keys    = load_bucket_keys<Key, Value>(storage, base);
    // Slots fill in order, so a bucket's key can only follow filled slots. A match takes
    // precedence over an empty slot read in the same (not single-copy atomic) vector load.
    int32_t match = -1;
    int32_t empty = -1;
#pragma unroll
    for (int32_t i = 0; i < layout::slots_per_element; ++i) {
      if (match < 0 && keys[i] == key) { match = i; }
      if (empty < 0 && keys[i] == empty_key) { empty = i; }
    }
    if (match >= 0) { return base + match; }
    if (empty >= 0) {
      int64_t const target = base + empty;
      Key const old = atomic_cas_key(layout::key_ptr(storage.data(), target), empty_key, key);
      if (old == empty_key) {
        count_fill(target);
        return target;
      }
      if (old == key) { return target; }
      continue;  // Lost the CAS to a different key: re-read this bucket.
    }
    ++probing_iter;
    if (*probing_iter == init_idx) { return -1; }
  }
}

/**
 * @brief Aggregate one input chunk into the map.
 *
 * A grid-stride loop runs over the chunk's rows, with whole warp tiles active so that tile
 * collectives stay convergent.
 *
 * # Per-row Handling
 *
 *  - Rows excluded by the valid-row mask are skipped.
 *  - NULL-key and sentinel-key rows combine into their out-of-band accumulators.
 *  - The remaining rows pre-aggregate within the warp tile. `match_any` groups equal keys, and
 *    only the group leader probes the map and issues one atomic.
 *  - NULL values contribute the identity and do not mark the group's value as non-null.
 *
 * # Bound Violation
 *
 * A leader whose probe exhausts the map sets the bound-violation flag and drops the row. The
 * build task's `finalize` turns the flag into an error.
 *
 * # NOT NULL Values
 *
 * Under `N == nullability::not_null`, every selected row has a non-null value. The kernel then
 * reads no value null mask and writes no value validity, so `value_validity` is unused.
 *
 * @tparam N Declared nullability of the value column, and of the ARGMIN/ARGMAX payload column.
 */
template <typename Key, typename Input, typename Value, cudf::aggregation::Kind K, nullability N>
__global__ void insert_kernel(map_storage_ref<Key, Value> storage,
                              map_probing_scheme<Key> probing_scheme,
                              uint8_t* value_validity,
                              int64_t* wrap_counters,
                              control_block* control,
                              insert_column_args<Key, Input> args)
{
  constexpr Key empty_key = cuda::std::numeric_limits<Key>::max();
  constexpr bool is_count =
    K == cudf::aggregation::COUNT_VALID || K == cudf::aggregation::COUNT_ALL;
  constexpr bool carries_payload = K == cudf::aggregation::ARGMIN || K == cudf::aggregation::ARGMAX;
  // Only a result that can be NULL writes value validity (see `tracks_value_validity` in the
  // build task).
  constexpr bool result_may_be_null = !is_count && N == nullability::nullable;
  // Contributions, the warp reduction, and the out-of-band groups use the accumulation type.
  using Accumulation = accumulation_t<Value>;

  auto const tile            = cg::tiled_partition<32>(cg::this_thread_block());
  auto const num_rows        = static_cast<int64_t>(args.num_rows);
  auto index                 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const stride       = static_cast<int64_t>(gridDim.x) * blockDim.x;
  unsigned const lanemask_lt = (1u << tile.thread_rank()) - 1u;

  for (; tile.any(index < num_rows); index += stride) {
    bool const in_range = index < num_rows;
    bool const selected =
      in_range && (args.row_valid == nullptr || args.row_valid[index] != uint8_t{0});

    // Split the row into the in-map path (inserts) and the out-of-band path (oob). Lanes that
    // insert carry their key. All other lanes carry the empty sentinel, which no inserting
    // lane can carry, so the tile's `match_any` groups them out of the way.
    bool key_null = false;
    bool oob      = false;
    Key key       = empty_key;
    if (selected) {
      key_null = args.key_null_mask != nullptr &&
                 !cudf::bit_is_set(args.key_null_mask,
                                   static_cast<cudf::size_type>(args.key_mask_offset + index));
      if (key_null) {
        oob = true;
      } else {
        Key const raw_key = args.keys[index];
        if (raw_key == empty_key) {
          oob = true;
        } else {
          key = raw_key;
        }
      }
    }
    bool const inserts = selected && !oob;

    // Whether the row's value is valid. COUNT_ALL counts every row, and a value column declared
    // NOT NULL has no null mask to consult.
    bool value_valid = false;
    if (selected) {
      if constexpr (K == cudf::aggregation::COUNT_ALL || N == nullability::not_null) {
        value_valid = true;
      } else {
        value_valid =
          args.value_null_mask == nullptr ||
          cudf::bit_is_set(args.value_null_mask,
                           static_cast<cudf::size_type>(args.value_mask_offset + index));
        if constexpr (carries_payload) {
          // A NULL payload excludes the row, like a NULL ordering value: the winner's payload
          // validity cannot travel in the 16-byte pair, so a NULL payload cannot be carried.
          if (value_valid && args.payload_null_mask != nullptr) {
            value_valid =
              cudf::bit_is_set(args.payload_null_mask,
                               static_cast<cudf::size_type>(args.payload_mask_offset + index));
          }
        }
      }
    }

    // The row's contribution. A null value contributes the identity, and the counts contribute
    // one per valid row.
    Accumulation contribution = identity_value<Accumulation, K>();
    if (value_valid) {
      if constexpr (is_count) {
        contribution = Accumulation{1};
      } else if constexpr (carries_payload) {
        // The ordering value comes from the value column, the carried payload from the payload
        // column, read as raw 8-byte values.
        contribution = Accumulation{args.values[index], args.payload[index]};
      } else {
        contribution = widen<Input, Accumulation>(args.values[index]);
        // SUM_OF_SQUARES squares after widening, then reuses the SUM combine path.
        if constexpr (K == cudf::aggregation::SUM_OF_SQUARES) {
          contribution = contribution * contribution;
        }
      }
    }

    // Out-of-band rows combine directly. They are rare, so no warp pre-aggregation.
    if (oob) {
      out_of_band_group* group = key_null ? &control->null_key : &control->sentinel_key;
      // The status flags here and below are idempotent plain stores. See the state header's
      // insert-phase write contract for the memory-model argument.
      group->seen       = 1;
      auto* accumulator = reinterpret_cast<Accumulation*>(group->value);
      if constexpr (is_count) {
        if (contribution != Accumulation{0}) { atomic_add_value(accumulator, contribution); }
      } else {
        if (value_valid) {
          atomic_combine<Accumulation, K>(accumulator, contribution);
          if constexpr (result_may_be_null) { group->value_valid = 1; }
        }
      }
    }

    // Warp pre-aggregation, unconditional: measured as a strict freeroll on uniform input and a
    // large win on clustered input.
    unsigned const peers = tile.match_any(key);
    auto const rank      = static_cast<unsigned>(cuda::std::popcount(peers & lanemask_lt));
    // A leader combines only when some lane of its group holds a non-null value. When the result
    // cannot be NULL, every inserting lane's value is non-null, so the leader's group has one.
    bool group_has_valid_value = true;
    if constexpr (result_may_be_null) {
      group_has_valid_value = (tile.ballot(inserts && value_valid) & peers) != 0u;
    }
    Accumulation const combined = warp_group_reduce<Accumulation, K>(
      tile, peers, inserts ? contribution : identity_value<Accumulation, K>(), rank);

    if (rank == 0 && inserts) {
      // The occupancy histogram follows the control block in the control region.
      int32_t* const histogram = reinterpret_cast<int32_t*>(control + 1);
      int64_t const slot = probe_and_fill<Key, Value>(storage, probing_scheme, key, histogram);
      if (slot < 0) {
        control->bound_violation = 1;
      } else if constexpr (is_count) {
        if (combined != Value{0}) {
          atomic_add_value(slot_layout<Key, Value>::value_ptr(storage.data(), slot), combined);
        }
      } else if (group_has_valid_value) {
        if constexpr (std::is_same_v<Value, split_sum128>) {
          split_sum_add(slot_layout<Key, Value>::value_ptr(storage.data(), slot),
                        wrap_counters + slot,
                        combined);
        } else {
          atomic_combine<Value, K>(slot_layout<Key, Value>::value_ptr(storage.data(), slot),
                                   combined);
        }
        if constexpr (result_may_be_null) { value_validity[slot] = uint8_t{1}; }
      }
    }
  }
}

template <typename Key, typename Input, typename Value, cudf::aggregation::Kind K, nullability N>
void launch_insert(keyed_accumulator_state& map,
                   insert_column_args<Key, Input> const& args,
                   cuda::stream_ref stream)
{
  auto& storage = static_cast<typed_map_storage<Key, Value>&>(map.storage()).storage();
  auto const grid_size =
    utility::detect_launch_grid_size(insert_kernel<Key, Input, Value, K, N>, launch_block_size);
  insert_kernel<Key, Input, Value, K, N>
    <<<grid_size, launch_block_size, 0, stream.get()>>>(storage.ref(),
                                                        map_probing_scheme<Key>{},
                                                        map.value_validity(),
                                                        map.wrap_counters(),
                                                        static_cast<control_block*>(map.control()),
                                                        args);
  GQE_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
