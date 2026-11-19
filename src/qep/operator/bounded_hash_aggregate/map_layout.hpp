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

#include <gqe/utility/helpers.hpp>

#include <cuda/std/bit>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief Device-storage layout of the bounded hash aggregate map.
 *
 * Host-safe plain-old-data definitions shared between the host-side operator code and the CUDA
 * kernels. The `keyed_accumulator_state` stores these structures type-erased, and the
 * operator's runtime type dispatch reinterprets them with the types recorded in the state's
 * descriptors. The cuco-typed storage and probing aliases live in `map_storage.cuh`, so this
 * header stays includable from host-only translation units.
 */

/**
 * @brief Two-limb 128-bit accumulator storage.
 *
 * Used for `DECIMAL128` sums. The value is `hi * 2^64 + lo` in two's complement, with `lo` the
 * unsigned low limb. Addition decomposes into two independent 64-bit atomic adds with carry
 * propagation, so a 16-byte atomic is never needed. Torn intermediate limb states are
 * unobservable because no map reads happen until all inserts quiesce.
 *
 * `alignas(8)` keeps the type at 16 unpadded bytes, so the mirrored bucket stays at 24 bytes
 * per slot. The bucket's member alignment places each accumulator on a 16-byte boundary.
 */
struct alignas(8) two_limb {
  unsigned long long lo;
  long long hi;
};

static_assert(sizeof(two_limb) == 16, "two_limb must pack into 16 bytes");

/** @brief Slots per keys-first bucket. */
inline constexpr int32_t keys_first_slots = 4;

/**
 * @brief Inline signed 64-bit part of a split DECIMAL128 SUM accumulator.
 *
 *  - The full sum is `((int128)wrap_counter << 64) + (int128)lo`. The per-slot wrap counter lives
 *    in the map state's wrap-counter array (see `split_sum_add`).
 *  - Sums that stay within the int64 range, the common case, never touch the counter.
 *  - The DECIMAL128 SUM therefore uses the 8-byte keys-first bucket instead of the
 *    24-byte-per-slot mirrored bucket.
 *  - Contributions, the warp reduction, and the out-of-band groups keep the full 128-bit
 *    `two_limb`.
 */
struct split_sum128 {
  int64_t lo;
};

static_assert(sizeof(split_sum128) == 8, "the split sum's inline part must be 8 bytes");

/**
 * @brief Keys-first four-slot bucket for accumulators up to 8 bytes.
 *
 * All four keys come first, then all four accumulators. For 64-bit keys and 8-byte accumulators
 * a bucket is 64 bytes, the L2 fetch unit of a DRAM miss, aligned to it:
 *
 *  - The keys form one span aligned to its size, so a probe reads all four with one vector load
 *    (one 256-bit load on sm_100 and newer, two 128-bit loads before, one 128-bit load for 32-bit
 *    keys).
 *  - A slot's accumulator lies in the same bucket as its key, as in the packed layout, so the
 *    atomic combine targets data the probe just brought into L2.
 *  - Four slots per probing step shorten the probe chains at high load factors.
 *
 * The bucket is aligned to its size rounded up to a power of two, so it never straddles a fetch
 * unit. That pads the 48-byte buckets of 32-bit keys with 8-byte accumulators, and of 64-bit keys
 * with 4-byte accumulators, to 64 bytes.
 *
 * @tparam Key Physical key storage type (`int32_t` or `int64_t`).
 * @tparam Value Physical accumulator storage type (at most 8 bytes).
 */
template <typename Key, typename Value>
struct alignas(cuda::std::bit_ceil(keys_first_slots *
                                   (sizeof(Key) + sizeof(Value)))) keys_first_bucket {
  Key keys[keys_first_slots];
  Value values[keys_first_slots];
};

static_assert(sizeof(keys_first_bucket<int64_t, double>) == 64,
              "the 8-byte-accumulator bucket must fill one 64-byte fetch unit");

/**
 * @brief Slot-layout policy mapping the accumulator type onto the map's storage element.
 *
 * The map's `cuco::bucket_storage` stores `storage_element`s. The accessors translate a
 * logical slot index into the element's key and accumulator addresses, so the kernels stay
 * layout-agnostic. One probing-scheme step visits one bucket: `slots_per_element` logical
 * slots.
 *
 * cuco's own slot abstraction is the storage element type itself (`cuco::pair` in its maps),
 * which fixes one slot per element and the pair's padded layout. This policy plugs the
 * keys-first and mirrored buckets into that element-type seam and adds the indirection cuco does
 * not offer. The element layout, and how logical slots map into it, becomes a per-instantiation
 * choice.
 *
 * Accumulators up to 8 bytes use the four-slot keys-first bucket. 16-byte accumulators use the
 * two-slot `mirrored_bucket` specialization below. Both store one bucket per element with all of
 * its keys adjacent, starting at the bucket's first slot.
 */
template <typename Key, typename Value>
struct slot_layout {
  using storage_element = keys_first_bucket<Key, Value>;

  /** @brief Logical slots per storage element. */
  static constexpr int32_t slots_per_element = keys_first_slots;

  /** @brief Storage elements per probing-scheme step. */
  static constexpr int32_t elements_per_probe = 1;

  /** @brief Address of the slot's key. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Key* key_ptr(storage_element* elements,
                                                              int64_t slot)
  {
    return &elements[slot / keys_first_slots].keys[slot % keys_first_slots];
  }

  /** @brief Address of the slot's key. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Key const* key_ptr(storage_element const* elements,
                                                                    int64_t slot)
  {
    return &elements[slot / keys_first_slots].keys[slot % keys_first_slots];
  }

  /** @brief Address of the slot's accumulator. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Value* value_ptr(storage_element* elements,
                                                                  int64_t slot)
  {
    return &elements[slot / keys_first_slots].values[slot % keys_first_slots];
  }

  /** @brief Address of the slot's accumulator. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Value const* value_ptr(
    storage_element const* elements, int64_t slot)
  {
    return &elements[slot / keys_first_slots].values[slot % keys_first_slots];
  }

  /** @brief Build the element every slot is initialized with before the first insert. */
  [[nodiscard]] static constexpr storage_element fill_element(Key empty_key, Value identity)
  {
    storage_element element{};
    for (int32_t i = 0; i < keys_first_slots; ++i) {
      element.keys[i]   = empty_key;
      element.values[i] = identity;
    }
    return element;
  }
};

/**
 * @brief Ordering-value and carried-payload pair of a MIN_BY/MAX_BY accumulator.
 *
 * ARGMIN/ARGMAX order groups by `value` and carry the winning row's payload. The payload is
 * type-erased to its 8 raw bytes, because the kernels never interpret it. The scan emits it
 * into a column of the declared payload type.
 *
 * @tparam Ordering Physical ordering storage type (`int64_t` or `double`).
 */
template <typename Ordering>
struct alignas(16) arg_pair {
  Ordering value;
  uint64_t payload;
};

static_assert(sizeof(arg_pair<int64_t>) == 16 && sizeof(arg_pair<double>) == 16,
              "the pair must fill exactly one 128-bit CAS operand");

/**
 * @brief Mirror-symmetric two-slot bucket for 16-byte accumulators.
 *
 * Stores two slots in 48 bytes, the same 24 bytes per slot as a packed slot, but each
 * accumulator is 16-byte aligned (never straddles a 32-byte sector, and the 128-bit CAS of
 * DECIMAL128 MIN/MAX and ARGMIN/ARGMAX requires the alignment) and the two keys sit adjacently, so
 * a bucket probe reads both from one 16-byte span.
 *
 * Measured at equal map capacity, this layout is about 1% faster than the packed 24-byte slot
 * on the SF1000 decimal insert, with the scan at parity.
 */
template <typename Key, typename Value>
struct mirrored_bucket {
  alignas(16) Value value0;
  Key key0;
  Key key1;
  alignas(16) Value value1;
};

namespace layout_checks {
using decimal_bucket = mirrored_bucket<int64_t, two_limb>;
using pair_bucket    = mirrored_bucket<int32_t, arg_pair<double>>;
static_assert(sizeof(decimal_bucket) == 48 && sizeof(pair_bucket) == 48,
              "the mirrored bucket must keep 24 bytes per slot");
static_assert(offsetof(decimal_bucket, value1) == 32 && offsetof(pair_bucket, value1) == 32,
              "both mirrored accumulators must be 16-byte aligned");
}  // namespace layout_checks

/**
 * @brief Element type the scan emits for an accumulator type.
 *
 * Most accumulators emit themselves. The ARGMIN/ARGMAX pair emits only its raw 8-byte
 * payload, which the chunk builder wraps in a column of the declared payload type.
 */
template <typename Value>
struct scan_output {
  using type = Value;
};

template <>
struct scan_output<split_sum128> {
  using type = two_limb;
};

template <typename Ordering>
struct scan_output<arg_pair<Ordering>> {
  using type = uint64_t;
};

template <typename Value>
using scan_output_t = typename scan_output<Value>::type;

/**
 * @brief Type of contributions, warp reductions, and out-of-band accumulators for an accumulator.
 *
 * Equal to the accumulator, except for the split DECIMAL128 SUM, which accumulates `two_limb`
 * values and only splits them when combining into a map slot.
 */
template <typename Value>
struct accumulation {
  using type = Value;
};

template <>
struct accumulation<split_sum128> {
  using type = two_limb;
};

template <typename Value>
using accumulation_t = typename accumulation<Value>::type;

/** @brief `true` for the ARGMIN/ARGMAX ordering-payload pair types. */
template <typename Value>
inline constexpr bool is_arg_pair = false;

template <typename Ordering>
inline constexpr bool is_arg_pair<arg_pair<Ordering>> = true;

/** @brief `true` for the 16-byte accumulators stored in a `mirrored_bucket`. */
template <typename Value>
inline constexpr bool is_mirrored_value =
  std::is_same_v<Value, two_limb> || std::is_same_v<Value, arg_pair<int64_t>> ||
  std::is_same_v<Value, arg_pair<double>>;

template <typename Key, typename Value>
  requires is_mirrored_value<Value>
struct slot_layout<Key, Value> {
  using storage_element = mirrored_bucket<Key, Value>;

  /** @brief Logical slots per storage element. */
  static constexpr int32_t slots_per_element = 2;

  /** @brief Storage elements per probing-scheme step (one whole bucket). */
  static constexpr int32_t elements_per_probe = 1;

  /** @brief Address of the slot's key. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Key* key_ptr(storage_element* elements,
                                                              int64_t slot)
  {
    auto& element = elements[slot / 2];
    return slot % 2 == 0 ? &element.key0 : &element.key1;
  }

  /** @brief Address of the slot's key. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Key const* key_ptr(storage_element const* elements,
                                                                    int64_t slot)
  {
    auto const& element = elements[slot / 2];
    return slot % 2 == 0 ? &element.key0 : &element.key1;
  }

  /** @brief Address of the slot's accumulator. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Value* value_ptr(storage_element* elements,
                                                                  int64_t slot)
  {
    auto& element = elements[slot / 2];
    return slot % 2 == 0 ? &element.value0 : &element.value1;
  }

  /** @brief Address of the slot's accumulator. */
  [[nodiscard]] static constexpr GQE_HOST_DEVICE Value const* value_ptr(
    storage_element const* elements, int64_t slot)
  {
    auto const& element = elements[slot / 2];
    return slot % 2 == 0 ? &element.value0 : &element.value1;
  }

  /** @brief Build the element every slot is initialized with before the first insert. */
  [[nodiscard]] static constexpr storage_element fill_element(Key empty_key, Value identity)
  {
    return storage_element{identity, empty_key, empty_key, identity};
  }
};

/**
 * @brief Accumulator for one out-of-band group (the NULL-key or sentinel-key group).
 *
 * The value storage is reinterpreted as the physical accumulator type, and only its leading
 * bytes are used for accumulators narrower than 16 bytes. `seen` records that the group exists
 * (at least one row carried its key). `value_valid` records that at least one non-null value
 * was combined, which decides output-value nullness for SUM/MIN/MAX.
 *
 * `alignas(16)` keeps the accumulator legal for the 128-bit CAS of a DECIMAL128 MIN/MAX.
 */
struct alignas(16) out_of_band_group {
  unsigned long long value[2];
  int32_t seen;
  int32_t value_valid;
};

/**
 * @brief Fixed-size control block leading the map state's control region.
 *
 * The **control region** is the control block followed by the occupancy histogram (see
 * `histogram_segment_slots`). `control_region_bytes` computes its total size. The block is
 * initialized together with the slots, and `finalize` reads it once after the build.
 */
struct alignas(16) control_block {
  /** @brief Accumulator of the NULL-key group. */
  out_of_band_group null_key;

  /** @brief Accumulator of the sentinel-key group. */
  out_of_band_group sentinel_key;

  /** @brief Set by a saturated insert, which violates the group-count bound. */
  int32_t bound_violation;

  /** @brief Set by a NULL in a selected row of a column declared NOT NULL. */
  int32_t not_null_violation;
};

/**
 * @brief Slots per occupancy-histogram segment (a power of two).
 *
 * The insert kernel counts each newly filled slot into a per-segment `int32_t` histogram stored
 * behind the control block. A slot transitions empty-to-filled at most once, so the count
 * costs one atomic per distinct group, measured free on the insert. The scan reads the
 * histogram once under its initialize sync and then knows every segment-aligned window's exact
 * row count, so emitting those chunks needs no per-chunk device round trip.
 *
 * The segment size balances two costs. Smaller segments give the scan finer-grained exact
 * windows but grow the histogram. Larger segments shrink it too far. At 2^20 slots a
 * TPC-H-scale map has only ~47 counters on ~3 cache lines, and the measured same-line atomic
 * contention costs 8% on the DECIMAL128 insert. At 2^16 slots the counters spread over enough
 * lines that the insert cost is not measurable, and the histogram stays at 4 bytes per 64Ki
 * slots.
 */
inline constexpr int32_t histogram_segment_log2  = 16;
inline constexpr int64_t histogram_segment_slots = int64_t{1} << histogram_segment_log2;

/**
 * @brief Number of occupancy-histogram segments covering `capacity` slots (the last may be short).
 */
[[nodiscard]] constexpr int64_t histogram_segments(int64_t capacity)
{
  return utility::divide_round_up(capacity, histogram_segment_slots);
}

/** @brief Size of the control region: the control block plus the occupancy histogram. */
[[nodiscard]] constexpr std::size_t control_region_bytes(int64_t capacity)
{
  return sizeof(control_block) +
         static_cast<std::size_t>(histogram_segments(capacity)) * sizeof(int32_t);
}

/**
 * @brief Slots of the smallest probing bucket (the mirrored bucket's two slots).
 *
 * Each probing step visits one bucket: four adjacent slots for accumulators up to 8 bytes, two for
 * the 16-byte accumulators, all of a bucket's keys checked per step. Map capacity is always a
 * multiple of the bucket's slot count.
 */
inline constexpr int32_t bucket_slots = 2;

/**
 * @brief Host-side per-chunk column arguments of one insert-kernel launch.
 *
 * All pointers are device pointers. Column data pointers are pre-adjusted for the column
 * offset. Null-mask bit indices still require the recorded offsets, because cuDF bitmasks
 * cannot be pointer-adjusted to sub-word offsets.
 *
 * @tparam Key Physical key storage type.
 * @tparam Input Physical type of the aggregated input column.
 */
template <typename Key, typename Input>
struct insert_column_args {
  Key const* keys;

  /** Key-column null mask, or `nullptr` when non-nullable. */
  uint32_t const* key_null_mask;

  int32_t key_mask_offset;

  /** Aggregated input values, or `nullptr` for COUNT aggregations without a value column. */
  Input const* values;

  /** Value-column null mask, or `nullptr` when non-nullable. */
  uint32_t const* value_null_mask;

  int32_t value_mask_offset;

  /** Carried payload as raw 8-byte values (ARGMIN/ARGMAX only), or `nullptr`. */
  uint64_t const* payload;

  /** Payload-column null mask, or `nullptr` when non-nullable. */
  uint32_t const* payload_null_mask;

  int32_t payload_mask_offset;

  /** BOOL8 valid-row mask data (non-nullable), or `nullptr` for unmasked chunks. */
  uint8_t const* row_valid;

  int32_t num_rows;
};

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
