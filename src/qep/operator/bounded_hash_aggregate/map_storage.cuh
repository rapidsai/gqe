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

#include "map_layout.hpp"

#include <gqe/qep/shapes/keyed_accumulator.hpp>
#include <gqe/utility/helpers.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/storage.cuh>
#include <cuda/stream>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cstdint>
#include <memory>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief cuco-typed storage of the bounded hash aggregate map.
 *
 * The map is built on cuco components:
 *
 *  - `cuco::bucket_storage` owns the slot array, allocated through an rmm polymorphic
 *    allocator over the passed memory resource and sentinel-filled by `initialize_async`;
 *  - `cuco::make_valid_extent` sizes it for the probing scheme and load factor;
 *  - `cuco::double_hashing` over `cuco::xxhash_64` drives the probe sequence.
 *
 * The probe loop itself is custom, built on those components (see `insert_kernels.cuh`),
 * because:
 *
 *  - the loop terminates when the probing iterator wraps to its initial bucket, which detects a
 *    violated group-count bound deterministically (cuco's own insert loops livelock on a
 *    saturated map);
 *  - masked rows, NULL keys, and sentinel-valued keys are skipped or redirected inline, without
 *    a separate pre-pass;
 *  - the insert/combine split (CAS on the key, independent atomics on the accumulator) supports
 *    the two-limb DECIMAL128 accumulator. `cuco::static_map` rejects its 16-byte payload (the
 *    current cuco development branch allows it on sm_90+ only, which the sm_80 build target
 *    rules out), and `cuco::pair` would pad the packed 24-byte decimal slot to 32 bytes (see
 *    `map_layout.hpp`).
 */

/**
 * @brief xxhash_64 keyed on the physical key type.
 *
 * Chosen over the murmur3 fmix finalizer for extensibility, not speed. Both are measured
 * within 0.6% on this memory-bound insert, and both avalanche fully on single-word keys.
 * xxhash however generalizes to byte ranges, which future composite keys hash through the
 * same policy. The 64-bit output covers billion-slot capacities for both key widths.
 */
template <typename Key>
using map_hasher = cuco::xxhash_64<Key>;

/**
 * @brief Double hashing over buckets (one storage element each), one thread per probe sequence.
 *
 * Chosen over linear probing on measurement, in two regimes:
 *
 *  - On the mixed TPC-H insert (find-dominated, occupancy sweeping up to the load factor)
 *    double hashing is flat in the load factor through 0.9.
 *  - In the isolated worst case (unique keys in uniform arrival, so every probe is a miss at
 *    growing occupancy) double hashing rises gently over its cost at 0.7 (+2% at 0.9, +4% at
 *    0.95) while linear probing degrades steeply (+14% at 0.9, +45% at 0.95).
 *
 * That robustness is what allows the 0.85 default load factor and with it the smaller map.
 * cuco rounds the bucket count to a prime for this scheme, so every per-key stride cycles
 * through all buckets.
 */
template <typename Key>
using map_probing_scheme = cuco::double_hashing<1, map_hasher<Key>>;

/** @brief Slot extent produced by `cuco::make_valid_extent`; identical for both key widths. */
using map_extent =
  decltype(cuco::make_valid_extent<map_probing_scheme<int64_t>, cuco::storage<bucket_slots>>(
    cuco::extent<int64_t>{1}, 0.5));

template <typename Key, typename Value>
using map_element = typename slot_layout<Key, Value>::storage_element;

template <typename Key, typename Value>
using map_allocator = rmm::mr::polymorphic_allocator<map_element<Key, Value>>;

template <typename Key, typename Value>
using map_storage = cuco::bucket_storage<map_element<Key, Value>,
                                         slot_layout<Key, Value>::elements_per_probe,
                                         map_extent,
                                         map_allocator<Key, Value>>;

template <typename Key, typename Value>
using map_storage_ref = typename map_storage<Key, Value>::ref_type;

/**
 * @brief Typed owner of the map's cuco bucket storage.
 *
 * Implements the type-erased `keyed_accumulator_storage` interface held by
 * `keyed_accumulator_state`. The kernel launchers recover the concrete type from the state's
 * type descriptors. The storage member is constructed in place and never moved, because the
 * cuco storage's deleter references its own allocator member.
 */
template <typename Key, typename Value>
class typed_map_storage final : public keyed_accumulator_storage {
 public:
  typed_map_storage(map_extent extent, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    : _storage{extent, map_allocator<Key, Value>{mr}, stream.get()}
  {
  }

  [[nodiscard]] void* data() noexcept override { return _storage.data(); }

  [[nodiscard]] void const* data() const noexcept override { return _storage.data(); }

  [[nodiscard]] int64_t capacity() const noexcept override
  {
    // The cuco capacity counts storage elements, while the state's counts logical slots.
    return static_cast<int64_t>(_storage.capacity()) * slot_layout<Key, Value>::slots_per_element;
  }

  [[nodiscard]] map_storage<Key, Value>& storage() noexcept { return _storage; }

 private:
  map_storage<Key, Value> _storage;
};

template <typename Key, typename Value>
std::unique_ptr<keyed_accumulator_storage> make_map_storage(int64_t group_count_bound,
                                                            double desired_load_factor,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr)
{
  // The extent counts storage elements: request enough elements that the logical slot count
  // (elements times slots_per_element) covers the bound at the load factor.
  auto const requested_elements = utility::divide_round_up(
    group_count_bound, static_cast<int64_t>(slot_layout<Key, Value>::slots_per_element));
  auto const extent =
    cuco::make_valid_extent<map_probing_scheme<Key>,
                            cuco::storage<slot_layout<Key, Value>::elements_per_probe>>(
      cuco::extent<int64_t>{requested_elements}, desired_load_factor);
  return std::make_unique<typed_map_storage<Key, Value>>(extent, stream, mr);
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
