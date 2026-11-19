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

#include <gqe/query_context.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace gqe {
namespace qep {

/**
 * @brief Type-erased owner of the map's slot storage.
 *
 * The slot storage is a typed `cuco::bucket_storage` whose element and extent types depend on
 * the dispatched key and accumulator types. This interface erases those types so the state
 * header stays free of device library includes. The typed subclass lives with the kernels, and
 * the kernel launchers recover it to build cuco storage and probing references.
 */
class keyed_accumulator_storage {
 public:
  virtual ~keyed_accumulator_storage() = default;

  /** @brief Return the device pointer to the slot array. */
  [[nodiscard]] virtual void* data() noexcept = 0;

  /** @brief Return the device pointer to the slot array. */
  [[nodiscard]] virtual void const* data() const noexcept = 0;

  /** @brief Return the number of slots. */
  [[nodiscard]] virtual int64_t capacity() const noexcept = 0;
};

/**
 * @brief Device hash-map state shared between keyed-accumulator tasks.
 *
 * Owns the device storage of a pre-allocated, open-addressing hash map that accumulates one
 * aggregate value per distinct key. The **build task** (e.g. the bounded hash aggregate build
 * task) fills the map with device-scope atomics. **Read tasks** (the scan task, and a future
 * groupjoin probe task) read or update the finalized map. The map is the pipeline-breaker state
 * between them, and no cuDF materialization happens at the boundary.
 *
 * # Storage Layout
 *
 * The state is type-erased. It owns the slot storage through the `keyed_accumulator_storage`
 * interface, raw device buffers for the auxiliary arrays, and the type descriptors that
 * read tasks need to re-dispatch onto typed kernels. The build task decides the layout
 * and sizes, and this class only tracks ownership and geometry.
 *
 * The map is specialized for a **single aggregate function**. Each slot packs the key and
 * its one accumulator in an array-of-structures layout, chosen per accumulator type. The
 * insert that fills a slot loads the co-located accumulator's cache line, which makes the
 * subsequent atomic combine nearly free, and the layout minimizes map storage. Support for
 * multiple aggregates is deferred and will keep this single-aggregate specialization as its
 * own code path.
 *
 * The state consists of four device buffers:
 *
 *  - *Slot storage*: `capacity()` logical slots `{key, accumulator}`, arranged by a
 *    slot-layout policy of the accumulator type. A slot is **empty** until an insert fills it
 *    with a key, and **filled** from then on. The empty-slot sentinel is the key type's
 *    maximum value.
 *  - *Value-validity storage*: one byte per slot, set when the slot's group has combined at
 *    least one non-null value. Empty for aggregations whose result is never null: counts, and
 *    aggregations over a value column declared NOT NULL.
 *  - *Wrap-counter storage*: one `int64_t` per slot, holding the high part of the split
 *    DECIMAL128 SUM. Empty for every other accumulator.
 *  - *Control storage*, in order:
 *     - the two out-of-band group accumulators (the NULL-key and the sentinel-key group),
 *       each carrying its own seen and value-valid flags, because these groups live outside
 *       the slot array and the per-slot arrays cannot describe them;
 *     - the bound-violation flag;
 *     - the **occupancy histogram**: one `int32_t` count of filled slots per **segment**, a
 *       span of 2^16 consecutive slots (the last segment of a map may be shorter). The build
 *       task increments a segment's count when an insert fills a slot, which happens at most
 *       once per slot. Read tasks then know every segment-aligned slot range's exact row
 *       count from one host-side copy, without synchronizing on further device work.
 *
 * ```
 * slot storage              value validity     wrap counters       control storage
 * [{key, acc} x capacity()] [u8 x capacity()]  [i64 x capacity()]  [null-key group |
 *                                                                   sentinel-key group |
 *                                                                   bound violation |
 *                                                                   occupancy histogram]
 * ```
 *
 * # Lifetime and Stream Ordering
 *
 * Multiple host threads access the map concurrently, each on its own stream, so cross-stream
 * ordering is explicit or it does not exist. The state obeys these rules:
 *
 *  - The build task's `initialize` allocates the slot storage and passes its ownership into
 *    the constructor. The constructor allocates the auxiliary buffers, on the same stream. No
 *    later call may allocate or resize map storage. The executor decides when `initialize`
 *    runs (it may defer it lazily), so the state itself never initializes lazily.
 *  - *Insert-phase writes are atomic or idempotent.* Between the build task's `initialize`
 *    and `finalize`, key inserts and accumulator updates go through device-scope atomics. The
 *    status flags (per-slot value validity, the out-of-band seen and value-valid flags, and
 *    the bound-violation flag) are plain stores that only ever write one value, which the
 *    PTX memory consistency model defines without any atomicity assumption:
 *     - As weak operations the racing stores are never *morally strong* (PTX ISA 8.7), so
 *       the single-copy-atomicity axiom does not apply to them. The argument does not need
 *       it.
 *     - Same-size data races leave the model's axioms applicable (8.7.2).
 *     - The coherence and causality axioms (8.10.1, 8.10.6) hide the initializing zero from
 *       any read ordered after the inserts, so every write such a read can observe stores
 *       the same value.
 *    No host code reads or writes map storage in this window. The executor's stream ordering
 *    of `initialize` before all `next` calls, and of all `next` calls before `finalize`,
 *    provides the required cross-stream visibility.
 *  - *The executor is responsible for the state's lifetime*, like for any other state.
 *    The map does not retain its `query_stream`; it relies on remaining QEP-owned and being
 *    destroyed before the corresponding `query_context`. The map must retain its capability if it
 *    is ever allowed to escape that scope. The executor drops the last state reference after all
 *    in-flight device work, and deallocation is stream-ordered on the buffers' recorded stream.
 *
 * # Thread Safety
 *
 * The host-side object is immutable after construction. Concurrent device-side access is the
 * accessing tasks' responsibility, per the state-container contract.
 */
class keyed_accumulator_state {
 public:
  /**
   * @brief Take ownership of the build task's slot storage and allocate the auxiliary buffers.
   *
   * The auxiliary buffers are allocated stream-ordered on `stream` and left uninitialized. The
   * build task initializes all buffers before first use: the slots with the empty-slot
   * sentinel and the aggregation identity, and the auxiliary buffers with zeros.
   *
   * @param[in] group_count_bound Upper bound on the number of distinct in-map keys.
   * @param[in] slots Allocated slot storage. Its capacity is at least `group_count_bound`.
   * @param[in] value_validity_bytes Size of the value-validity array in bytes. Zero when the
   *            aggregation result is never null.
   * @param[in] wrap_counter_bytes Size of the wrap-counter array in bytes. Zero unless the
   *            accumulator splits its value into an inline part and a per-slot wrap counter
   *            (the DECIMAL128 SUM).
   * @param[in] control_bytes Size of the control storage in bytes, including the occupancy
   *            histogram.
   * @param[in] load_factor Requested map load factor.
   * @param[in] key_type Logical cuDF type of the group key.
   * @param[in] value_type Logical cuDF type of the aggregate result.
   * @param[in] ordering_type Logical cuDF type that selects the accumulator layout. Equal to
   *            `value_type` except for ARGMIN/ARGMAX, whose accumulator is laid out by the
   *            ordering column while the result carries the payload.
   * @param[in] aggregation_kind Aggregation combined into the accumulators.
   * @param[in] stream_handle CUDA stream lifetime capability.
   * @param[in] mr Memory resource.
   *
   * @throws std::logic_error If the geometry is malformed. Validation precedes every
   *         allocation.
   */
  keyed_accumulator_state(
    int64_t group_count_bound,
    std::unique_ptr<keyed_accumulator_storage> slots,
    std::size_t value_validity_bytes,
    std::size_t wrap_counter_bytes,
    std::size_t control_bytes,
    double load_factor,
    cudf::data_type key_type,
    cudf::data_type value_type,
    cudf::data_type ordering_type,
    cudf::aggregation::Kind aggregation_kind,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /** @brief Return the distinct-key upper bound the map was sized from. */
  [[nodiscard]] int64_t group_count_bound() const noexcept { return _group_count_bound; }

  /** @brief Return the number of slots. */
  [[nodiscard]] int64_t capacity() const noexcept { return _slots->capacity(); }

  /** @brief Return the requested load factor. */
  [[nodiscard]] double load_factor() const noexcept { return _load_factor; }

  /** @brief Return the logical cuDF type of the group key. */
  [[nodiscard]] cudf::data_type key_type() const noexcept { return _key_type; }

  /** @brief Return the logical cuDF type of the aggregate result. */
  [[nodiscard]] cudf::data_type value_type() const noexcept { return _value_type; }

  /**
   * @brief Return the ordering-column type that selects the accumulator layout.
   *
   * Equal to `value_type()` except for ARGMIN/ARGMAX, whose output column carries the payload
   * type while the accumulator is laid out by the ordering type.
   */
  [[nodiscard]] cudf::data_type ordering_type() const noexcept { return _ordering_type; }

  /** @brief Return the aggregation kind combined into the accumulators. */
  [[nodiscard]] cudf::aggregation::Kind aggregation_kind() const noexcept
  {
    return _aggregation_kind;
  }

  /** @brief Return the device pointer to the slot storage. */
  [[nodiscard]] void* slots() noexcept { return _slots->data(); }

  /** @brief Return the device pointer to the slot storage. */
  [[nodiscard]] void const* slots() const noexcept { return _slots->data(); }

  /** @brief Return the typed slot storage, for the kernel launchers to recover. */
  [[nodiscard]] keyed_accumulator_storage& storage() noexcept { return *_slots; }

  /** @brief Return the typed slot storage, for the kernel launchers to recover. */
  [[nodiscard]] keyed_accumulator_storage const& storage() const noexcept { return *_slots; }

  /** @brief `true` when the state tracks per-slot value validity. */
  [[nodiscard]] bool tracks_value_validity() const noexcept { return _value_validity.size() != 0; }

  /** @brief Return the device pointer to the value-validity array, or nullptr when untracked. */
  [[nodiscard]] std::uint8_t* value_validity() noexcept
  {
    return static_cast<std::uint8_t*>(_value_validity.data());
  }

  /** @brief Return the device pointer to the value-validity array, or nullptr when untracked. */
  [[nodiscard]] std::uint8_t const* value_validity() const noexcept
  {
    return static_cast<std::uint8_t const*>(_value_validity.data());
  }

  /** @brief Return the device pointer to the wrap-counter array, or nullptr when unused. */
  [[nodiscard]] std::int64_t* wrap_counters() noexcept
  {
    return static_cast<std::int64_t*>(_wrap_counters.data());
  }

  /** @brief Return the device pointer to the wrap-counter array, or nullptr when unused. */
  [[nodiscard]] std::int64_t const* wrap_counters() const noexcept
  {
    return static_cast<std::int64_t const*>(_wrap_counters.data());
  }

  /** @brief Return the size of the wrap-counter array in bytes. */
  [[nodiscard]] std::size_t wrap_counter_bytes() const noexcept { return _wrap_counters.size(); }

  /** @brief Return the device pointer to the control storage. */
  [[nodiscard]] void* control() noexcept { return _control.data(); }

  /** @brief Return the device pointer to the control storage. */
  [[nodiscard]] void const* control() const noexcept { return _control.data(); }

 private:
  int64_t _group_count_bound;
  double _load_factor;
  cudf::data_type _key_type;
  cudf::data_type _value_type;
  cudf::data_type _ordering_type;
  cudf::aggregation::Kind _aggregation_kind;
  std::unique_ptr<keyed_accumulator_storage> _slots;
  rmm::device_buffer _value_validity;
  rmm::device_buffer _wrap_counters;
  rmm::device_buffer _control;
};

}  // namespace qep
}  // namespace gqe
