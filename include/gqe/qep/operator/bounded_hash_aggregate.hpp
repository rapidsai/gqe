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

#include <gqe/qep/task.hpp>
#include <gqe/qep/traits/static_iterate_split_trait.hpp>
#include <gqe/types.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

/**
 * @brief One aggregation of the bounded hash aggregate: the kind, its input columns, and their
 * declared nullability.
 *
 * Brace initialization lists all four members:
 *
 *  - Single-input kinds: `{kind, value_index, std::nullopt, value_nullability}`.
 *  - ARGMIN and ARGMAX also name the carried payload column: `{kind, ordering_index,
 *    payload_index, value_nullability}`.
 *  - `COUNT_ALL` reads no value column: `{kind, std::nullopt, std::nullopt,
 *    nullability::nullable}`.
 */
struct bounded_hash_aggregation {
  /** @brief The aggregation kind. */
  cudf::aggregation::Kind kind;

  /**
   * @brief Data-column index of the aggregated value. For ARGMIN/ARGMAX this is the ordering
   * column. Disengaged exactly for `COUNT_ALL`, which reads no value column.
   */
  std::optional<cudf::size_type> value_index;

  /**
   * @brief Data-column index of the carried payload, whose value at the group's winning ordering
   * row is the output. Engaged exactly for ARGMIN/ARGMAX.
   */
  std::optional<cudf::size_type> argument_index;

  /**
   * @brief Declared nullability of the aggregation's input columns.
   *
   *  - It covers the value column, and for ARGMIN/ARGMAX the payload column too.
   *  - Under `not_null` the operator omits its NULL handling of these columns (see `### Value
   *    columns declared NOT NULL`).
   *  - `COUNT_ALL` reads no value column, so it must stay `nullable`.
   */
  nullability value_nullability;
};

/**
 * @brief Pipelined grouped aggregate into a pre-allocated hash map.
 *
 * A **bounded hash aggregate** groups a pipelined input by a single key column and combines a
 * single aggregate value per group, directly into a device hash map. The map is pre-allocated
 * with a size based on `group_count_bound`, an upper bound on the group cardinality that is
 * typically derived from an FK/PK relationship. The map is the task's output. The paired
 * `bounded_hash_aggregate_scan_task` emits result chunks back out of it.
 *
 * # Design
 *
 * The operator pair occupies a different point in the design spectrum than the existing
 * apply-concat-apply aggregate.
 *
 * ## Relation to the apply-concat-apply aggregate
 *
 * The apply-concat-apply merge phase materializes every partial result. It stacks the
 * concatenated input, the hash-table scratch, and the output table as transient allocations,
 * so its peak memory grows with the input and spikes during the merge. This is an issue only
 * for large group cardinalities, where those transients exhaust the memory pool. The merge also
 * starts only after all partials arrived, so its work cannot overlap the input transfers.
 *
 * This operator pre-allocates the map once and aggregates each chunk on arrival:
 *
 *  - peak memory is the fixed map plus the input chunks currently inside `next` calls (how
 *    many chunks the executor keeps in flight is the executor's decision);
 *  - the aggregation overlaps completely with upstream transfer and decompression.
 *
 * ## Single-aggregate map layout
 *
 * The map state is specialized for a **single aggregate function**. `keyed_accumulator_state`
 * documents the storage layout: the slot packing per accumulator type, the value-validity,
 * wrap-counter, and control storage, and the occupancy histogram this header's chunk-geometry
 * guarantees build on. Support for multiple aggregates is deferred and will keep the
 * single-aggregate specialization as its own code path.
 *
 * ## Map sizing
 *
 * The map allocates roughly `group_count_bound / desired_load_factor` slots. Keep the default
 * load factor. Lower values buy shorter probe chains with a proportionally larger map, but
 * measurements showed flat insert cost up to a load factor of 0.9 under the double-hashing
 * probing scheme, so the default is chosen purely for footprint, which dominates at large
 * group cardinalities.
 *
 * # Semantics
 *
 * Each `next` call aggregates one regular or masked table chunk into the shared map on the
 * call's stream. There is no per-chunk output and no accumulation of chunk copies.
 *
 * `finalize` checks the bound and emits the map state unchanged.
 *
 * ## Masked rows
 *
 * Rows excluded by the valid-row mask are skipped entirely. They contribute to no group and to
 * no count.
 *
 * ## Out-of-band groups
 *
 * NULL-key rows form a single group, held in an out-of-band accumulator, because SQL groups
 * all NULL keys together.
 *
 * Rows whose key equals the key type's maximum value form a second out-of-band group. That
 * value is the map's empty-slot sentinel, so keeping it out of band makes the operator total
 * over the key domain. No input value is reserved or rejected.
 *
 * ## NULL values
 *
 * NULL values follow cuDF semantics. SUM, MIN, and MAX skip them. `COUNT_VALID` counts only
 * non-null values, and `COUNT_ALL` counts every row. A group that never combines a non-null
 * value aggregates to NULL, while counts aggregate to zero instead.
 *
 * A group whose values are all NULL still fills its map slot. It counts toward the map
 * capacity and toward the group-count bound, so the bound must cover distinct keys regardless
 * of value nullness.
 *
 * ### Value columns declared NOT NULL
 *
 * `bounded_hash_aggregation::value_nullability` declares the value column NOT NULL, and for
 * ARGMIN/ARGMAX the payload column too. `gqe::nullability` defines which rows the declaration
 * covers. Every group then combines at least one value, so the operator drops the NULL
 * handling of the value column:
 *
 *  - The insert kernel reads no value null mask and writes no value-validity byte, which saves
 *    one byte store per combine into the map.
 *  - The map allocates no value-validity storage.
 *  - The result column is never NULL.
 *
 * The declaration is checked like the group-count bound (see `# Bound violation`):
 *
 *  - `next` sets the control block's `not_null_violation` flag when a selected row is NULL.
 *    The pass is stream-ordered and runs only for a chunk whose column has NULL values.
 *  - `finalize` throws `std::runtime_error` on the flag.
 *
 * `COUNT_VALID` writes no value-validity byte in the first place, so the declaration only
 * skips its null-mask read and the kernel counts every selected row. `COUNT_ALL` has no value
 * column to declare.
 *
 * # Bound violation
 *
 * The bound is normally a hard invariant (FK/PK containment), so a violated bound is a metadata
 * bug. Violations are detected in two layers, and `finalize` throws on both:
 *
 *  - `finalize` sums the occupancy histogram (see `keyed_accumulator_state`), which counts
 *    exactly the distinct in-map groups, and
 *    compares the sum against the bound. This detects any excess, even while the map still has
 *    capacity headroom from the load factor.
 *  - The insert kernel bounds its probe loop at the map capacity, which detects saturation
 *    deterministically instead of livelocking. Saturated inserts drop their rows, so the map
 *    contents are unusable after saturation.
 *
 * The error is the only valid outcome of a violated bound.
 *
 * # Supported types
 *
 * `MEAN` never reaches the operator, because the logical optimizer rewrites it into SUM plus
 * COUNT. Unsupported shapes throw `std::logic_error` at construction, before any state is
 * allocated, so a planner must select the fallback aggregate for them.
 *
 * ## Key types
 *
 * Keys dispatch on their integer storage representation. INT32, INT64, timestamps, durations,
 * and DECIMAL32/DECIMAL64 are supported. DECIMAL128 keys are rejected, because filling a
 * 16-byte key slot needs a 128-bit CAS with no decomposition trick and no known query shape
 * needs it.
 *
 * ## Aggregate types
 *
 * The bullets below describe the supported aggregate result types per aggregation kind.
 *
 *  - `SUM` targets FLOAT32, FLOAT64, INT64, or DECIMAL128. Floating-point targets keep the
 *    source type. Integral inputs of any width accumulate into an INT64 target. DECIMAL32/64/128
 *    inputs accumulate into a DECIMAL128 target at the input's scale, following the engine's
 *    Spark-style widening. The DECIMAL128 SUM accumulator splits into an inline int64 sum in
 *    the slot and a per-slot int64 wrap counter, 8 extra bytes per slot. It never needs a
 *    16-byte atomic, because a wrap of the inline sum is detected by exactly one addition and
 *    carried into the counter.
 *  - `COUNT_VALID` and `COUNT_ALL` target INT32 or INT64. Unlike a single cuDF call, whose row
 *    count is capped by `cudf::size_type`, this operator aggregates unbounded pipelined rows
 *    into one map, so an INT32 count wraps silently once a single group receives more than
 *    2^31 - 1 counted rows. The planner is responsible for choosing an INT64 target when the
 *    input can exceed that.
 *  - `PRODUCT` and `SUM_OF_SQUARES` follow the SUM widening without the decimal arm: integral
 *    inputs of any width accumulate into an INT64 target, and floating-point targets keep the
 *    source type. Decimals are unsupported, because both operations change the scale. PRODUCT
 *    combines through a CAS loop, since no multiply atomic exists. SUM_OF_SQUARES squares
 *    after widening and reuses the SUM combine.
 *  - `ARGMIN` and `ARGMAX` implement MIN_BY/MAX_BY: they order each group by the value
 *    column and emit the payload (argument) column's value at the winning row. Ordering and
 *    payload types must both have 8-byte fixed-width storage, and `output_data_types` is
 *    `[key type, payload type, ordering type]`. A NULL payload excludes its row from the
 *    aggregation, like a NULL ordering value, because the 16-byte pair has no room to carry
 *    payload validity. A group whose ordering or payload values are all NULL emits NULL, and
 *    ties and all-NaN orderings pick an arbitrary qualifying row.
 *  - `MIN` and `MAX` target integer-storage types (including timestamps and DECIMAL32/64),
 *    FLOAT32/FLOAT64, and DECIMAL128. Integral targets use native atomics. Floating-point
 *    targets use a CAS loop over a NaN-aware total order in which NaN is the weakest element,
 *    matching the cuDF hash groupby. A NaN never wins against a real value, and an all-NaN
 *    group yields NaN.
 *
 * DECIMAL128 MIN/MAX and ARGMIN/ARGMAX combine through a 128-bit CAS on the 16-byte-aligned
 * mirrored accumulator. The instruction needs compute capability 9.0, and the build task's
 * `initialize` rejects these shapes on older devices, before any state is allocated.
 *
 * # Input shape
 *
 * Any column-bearing relation accepted by `logical_table_view`, including direct and indirect
 * columns with an optional QEP valid mask. All chunks of one input must share a logical schema.
 * Key, value, and argument indices address that logical schema. Only the distinct columns
 * consumed by the aggregation are materialized, using the supplied stream and memory resource.
 * The relation mask remains in input row coordinates and is applied by the insertion kernels.
 *
 * # Output shape
 *
 * `[keyed_accumulator]` -- the map state, consumed by the successor scan task's `initialize`.
 * The map never travels as a pipelined chunk.
 *
 * # Thread Safety
 *
 * `next` is safe to call concurrently from multiple host threads on multiple streams. The map
 * is written exclusively through device-scope atomics and no host state is mutated. The
 * executor's stream ordering of `initialize` before all `next` calls, and of all `next` calls
 * before `finalize`, provides the required cross-stream visibility.
 */
class bounded_hash_aggregate_build_task final : public fold_task {
 public:
  /**
   * @brief Construct a bounded hash aggregate build task.
   *
   * @param[in] key_index Data-column index of the single grouping key.
   * @param[in] aggregation The aggregation kind and its input-column indices (see
   *            `bounded_hash_aggregation`).
   * @param[in] output_data_types `[key type, aggregate result type]`, or
   *            `[key type, payload type, ordering type]` for ARGMIN/ARGMAX, whose result
   *            carries the payload. Taken as given, since the task never derives result
   *            types.
   * @param[in] group_count_bound Upper bound on the number of distinct non-null,
   *            non-sentinel-valued keys. Must be positive.
   * @param[in] desired_load_factor Map load factor in `(0, 1]`: the fraction of slots the
   *            bound may fill. Keep the default (see `## Map sizing`).
   *
   * @throws std::logic_error If the shape is unsupported (see `# Supported types`), an index is
   *         negative, the bound is not positive, the load factor is out of range, or
   *         `COUNT_ALL` declares its missing value column NOT NULL. These are planner errors:
   *         the planner must reject the shape before constructing the task.
   */
  bounded_hash_aggregate_build_task(cudf::size_type key_index,
                                    bounded_hash_aggregation aggregation,
                                    std::vector<cudf::data_type> output_data_types,
                                    int64_t group_count_bound,
                                    double desired_load_factor = 0.85);

  /**
   * @copydoc gqe::qep::fold_task::initialize
   */
  [[nodiscard]] state_container initialize(
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::next
   *
   * @throws std::runtime_error If a column declared NOT NULL holds a NULL value in the chunk.
   */
  void next(
    state_container_view inputs,
    state_container_view accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::fold_task::finalize
   *
   * @throws std::runtime_error If the group-count bound was violated.
   */
  [[nodiscard]] state_container finalize(
    state_container&& accumulator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  cudf::size_type _key_index;
  bounded_hash_aggregation _aggregation;
  // Immutable selection shared by all chunks. The materialized key is always column zero.
  std::vector<cudf::size_type> _materialized_indices;
  bounded_hash_aggregation _materialized_aggregation;
  std::vector<cudf::data_type> _output_data_types;
  int64_t _group_count_bound;
  double _desired_load_factor;
};

/**
 * @brief Emit the bounded hash aggregate's result chunks out of the map.
 *
 * An iterate task whose `initialize` consumes the build task's finalized map state. Each `next`
 * call acquires the next fixed-size window of map slots, compacts the filled slots into an
 * owning `[key, aggregate]` table chunk, and returns it. `std::nullopt` signals end of scan.
 * Because results leave as pipelined chunks, downstream tasks consume each chunk as it appears
 * and the full group table never needs to exist in memory. For example, a downstream filter
 * (Q18's `HAVING`) reduces every chunk to its qualifying rows on arrival.
 *
 * ## Consumption modes
 *
 * The task serves each of the two executors in its own mode:
 *
 *  - *Bounded chunks* for a pipelined executor: the unsplit task emits chunks of at most
 *    `max_rows_per_output` rows (or one chunk containing everything, under the
 *    `single_output` tag), with at most one in-flight chunk per worker. The parameters follow
 *    the `intermediate_state_read_task` conventions.
 *  - *Split scans* for the task-graph builder: the builder probes `static_iterate_split_trait`
 *    and runs N independent one-chunk split pipelines, so each slice is filtered and
 *    concatenated on its own and the full group table never materializes in one piece.
 *
 * # Chunk geometry
 *
 * Each `next` call acquires a window of at most `max_rows_per_output` slots and emits its
 * filled slots as one chunk.
 *
 *  - A chunk holds at most `max_rows_per_output` rows. The final chunk may add the two
 *    out-of-band rows.
 *  - Chunk sizes are not uniform. Uniform hashing puts the expected row count at roughly
 *    `load_factor * max_rows_per_output`, but downstream tasks must not rely on it.
 *  - Windows with no filled slots emit schema-correct empty chunks, so the output carries
 *    the schema even for an empty map.
 *  - Row bounds of at least one segment (2^16 slots) are rounded down to whole segments.
 *    Segment-aligned windows have host-known row counts, because the scan reads the
 *    occupancy histogram (see `keyed_accumulator_state`) once at `initialize` under the
 *    synchronize that already reads the out-of-band flags. Emitting those chunks allocates
 *    exactly and never waits on the device.
 *  - Smaller row bounds keep slot-granular windows and read the compacted count back per
 *    chunk, synchronizing the calling thread's stream.
 *
 * # Splitting
 *
 * The task implements `static_iterate_split_trait`. `split(index, count)` returns a task that
 * emits slice `index` of `count` near-equal, segment-aligned slot partitions as exactly one
 * chunk (possibly empty) and then ends, per the trait's one-shot contract.
 *
 *  - Segment alignment keeps every slice's row count host-known through the occupancy
 *    histogram, so split scans allocate exactly and never wait on the device.
 *  - Splits beyond the map's segment count receive empty slices.
 *  - Splits share the map state through its handle, and their outputs partition the group
 *    table. The last split carries the out-of-band group rows.
 *
 * # Output shape
 *
 * `[cudf_column (key), cudf_column (aggregate)]`. Group order across chunks is unspecified. The
 * out-of-band groups, if populated, are appended to the chunk that scans the final window: one
 * row with a NULL key, and one row keyed by the key type's maximum value. Aggregate values are
 * NULL for groups that never combined a non-null value, while counts are never NULL.
 *
 * # Thread Safety
 *
 * Concurrent `next` calls from multiple host threads on multiple streams atomically acquire
 * exhaustive, non-overlapping windows. Return order is not guaranteed, and output ordering
 * is an executor concern.
 *
 * # Lifetime
 *
 * The executor is responsible for the map state's lifetime, like for any other state. The map
 * is freed when the last state reference is dropped, which the executor orders after all
 * in-flight device work.
 */
class bounded_hash_aggregate_scan_task final : public iterate_task,
                                               public static_iterate_split_trait {
 public:
  /**
   * @brief Tag selecting one chunk containing the whole map's groups.
   */
  struct single_output_t {
    explicit constexpr single_output_t() = default;
  };

  inline static constexpr single_output_t single_output{};

  /**
   * @brief Construct a scan that emits the whole map as one chunk.
   */
  explicit bounded_hash_aggregate_scan_task(single_output_t);

  /**
   * @brief Construct a scan over the whole map with bounded chunks.
   *
   * @param[in] max_rows_per_output Positive maximum number of rows in each output. The final
   *            chunk may additionally carry the two out-of-band rows. Bounds of at least one
   *            map segment (2^16 slots) are rounded down to whole segments, which makes every
   *            chunk's row count host-known (see `# Chunk geometry`).
   *
   * @throws std::logic_error If `max_rows_per_output` is not positive.
   */
  explicit bounded_hash_aggregate_scan_task(cudf::size_type max_rows_per_output);

  /**
   * @copydoc gqe::qep::static_split_trait::max_splits
   *
   * Every count is useful: a split whose slice is empty still emits its schema-carrying empty
   * chunk, so the builder's configuration decides the count.
   */
  [[nodiscard]] cudf::size_type max_splits() const noexcept override;

  /**
   * @copydoc gqe::qep::static_split_trait::split
   *
   * @throws std::logic_error If the arguments do not satisfy `0 <= split_idx < total_splits`.
   */
  [[nodiscard]] std::unique_ptr<iterate_task> split(cudf::size_type split_idx,
                                                    cudf::size_type total_splits) const override;

  /**
   * @copydoc gqe::qep::iterate_task::initialize
   */
  [[nodiscard]] state_container initialize(
    state_container_view inputs,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::iterate_task::next
   */
  [[nodiscard]] std::optional<state_container> next(
    state_container_view iterator,
    context_reference ctx_ref,
    query_stream const& stream_handle = query_stream::default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const override;

  /**
   * @copydoc gqe::qep::task::clone
   */
  [[nodiscard]] std::unique_ptr<task> clone() const override;

 private:
  /** @brief One slice of a split scan: `index` of `count` near-equal slot partitions. */
  struct slice {
    cudf::size_type index;
    cudf::size_type count;
  };

  /** @brief Construct a split product: one slice, emitted as a single chunk. */
  bounded_hash_aggregate_scan_task(std::optional<cudf::size_type> max_rows_per_output,
                                   slice scan_slice);

  /** @brief Row bound per chunk, disengaged when the slice is emitted as one chunk. */
  std::optional<cudf::size_type> _max_rows_per_output;

  slice _slice;
};

}  // namespace qep
}  // namespace gqe
