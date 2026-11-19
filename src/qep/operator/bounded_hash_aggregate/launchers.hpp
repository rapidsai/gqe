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
#include <gqe/types.hpp>

#include <cuda/stream>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief Host-callable factories and kernel launchers of the bounded hash aggregate.
 *
 * Template declarations only. The definitions live in `insert_kernels.cuh` and
 * `scan_kernels.cuh`. The explicit instantiations are split across one translation unit per
 * key type (`insert_kernels_*.cu`, `scan_kernels_*.cu`), so the device compilations run in
 * parallel off the build's critical path.
 *
 * The instantiation X-macros below are the single source of truth for the supported type
 * matrix. The runtime dispatch and the instantiation files both expand them, so they cannot
 * drift apart.
 *
 * Every launcher checks the launch with `GQE_CUDA_TRY(cudaPeekAtLastError())`. Callers must
 * not launch with zero elements.
 */

/**
 * @brief Allocate the map's cuco bucket storage.
 *
 * The capacity comes from `cuco::make_valid_extent` for the probing scheme, bucket size, and
 * requested load factor. The allocation is stream-ordered on `stream` through an rmm
 * polymorphic allocator over `mr`, and the slots are left uninitialized.
 */
template <typename Key, typename Value>
[[nodiscard]] std::unique_ptr<keyed_accumulator_storage> make_map_storage(
  int64_t group_count_bound,
  double desired_load_factor,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Fill the map slots with empty-key sentinels and the aggregation identity, and reset
 *        the control block and value-validity array.
 */
template <typename Key, typename Value>
void launch_initialize_map(keyed_accumulator_state& map,
                           cudf::aggregation::Kind kind,
                           cuda::stream_ref stream);

/**
 * @brief Aggregate one input chunk into the map.
 *
 * @tparam N Declared nullability of the value column, and of the ARGMIN/ARGMAX payload column.
 */
template <typename Key, typename Input, typename Value, cudf::aggregation::Kind K, nullability N>
void launch_insert(keyed_accumulator_state& map,
                   insert_column_args<Key, Input> const& args,
                   cuda::stream_ref stream);

/**
 * @brief Set `control->not_null_violation` if a row that the valid-row mask selects is NULL in
 *        `column`.
 *
 * `row_valid` is the chunk's BOOL8 valid-row mask data, or `nullptr` when the chunk is
 * unmasked, in which case every row is selected. The pass is stream-ordered and does not
 * synchronize the host. The caller skips it for a column without NULL values.
 */
void launch_flag_selected_nulls(cudf::column_view column,
                                uint8_t const* row_valid,
                                control_block* control,
                                cuda::stream_ref stream);

/** @brief Compact the filled slots of `[begin_slot, end_slot)` into the output arrays. */
template <typename Key, typename Value>
void launch_scan(typename slot_layout<Key, Value>::storage_element const* elements,
                 int64_t begin_slot,
                 int64_t end_slot,
                 uint8_t const* value_validity,
                 int64_t const* wrap_counters,
                 Key* out_keys,
                 scan_output_t<Value>* out_values,
                 uint8_t* out_value_valid,
                 cudf::size_type* out_count,
                 cuda::stream_ref stream);

/** @brief Append the populated out-of-band group rows at `base` in the output arrays. */
template <typename Key, typename Value>
void launch_append_out_of_band(control_block const* control,
                               bool append_null_key,
                               bool append_sentinel_key,
                               cudf::size_type base,
                               Key* out_keys,
                               scan_output_t<Value>* out_values,
                               uint8_t* out_value_valid,
                               cuda::stream_ref stream);

// ----- Instantiation matrix -----

// clang-format off

/**
 * @brief The supported `(input, accumulator, kind)` insert combinations of one key type whose
 * result can be NULL, for the value nullability `N`.
 *
 * The accumulator targets follow the engine's aggregate typing:
 *
 *  - SUM of any integral width targets `int64_t` (cuDF `target_type_t` semantics).
 *  - SUM of FLOAT32/FLOAT64 targets the source type.
 *  - SUM of DECIMAL32/64/128 reps targets the split DECIMAL128 SUM accumulator.
 *  - PRODUCT and SUM_OF_SQUARES follow the SUM widening without the decimal arm.
 *  - MIN / MAX keep the source rep: `int32_t`, `int64_t`, `float`, `double`, or the
 *    two-limb DECIMAL128 accumulator (128-bit CAS, compute capability 9.0).
 *  - ARGMIN / ARGMAX read an `int64_t`- or `double`-rep ordering column plus a raw 8-byte
 *    payload column into an `arg_pair` accumulator (128-bit CAS, compute capability 9.0).
 *
 * Every combination is instantiated for a nullable value column and for one declared NOT NULL,
 * in separate translation units.
 */
#define GQE_BHA_INSERT_VALUE_INSTANTIATIONS(F, Key, N)                  \
  F(Key, int8_t,   int64_t,  cudf::aggregation::SUM, N)             \
  F(Key, int16_t,  int64_t,  cudf::aggregation::SUM, N)             \
  F(Key, int32_t,  int64_t,  cudf::aggregation::SUM, N)             \
  F(Key, int64_t,  int64_t,  cudf::aggregation::SUM, N)             \
  F(Key, float,    float,    cudf::aggregation::SUM, N)             \
  F(Key, double,   double,   cudf::aggregation::SUM, N)             \
  F(Key, int32_t,  split_sum128, cudf::aggregation::SUM, N)         \
  F(Key, int64_t,  split_sum128, cudf::aggregation::SUM, N)         \
  F(Key, two_limb, split_sum128, cudf::aggregation::SUM, N)         \
  F(Key, int32_t,  int32_t,  cudf::aggregation::MIN, N)             \
  F(Key, int64_t,  int64_t,  cudf::aggregation::MIN, N)             \
  F(Key, float,    float,    cudf::aggregation::MIN, N)             \
  F(Key, double,   double,   cudf::aggregation::MIN, N)             \
  F(Key, two_limb, two_limb, cudf::aggregation::MIN, N)             \
  F(Key, int32_t,  int32_t,  cudf::aggregation::MAX, N)             \
  F(Key, int64_t,  int64_t,  cudf::aggregation::MAX, N)             \
  F(Key, float,    float,    cudf::aggregation::MAX, N)             \
  F(Key, double,   double,   cudf::aggregation::MAX, N)             \
  F(Key, two_limb, two_limb, cudf::aggregation::MAX, N)             \
  F(Key, int8_t,   int64_t,  cudf::aggregation::PRODUCT, N)         \
  F(Key, int16_t,  int64_t,  cudf::aggregation::PRODUCT, N)         \
  F(Key, int32_t,  int64_t,  cudf::aggregation::PRODUCT, N)         \
  F(Key, int64_t,  int64_t,  cudf::aggregation::PRODUCT, N)         \
  F(Key, float,    float,    cudf::aggregation::PRODUCT, N)         \
  F(Key, double,   double,   cudf::aggregation::PRODUCT, N)         \
  F(Key, int64_t,  arg_pair<int64_t>, cudf::aggregation::ARGMIN, N) \
  F(Key, double,   arg_pair<double>,  cudf::aggregation::ARGMIN, N) \
  F(Key, int64_t,  arg_pair<int64_t>, cudf::aggregation::ARGMAX, N) \
  F(Key, double,   arg_pair<double>,  cudf::aggregation::ARGMAX, N) \
  F(Key, int8_t,   int64_t,  cudf::aggregation::SUM_OF_SQUARES, N)  \
  F(Key, int16_t,  int64_t,  cudf::aggregation::SUM_OF_SQUARES, N)  \
  F(Key, int32_t,  int64_t,  cudf::aggregation::SUM_OF_SQUARES, N)  \
  F(Key, int64_t,  int64_t,  cudf::aggregation::SUM_OF_SQUARES, N)  \
  F(Key, float,    float,    cudf::aggregation::SUM_OF_SQUARES, N)  \
  F(Key, double,   double,   cudf::aggregation::SUM_OF_SQUARES, N)

/**
 * @brief The count insert combinations of one key type.
 *
 * COUNT_VALID / COUNT_ALL target `int32_t` or `int64_t` and read no input values. A count is
 * never NULL, so the counts write no value validity and have no NOT NULL variant.
 */
#define GQE_BHA_INSERT_COUNT_INSTANTIATIONS(F, Key)                                    \
  F(Key, int32_t,  int32_t,  cudf::aggregation::COUNT_VALID, nullability::nullable) \
  F(Key, int64_t,  int64_t,  cudf::aggregation::COUNT_VALID, nullability::nullable) \
  F(Key, int32_t,  int32_t,  cudf::aggregation::COUNT_ALL, nullability::nullable)   \
  F(Key, int64_t,  int64_t,  cudf::aggregation::COUNT_ALL, nullability::nullable)

/** @brief The supported accumulator types of the storage/scan/append paths for one key type. */
#define GQE_BHA_SCAN_INSTANTIATIONS(F, Key) \
  F(Key, int32_t)                           \
  F(Key, int64_t)                           \
  F(Key, float)                             \
  F(Key, double)                            \
  F(Key, two_limb)                          \
  F(Key, split_sum128)                      \
  F(Key, arg_pair<int64_t>)                 \
  F(Key, arg_pair<double>)

// clang-format on

#define GQE_BHA_EXTERN_INSERT(Key, Input, Value, Kind, Nullability)         \
  extern template void launch_insert<Key, Input, Value, Kind, Nullability>( \
    keyed_accumulator_state&, insert_column_args<Key, Input> const&, cuda::stream_ref);

#define GQE_BHA_EXTERN_SCAN(Key, Value)                                                    \
  extern template std::unique_ptr<keyed_accumulator_storage> make_map_storage<Key, Value>( \
    int64_t, double, cuda::stream_ref, rmm::device_async_resource_ref);                    \
  extern template void launch_initialize_map<Key, Value>(                                  \
    keyed_accumulator_state&, cudf::aggregation::Kind, cuda::stream_ref);                  \
  extern template void launch_scan<Key, Value>(                                            \
    typename slot_layout<Key, Value>::storage_element const*,                              \
    int64_t,                                                                               \
    int64_t,                                                                               \
    uint8_t const*,                                                                        \
    int64_t const*,                                                                        \
    Key*,                                                                                  \
    scan_output_t<Value>*,                                                                 \
    uint8_t*,                                                                              \
    cudf::size_type*,                                                                      \
    cuda::stream_ref);                                                                     \
  extern template void launch_append_out_of_band<Key, Value>(control_block const*,         \
                                                             bool,                         \
                                                             bool,                         \
                                                             cudf::size_type,              \
                                                             Key*,                         \
                                                             scan_output_t<Value>*,        \
                                                             uint8_t*,                     \
                                                             cuda::stream_ref);

GQE_BHA_INSERT_VALUE_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int32_t, nullability::nullable)
GQE_BHA_INSERT_VALUE_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int32_t, nullability::not_null)
GQE_BHA_INSERT_COUNT_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int32_t)
GQE_BHA_INSERT_VALUE_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int64_t, nullability::nullable)
GQE_BHA_INSERT_VALUE_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int64_t, nullability::not_null)
GQE_BHA_INSERT_COUNT_INSTANTIATIONS(GQE_BHA_EXTERN_INSERT, int64_t)
GQE_BHA_SCAN_INSTANTIATIONS(GQE_BHA_EXTERN_SCAN, int32_t)
GQE_BHA_SCAN_INSTANTIATIONS(GQE_BHA_EXTERN_SCAN, int64_t)

#undef GQE_BHA_EXTERN_INSERT
#undef GQE_BHA_EXTERN_SCAN

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
