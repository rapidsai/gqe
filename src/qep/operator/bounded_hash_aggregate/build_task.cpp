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

#include "../apply_concat_apply_aggregate_common.hpp"
#include "launchers.hpp"
#include "map_layout.hpp"
#include "map_state_access.hpp"
#include "type_dispatch.hpp"

#include <gqe/device_properties.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/operator/bounded_hash_aggregate.hpp>
#include <gqe/qep/shapes/keyed_accumulator.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

namespace {

namespace bha = bounded_hash_aggregate::detail;

// -----------------------------------------------------------------------------
// Configuration and schema validation
// -----------------------------------------------------------------------------

[[nodiscard]] bool is_signed_integer_id(cudf::type_id id)
{
  return id == cudf::type_id::INT8 || id == cudf::type_id::INT16 || id == cudf::type_id::INT32 ||
         id == cudf::type_id::INT64;
}

[[nodiscard]] bool is_decimal_id(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
         id == cudf::type_id::DECIMAL128;
}

/**
 * @brief `true` when the aggregation's result can be NULL, requiring value-validity storage.
 *
 * Counts are never NULL, and a value column declared NOT NULL combines a value into every group.
 */
[[nodiscard]] bool tracks_value_validity(bounded_hash_aggregation const& aggregation)
{
  auto const kind        = aggregation.kind;
  bool const can_be_null = kind == cudf::aggregation::SUM || kind == cudf::aggregation::MIN ||
                           kind == cudf::aggregation::MAX || kind == cudf::aggregation::PRODUCT ||
                           kind == cudf::aggregation::SUM_OF_SQUARES ||
                           kind == cudf::aggregation::ARGMIN || kind == cudf::aggregation::ARGMAX;
  return can_be_null && aggregation.value_nullability == nullability::nullable;
}

/** @brief `true` when the accumulator is the split DECIMAL128 SUM, which needs wrap counters. */
[[nodiscard]] bool uses_split_sum(cudf::aggregation::Kind kind, cudf::data_type result_type)
{
  return kind == cudf::aggregation::SUM && result_type.id() == cudf::type_id::DECIMAL128;
}

/** @brief `true` for the two-input MIN_BY/MAX_BY kinds. */
[[nodiscard]] bool carries_payload(cudf::aggregation::Kind kind)
{
  return kind == cudf::aggregation::ARGMIN || kind == cudf::aggregation::ARGMAX;
}

/**
 * @brief Validate the construction-time type matrix.
 *
 * Rejecting unsupported shapes here keeps the planner's fallback decision ahead of any state
 * allocation.
 */
void validate_configuration(cudf::size_type key_index,
                            bounded_hash_aggregation const& aggregation,
                            std::vector<cudf::data_type> const& output_data_types,
                            int64_t group_count_bound,
                            double desired_load_factor)
{
  GQE_EXPECTS(key_index >= 0,
              "bounded_hash_aggregate_build_task: key_index must not be negative",
              std::logic_error);
  bool const wants_ordering_type =
    aggregation.kind == cudf::aggregation::ARGMIN || aggregation.kind == cudf::aggregation::ARGMAX;
  GQE_EXPECTS(output_data_types.size() == (wants_ordering_type ? 3u : 2u),
              "bounded_hash_aggregate_build_task: output types must be [key type, aggregate "
              "result type], plus the ordering type for ARGMIN/ARGMAX",
              std::logic_error);
  GQE_EXPECTS(group_count_bound > 0,
              "bounded_hash_aggregate_build_task: group_count_bound must be positive",
              std::logic_error);
  GQE_EXPECTS(desired_load_factor > 0.0 && desired_load_factor <= 1.0,
              "bounded_hash_aggregate_build_task: desired_load_factor must be in (0, 1]",
              std::logic_error);
  GQE_EXPECTS(bha::has_integer_storage(output_data_types[0]),
              "bounded_hash_aggregate_build_task: the key type must have INT32 or INT64 storage "
              "(DECIMAL128 keys need an undecomposable 128-bit CAS and are deferred)",
              std::logic_error);

  auto const [kind, value_column_index, argument_column_index, value_nullability] = aggregation;
  GQE_EXPECTS(value_column_index.has_value() == (kind != cudf::aggregation::COUNT_ALL),
              "bounded_hash_aggregate_build_task: the value-column index must be disengaged "
              "exactly for COUNT_ALL",
              std::logic_error);
  GQE_EXPECTS(!value_column_index.has_value() || *value_column_index >= 0,
              "bounded_hash_aggregate_build_task: value-column index must not be negative",
              std::logic_error);
  GQE_EXPECTS(argument_column_index.has_value() == carries_payload(kind),
              "bounded_hash_aggregate_build_task: the argument-column index must be engaged "
              "exactly for ARGMIN/ARGMAX",
              std::logic_error);
  GQE_EXPECTS(!argument_column_index.has_value() || *argument_column_index >= 0,
              "bounded_hash_aggregate_build_task: argument-column index must not be negative",
              std::logic_error);
  GQE_EXPECTS(value_column_index.has_value() || value_nullability == nullability::nullable,
              "bounded_hash_aggregate_build_task: COUNT_ALL reads no value column, so it cannot "
              "declare one NOT NULL",
              std::logic_error);

  auto const result_type = output_data_types[1];
  auto const result_id   = result_type.id();
  switch (kind) {
    case cudf::aggregation::SUM:
      GQE_EXPECTS(result_id == cudf::type_id::INT64 || result_id == cudf::type_id::FLOAT32 ||
                    result_id == cudf::type_id::FLOAT64 || result_id == cudf::type_id::DECIMAL128,
                  "bounded_hash_aggregate_build_task: SUM targets INT64, FLOAT32, FLOAT64, or "
                  "DECIMAL128",
                  std::logic_error);
      break;
    case cudf::aggregation::COUNT_VALID:
    case cudf::aggregation::COUNT_ALL:
      GQE_EXPECTS(result_id == cudf::type_id::INT32 || result_id == cudf::type_id::INT64,
                  "bounded_hash_aggregate_build_task: counts target INT32 or INT64",
                  std::logic_error);
      break;
    case cudf::aggregation::PRODUCT:
    case cudf::aggregation::SUM_OF_SQUARES:
      GQE_EXPECTS(result_id == cudf::type_id::INT64 || result_id == cudf::type_id::FLOAT32 ||
                    result_id == cudf::type_id::FLOAT64,
                  "bounded_hash_aggregate_build_task: PRODUCT and SUM_OF_SQUARES target INT64, "
                  "FLOAT32, or FLOAT64 (decimals change scale and are unsupported)",
                  std::logic_error);
      break;
    case cudf::aggregation::MIN:
    case cudf::aggregation::MAX:
      GQE_EXPECTS(bha::has_integer_storage(result_type) || result_id == cudf::type_id::FLOAT32 ||
                    result_id == cudf::type_id::FLOAT64 || result_id == cudf::type_id::DECIMAL128,
                  "bounded_hash_aggregate_build_task: MIN/MAX target integer-storage, "
                  "floating-point, or DECIMAL128 types",
                  std::logic_error);
      break;
    case cudf::aggregation::ARGMIN:
    case cudf::aggregation::ARGMAX:
      // The result carries the payload column's value, whose type only has to be a fixed
      // 8-byte width, because the kernels move it as raw bytes. The ordering type selects the
      // pair accumulator and must reduce to an 8-byte rep.
      GQE_EXPECTS(bha::has_8byte_storage(result_type),
                  "bounded_hash_aggregate_build_task: ARGMIN/ARGMAX payloads must have an "
                  "8-byte fixed-width type",
                  std::logic_error);
      GQE_EXPECTS(bha::has_8byte_storage(output_data_types[2]),
                  "bounded_hash_aggregate_build_task: the ARGMIN/ARGMAX ordering type must "
                  "have an 8-byte fixed-width type",
                  std::logic_error);
      break;
    default:
      GQE_EXPECTS(false,
                  std::format("bounded_hash_aggregate_build_task: unsupported aggregation "
                              "kind {}",
                              static_cast<int>(kind)),
                  std::logic_error);
  }
}

// -----------------------------------------------------------------------------
// Input resolution
// -----------------------------------------------------------------------------

/** @brief The columns of one chunk that the insert kernel reads, resolved against the task. */
struct insert_columns {
  /** @brief The group key column. */
  cudf::column_view key;

  /** @brief The aggregated value column, disengaged for `COUNT_ALL`. */
  std::optional<cudf::column_view> value;

  /** @brief The ARGMIN/ARGMAX payload column, disengaged for every other kind. */
  std::optional<cudf::column_view> payload;

  /** @brief BOOL8 valid-row mask data (non-nullable), or `nullptr` for unmasked chunks. */
  uint8_t const* row_valid;

  /** @brief Number of rows in the chunk. */
  cudf::size_type num_rows;
};

/** @brief Zero-length view used as a declared-type token for `cudf::have_same_types`. */
[[nodiscard]] cudf::column_view declared_type_token(cudf::data_type type)
{
  return cudf::column_view{type, 0, nullptr, nullptr, 0, 0, {}};
}

/**
 * @brief Validate one chunk against the declared configuration.
 *
 * @throws std::out_of_range If a configured column index exceeds the chunk width.
 * @throws std::logic_error If a column type contradicts the declared output types.
 */
void validate_chunk_schema(cudf::table_view const& table,
                           cudf::size_type key_index,
                           bounded_hash_aggregation const& aggregation,
                           std::vector<cudf::data_type> const& output_data_types)
{
  if (key_index >= table.num_columns()) {
    throw std::out_of_range(
      std::format("bounded_hash_aggregate_build_task::next: key index {} exceeds the input's "
                  "{} columns",
                  key_index,
                  table.num_columns()));
  }
  GQE_EXPECTS(
    cudf::have_same_types(table.column(key_index), declared_type_token(output_data_types[0])),
    "bounded_hash_aggregate_build_task::next: the key column type does not match the "
    "declared key output type",
    std::logic_error);

  auto const [kind, value_column_index, argument_column_index, value_nullability] = aggregation;
  if (!value_column_index.has_value()) { return; }

  if (*value_column_index >= table.num_columns()) {
    throw std::out_of_range(
      std::format("bounded_hash_aggregate_build_task::next: value index {} exceeds the input's "
                  "{} columns",
                  *value_column_index,
                  table.num_columns()));
  }
  auto const value_column = table.column(*value_column_index);
  auto const result_type  = output_data_types[1];
  switch (kind) {
    case cudf::aggregation::SUM:
      if (result_type.id() == cudf::type_id::INT64) {
        GQE_EXPECTS(is_signed_integer_id(value_column.type().id()),
                    "bounded_hash_aggregate_build_task::next: an INT64 SUM aggregates signed "
                    "integral inputs",
                    std::logic_error);
      } else if (result_type.id() == cudf::type_id::DECIMAL128) {
        GQE_EXPECTS(is_decimal_id(value_column.type().id()),
                    "bounded_hash_aggregate_build_task::next: a DECIMAL128 SUM aggregates "
                    "decimal inputs",
                    std::logic_error);
        GQE_EXPECTS(value_column.type().scale() == result_type.scale(),
                    "bounded_hash_aggregate_build_task::next: the input scale must equal the "
                    "declared DECIMAL128 output scale",
                    std::logic_error);
      } else {
        GQE_EXPECTS(cudf::have_same_types(value_column, declared_type_token(result_type)),
                    "bounded_hash_aggregate_build_task::next: a floating-point SUM keeps the "
                    "source type",
                    std::logic_error);
      }
      break;
    case cudf::aggregation::PRODUCT:
    case cudf::aggregation::SUM_OF_SQUARES:
      if (result_type.id() == cudf::type_id::INT64) {
        GQE_EXPECTS(is_signed_integer_id(value_column.type().id()),
                    "bounded_hash_aggregate_build_task::next: an INT64 PRODUCT or "
                    "SUM_OF_SQUARES aggregates signed integral inputs",
                    std::logic_error);
      } else {
        GQE_EXPECTS(cudf::have_same_types(value_column, declared_type_token(result_type)),
                    "bounded_hash_aggregate_build_task::next: a floating-point PRODUCT or "
                    "SUM_OF_SQUARES keeps the source type",
                    std::logic_error);
      }
      break;
    case cudf::aggregation::MIN:
    case cudf::aggregation::MAX:
      GQE_EXPECTS(cudf::have_same_types(value_column, declared_type_token(result_type)),
                  "bounded_hash_aggregate_build_task::next: MIN/MAX keep the source type",
                  std::logic_error);
      break;
    case cudf::aggregation::ARGMIN:
    case cudf::aggregation::ARGMAX: {
      GQE_EXPECTS(cudf::have_same_types(value_column, declared_type_token(output_data_types[2])),
                  "bounded_hash_aggregate_build_task::next: the ordering column type does not "
                  "match the declared ordering type",
                  std::logic_error);
      if (*argument_column_index >= table.num_columns()) {
        throw std::out_of_range(
          std::format("bounded_hash_aggregate_build_task::next: argument index {} exceeds the "
                      "input's {} columns",
                      *argument_column_index,
                      table.num_columns()));
      }
      auto const payload_column = table.column(*argument_column_index);
      GQE_EXPECTS(cudf::have_same_types(payload_column, declared_type_token(result_type)),
                  "bounded_hash_aggregate_build_task::next: the payload column must match the "
                  "declared output type",
                  std::logic_error);
      break;
    }
    default:
      // COUNT_VALID accepts any value-column type, because it reads only the null mask.
      break;
  }
}

/**
 * @brief Set the control block's `not_null_violation` flag if a selected row of the value or
 *        payload column is NULL (see the operator's `### Value columns declared NOT NULL`).
 *
 * A column without NULL values skips the device pass.
 */
void flag_not_null_violation(insert_columns const& columns,
                             keyed_accumulator_state& map,
                             cuda::stream_ref stream)
{
  auto* const control = static_cast<bha::control_block*>(map.control());
  for (auto const& column : {columns.value, columns.payload}) {
    if (column.has_value() && column->has_nulls()) {
      bha::launch_flag_selected_nulls(*column, columns.row_valid, control, stream);
    }
  }
}

// -----------------------------------------------------------------------------
// Storage and insert dispatch
// -----------------------------------------------------------------------------

/** @brief Typed call target of `dispatch_map_types` allocating the map storage. */
struct make_storage_fn {
  template <typename Key, typename Value>
  [[nodiscard]] std::unique_ptr<keyed_accumulator_storage> operator()(
    cudf::aggregation::Kind kind,
    int64_t group_count_bound,
    double desired_load_factor,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const
  {
    // ARGMIN/ARGMAX dispatch on the ordering type. Their accumulator is the pair.
    if constexpr (std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>) {
      if (kind == cudf::aggregation::ARGMIN || kind == cudf::aggregation::ARGMAX) {
        return bha::make_map_storage<Key, bha::arg_pair<Value>>(
          group_count_bound, desired_load_factor, stream, mr);
      }
    }
    // The DECIMAL128 SUM uses the split accumulator. MIN/MAX keep the two-limb CAS layout.
    if constexpr (std::is_same_v<Value, bha::two_limb>) {
      if (kind == cudf::aggregation::SUM) {
        return bha::make_map_storage<Key, bha::split_sum128>(
          group_count_bound, desired_load_factor, stream, mr);
      }
    }
    return bha::make_map_storage<Key, Value>(group_count_bound, desired_load_factor, stream, mr);
  }
};

/**
 * @brief Typed call target of `dispatch_map_types` initializing the map with sentinels and
 * identities.
 */
struct initialize_map_fn {
  template <typename Key, typename Value>
  void operator()(keyed_accumulator_state& map, cuda::stream_ref stream) const
  {
    auto const kind = map.aggregation_kind();
    // ARGMIN/ARGMAX dispatch on the ordering type. Their accumulator is the pair.
    if constexpr (std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>) {
      if (kind == cudf::aggregation::ARGMIN || kind == cudf::aggregation::ARGMAX) {
        bha::launch_initialize_map<Key, bha::arg_pair<Value>>(map, kind, stream);
        return;
      }
    }
    if constexpr (std::is_same_v<Value, bha::two_limb>) {
      if (kind == cudf::aggregation::SUM) {
        bha::launch_initialize_map<Key, bha::split_sum128>(map, kind, stream);
        return;
      }
    }
    bha::launch_initialize_map<Key, Value>(map, kind, stream);
  }
};

/**
 * @brief Launch the typed insert kernel over one chunk.
 *
 * The launch depends on the declared nullability of the value column and on the kind:
 *
 *  - `nullable`: the kernel reads the value and payload null masks.
 *  - `not_null`: no null mask is passed. The declaration promises that no selected row is NULL,
 *    and `flag_not_null_violation` flags a chunk that breaks the promise for `finalize`. The
 *    kernel's NOT NULL variant also writes no value validity.
 *  - Counts have no NOT NULL variant (see `GQE_BHA_INSERT_COUNT_INSTANTIATIONS`). They run the
 *    nullable variant, with the masks omitted under `not_null`.
 */
template <typename Key, typename Input, typename Value, cudf::aggregation::Kind K>
void run_insert(keyed_accumulator_state& map,
                nullability value_nullability,
                insert_columns const& columns,
                cuda::stream_ref stream)
{
  constexpr bool is_count =
    K == cudf::aggregation::COUNT_VALID || K == cudf::aggregation::COUNT_ALL;
  bool const reads_value_masks = value_nullability == nullability::nullable;

  bha::insert_column_args<Key, Input> args{};
  args.keys            = static_cast<Key const*>(columns.key.head()) + columns.key.offset();
  args.key_null_mask   = columns.key.null_mask();
  args.key_mask_offset = columns.key.offset();
  if (columns.value.has_value()) {
    if constexpr (!is_count) {
      args.values = static_cast<Input const*>(columns.value->head()) + columns.value->offset();
    }
    if (reads_value_masks) {
      args.value_null_mask   = columns.value->null_mask();
      args.value_mask_offset = columns.value->offset();
    }
  }
  if (columns.payload.has_value()) {
    args.payload =
      static_cast<uint64_t const*>(columns.payload->head()) + columns.payload->offset();
    if (reads_value_masks) {
      args.payload_null_mask   = columns.payload->null_mask();
      args.payload_mask_offset = columns.payload->offset();
    }
  }
  args.row_valid = columns.row_valid;
  args.num_rows  = columns.num_rows;

  if constexpr (is_count) {
    bha::launch_insert<Key, Input, Value, K, nullability::nullable>(map, args, stream);
  } else if (value_nullability == nullability::not_null) {
    bha::launch_insert<Key, Input, Value, K, nullability::not_null>(map, args, stream);
  } else {
    bha::launch_insert<Key, Input, Value, K, nullability::nullable>(map, args, stream);
  }
}

/**
 * @brief Inner input-type dispatch of a widening SUM, PRODUCT, or SUM_OF_SQUARES.
 *
 * Dispatched by `cudf::type_dispatcher` over the input column's logical type, with the key and
 * accumulator types already fixed by the outer dispatch. The constexpr filter admits exactly
 * the instantiated widenings: any integral rep into `int64_t` for all three kinds, and
 * DECIMAL32/64/128 reps into the split DECIMAL128 SUM accumulator for SUM only.
 */
template <typename Key, typename Value, cudf::aggregation::Kind K>
struct widening_input_fn {
  template <typename InputLogical, typename... Args>
  void operator()(Args&&... args) const
  {
    using input_rep = bha::storage_rep_t<InputLogical>;
    constexpr bool widens_to_int64 =
      std::is_same_v<Value, int64_t> &&
      (std::is_same_v<input_rep, int8_t> || std::is_same_v<input_rep, int16_t> ||
       std::is_same_v<input_rep, int32_t> || std::is_same_v<input_rep, int64_t>);
    constexpr bool widens_to_decimal128 =
      K == cudf::aggregation::SUM && std::is_same_v<Value, bha::two_limb> &&
      (std::is_same_v<input_rep, int32_t> || std::is_same_v<input_rep, int64_t> ||
       std::is_same_v<input_rep, __int128_t>);
    if constexpr (widens_to_int64) {
      run_insert<Key, input_rep, int64_t, K>(std::forward<Args>(args)...);
    } else if constexpr (widens_to_decimal128) {
      run_insert<Key, bha::kernel_value_t<input_rep>, bha::split_sum128, cudf::aggregation::SUM>(
        std::forward<Args>(args)...);
    } else {
      throw std::logic_error("bounded_hash_aggregate: unreachable widening input dispatch");
    }
  }
};

/** @brief Typed call target of `dispatch_map_types` aggregating one chunk into the map. */
struct insert_fn {
  template <typename Key, typename Value>
  void operator()(bounded_hash_aggregation const& aggregation,
                  keyed_accumulator_state& map,
                  insert_columns const& columns,
                  cuda::stream_ref stream) const
  {
    constexpr bool is_integer_value =
      std::is_same_v<Value, int32_t> || std::is_same_v<Value, int64_t>;
    constexpr bool is_float_value = std::is_same_v<Value, float> || std::is_same_v<Value, double>;

    auto const kind              = aggregation.kind;
    auto const value_nullability = aggregation.value_nullability;
    switch (kind) {
      case cudf::aggregation::SUM:
        if constexpr (is_float_value) {
          run_insert<Key, Value, Value, cudf::aggregation::SUM>(
            map, value_nullability, columns, stream);
          return;
        } else if constexpr (!std::is_same_v<Value, int32_t>) {
          // The input widens into the INT64 or DECIMAL128 accumulator.
          cudf::type_dispatcher(columns.value->type(),
                                widening_input_fn<Key, Value, cudf::aggregation::SUM>{},
                                map,
                                value_nullability,
                                columns,
                                stream);
          return;
        }
        break;
      case cudf::aggregation::PRODUCT:
        if constexpr (is_float_value) {
          run_insert<Key, Value, Value, cudf::aggregation::PRODUCT>(
            map, value_nullability, columns, stream);
          return;
        } else if constexpr (std::is_same_v<Value, int64_t>) {
          // The input widens into the INT64 accumulator.
          cudf::type_dispatcher(columns.value->type(),
                                widening_input_fn<Key, Value, cudf::aggregation::PRODUCT>{},
                                map,
                                value_nullability,
                                columns,
                                stream);
          return;
        }
        break;
      case cudf::aggregation::SUM_OF_SQUARES:
        if constexpr (is_float_value) {
          run_insert<Key, Value, Value, cudf::aggregation::SUM_OF_SQUARES>(
            map, value_nullability, columns, stream);
          return;
        } else if constexpr (std::is_same_v<Value, int64_t>) {
          // The input widens into the INT64 accumulator.
          cudf::type_dispatcher(columns.value->type(),
                                widening_input_fn<Key, Value, cudf::aggregation::SUM_OF_SQUARES>{},
                                map,
                                value_nullability,
                                columns,
                                stream);
          return;
        }
        break;
      case cudf::aggregation::COUNT_VALID:
        if constexpr (is_integer_value) {
          run_insert<Key, Value, Value, cudf::aggregation::COUNT_VALID>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      case cudf::aggregation::COUNT_ALL:
        if constexpr (is_integer_value) {
          run_insert<Key, Value, Value, cudf::aggregation::COUNT_ALL>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      case cudf::aggregation::MIN:
        if constexpr (is_integer_value || is_float_value || std::is_same_v<Value, bha::two_limb>) {
          run_insert<Key, Value, Value, cudf::aggregation::MIN>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      case cudf::aggregation::MAX:
        if constexpr (is_integer_value || is_float_value || std::is_same_v<Value, bha::two_limb>) {
          run_insert<Key, Value, Value, cudf::aggregation::MAX>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      case cudf::aggregation::ARGMIN:
        // ARGMIN/ARGMAX dispatch on the ordering type. Their accumulator is the pair.
        if constexpr (std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>) {
          run_insert<Key, Value, bha::arg_pair<Value>, cudf::aggregation::ARGMIN>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      case cudf::aggregation::ARGMAX:
        if constexpr (std::is_same_v<Value, int64_t> || std::is_same_v<Value, double>) {
          run_insert<Key, Value, bha::arg_pair<Value>, cudf::aggregation::ARGMAX>(
            map, value_nullability, columns, stream);
          return;
        }
        break;
      default: break;
    }
    throw std::logic_error("bounded_hash_aggregate: unreachable insert dispatch");
  }
};

}  // namespace

// -----------------------------------------------------------------------------
// bounded_hash_aggregate_build_task
// -----------------------------------------------------------------------------

bounded_hash_aggregate_build_task::bounded_hash_aggregate_build_task(
  cudf::size_type key_index,
  bounded_hash_aggregation aggregation,
  std::vector<cudf::data_type> output_data_types,
  int64_t group_count_bound,
  double desired_load_factor)
  : _key_index{key_index},
    _aggregation{aggregation},
    _output_data_types{std::move(output_data_types)},
    _group_count_bound{group_count_bound},
    _desired_load_factor{desired_load_factor}
{
  validate_configuration(
    _key_index, _aggregation, _output_data_types, _group_count_bound, _desired_load_factor);

  // Keep the key first and reuse its slot when an aggregate consumes the same column.
  _materialized_indices     = {_key_index};
  _materialized_aggregation = _aggregation;
  auto const remap          = [this](std::optional<cudf::size_type>& index) {
    if (!index.has_value()) { return; }
    index = detail::find_or_append_logical_index(_materialized_indices, *index);
  };
  remap(_materialized_aggregation.value_index);
  remap(_materialized_aggregation.argument_index);
}

state_container bounded_hash_aggregate_build_task::initialize(
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"bounded_hash_aggregate_build_task::initialize"};

  auto const kind = _aggregation.kind;

  // DECIMAL128 MIN/MAX and ARGMIN/ARGMAX combine through a 128-bit CAS, which needs compute
  // capability 9.0. Checked here, where the device is bound, and before any state is
  // allocated.
  bool const needs_cas_128 =
    carries_payload(kind) || ((kind == cudf::aggregation::MIN || kind == cudf::aggregation::MAX) &&
                              _output_data_types[1].id() == cudf::type_id::DECIMAL128);
  if (needs_cas_128) {
    auto const major =
      device_properties::instance().get<device_properties::computeCapabilityMajor>();
    GQE_EXPECTS(major >= 9,
                "bounded_hash_aggregate_build_task: DECIMAL128 MIN/MAX and ARGMIN/ARGMAX need "
                "a 128-bit CAS (compute capability 9.0); the planner must select the fallback "
                "aggregate on this device",
                std::logic_error);
  }

  // The kernels of the two-input ARG kinds are laid out and dispatched by the ordering
  // column's type. The declared output type is the payload's and only shapes the chunks.
  auto const ordering_type = carries_payload(kind) ? _output_data_types[2] : _output_data_types[1];

  // `cuco::make_valid_extent` sizes the map at `ceil(bound / load_factor)` slots, rounded up
  // for the probing scheme and bucket size. The range check precedes it: a metadata-bug bound
  // must fail deterministically instead of overflowing the extent or allocation arithmetic.
  // cuco performs the same division internally but validates only against 2^63, which leaves
  // its prime rounding and our byte-size arithmetic overflowable, so the check stays here.
  //
  // The limit is a power of two, so the floating-point comparison is exact. It leaves 64x
  // headroom below 2^63 for cuco's rounding (the next prime times the bucket size) and for
  // the downstream byte-size arithmetic, and it lies far beyond any allocatable map (2^57
  // 16-byte slots is 2 EiB).
  constexpr double slot_request_limit = 0x1p57;
  auto const requested_slots =
    std::ceil(static_cast<double>(_group_count_bound) / _desired_load_factor);
  GQE_EXPECTS(requested_slots < slot_request_limit,
              "bounded_hash_aggregate_build_task: group_count_bound / desired_load_factor "
              "exceeds the representable slot range");

  auto storage        = bha::dispatch_map_types(_output_data_types[0],
                                         ordering_type,
                                         make_storage_fn{},
                                         kind,
                                         _group_count_bound,
                                         _desired_load_factor,
                                         stream,
                                         mr);
  auto const capacity = storage->capacity();

  auto map = std::make_shared<keyed_accumulator_state>(
    _group_count_bound,
    std::move(storage),
    tracks_value_validity(_aggregation) ? static_cast<std::size_t>(capacity) : std::size_t{0},
    uses_split_sum(kind, _output_data_types[1])
      ? static_cast<std::size_t>(capacity) * sizeof(int64_t)
      : std::size_t{0},
    bha::control_region_bytes(capacity),
    _desired_load_factor,
    _output_data_types[0],
    _output_data_types[1],
    ordering_type,
    kind,
    stream_handle,
    mr);

  bha::dispatch_map_types(_output_data_types[0], ordering_type, initialize_map_fn{}, *map, stream);

  GQE_LOG_TRACE("Initialize QEP bounded hash aggregate: bound={}, capacity={}, kind={}",
                _group_count_bound,
                capacity,
                static_cast<int>(kind));

  return state_container_builder().add_state(state_kind::keyed_accumulator{std::move(map)}).build();
}

void bounded_hash_aggregate_build_task::next(state_container_view inputs,
                                             state_container_view accumulator,
                                             context_reference /* ctx_ref */,
                                             query_stream const& stream_handle,
                                             rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"bounded_hash_aggregate_build_task::next"};

  auto& map          = bha::get_accumulator(accumulator);
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "bounded_hash_aggregate_build_task: input is not a logical relation",
              std::logic_error);
  auto materialized     = materialize_columns(*logical, _materialized_indices, stream_handle, mr);
  auto const table      = materialized.table_view();
  auto const valid_mask = logical->valid_mask();
  validate_chunk_schema(table, 0, _materialized_aggregation, _output_data_types);

  auto const num_rows = table.num_rows();
  if (num_rows == 0) { return; }

  // Normalize a nullable QEP valid mask into plain BOOL8 bytes, because the kernel reads
  // values only.
  // The temporary is stream-ordered on `stream`, so its release cannot race the kernel.
  auto normalized =
    detail::normalize_active_mask(valid_mask.value_or(cudf::column_view{}), stream_handle, mr);
  auto const* row_valid =
    valid_mask.has_value()
      ? static_cast<uint8_t const*>(normalized.view.head()) + normalized.view.offset()
      : nullptr;

  insert_columns columns{table.column(0), {}, {}, row_valid, num_rows};
  if (_materialized_aggregation.value_index.has_value()) {
    columns.value = table.column(*_materialized_aggregation.value_index);
  }
  if (_materialized_aggregation.argument_index.has_value()) {
    columns.payload = table.column(*_materialized_aggregation.argument_index);
  }
  if (_aggregation.value_nullability == nullability::not_null) {
    flag_not_null_violation(columns, map, stream);
  }

  auto const kind = _aggregation.kind;
  bha::dispatch_map_types(_output_data_types[0],
                          carries_payload(kind) ? _output_data_types[2] : _output_data_types[1],
                          insert_fn{},
                          _aggregation,
                          map,
                          columns,
                          stream);

  order_deallocation_after_stream_use(inputs, stream_handle);

  GQE_LOG_TRACE(
    "QEP bounded hash aggregate chunk: rows={}, masked={}", num_rows, valid_mask.has_value());
}

state_container bounded_hash_aggregate_build_task::finalize(
  state_container&& accumulator,
  context_reference /* ctx_ref */,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref /* mr */) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"bounded_hash_aggregate_build_task::finalize"};

  auto const& map   = bha::get_accumulator(accumulator);
  auto const region = bha::read_control_region(map, stream);

  // Saturation first: saturated inserts dropped rows, so the group count below would be
  // meaningless. The saturation flag catches bounds violated past the load-factor headroom.
  GQE_EXPECTS(region.control.bound_violation == 0,
              std::format("bounded_hash_aggregate_build_task: the map saturated and violated "
                          "the group-count bound {}, so the bound metadata is wrong",
                          _group_count_bound),
              std::runtime_error);

  GQE_EXPECTS(region.control.not_null_violation == 0,
              "bounded_hash_aggregate_build_task: a column declared NOT NULL held a NULL in a "
              "selected row, so the nullability metadata is wrong",
              std::runtime_error);

  // The occupancy histogram counts exactly the distinct in-map groups (out-of-band groups do not
  // count toward the bound), so any excess is detected even while the map still has capacity
  // headroom from the load factor.
  int64_t distinct_groups = 0;
  for (auto const count : region.segment_counts) {
    distinct_groups += count;
  }
  GQE_EXPECTS(distinct_groups <= _group_count_bound,
              std::format("bounded_hash_aggregate_build_task: {} distinct groups exceed the "
                          "declared group-count bound {}, so the bound metadata is wrong",
                          distinct_groups,
                          _group_count_bound),
              std::runtime_error);

  return std::move(accumulator);
}

std::unique_ptr<task> bounded_hash_aggregate_build_task::clone() const
{
  return std::make_unique<bounded_hash_aggregate_build_task>(
    _key_index, _aggregation, _output_data_types, _group_count_bound, _desired_load_factor);
}

}  // namespace qep
}  // namespace gqe
