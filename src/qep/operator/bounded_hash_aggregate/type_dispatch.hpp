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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief Runtime type dispatch of the bounded hash aggregate.
 *
 * Reduces the operator's logical cuDF key and aggregate types onto the physical
 * representations the kernel matrix is instantiated for, and dispatches typed functors over
 * them.
 */

/**
 * @brief Compute a logical cuDF type's physical storage representation.
 *
 * Uses `cudf::device_storage_type_t`, so fixed-point decimals map onto their integer reps
 * exactly as cuDF stores them. Chrono types store themselves, so they normalize to the
 * same-width integer. Dictionaries yield `void`: their rep is an index, not a value.
 */
template <typename T>
constexpr auto storage_rep_impl()
{
  if constexpr (cudf::is_dictionary<T>()) {
    return std::type_identity<void>{};
  } else {
    using base = cudf::device_storage_type_t<T>;
    if constexpr (cudf::is_chrono<T>()) {
      if constexpr (sizeof(base) == 4) {
        return std::type_identity<int32_t>{};
      } else {
        return std::type_identity<int64_t>{};
      }
    } else {
      return std::type_identity<base>{};
    }
  }
}

template <typename T>
using storage_rep_t = typename decltype(storage_rep_impl<T>())::type;

/**
 * @brief Map a physical rep onto its kernel accumulator type (`__int128_t` uses two 64-bit limbs).
 */
template <typename Rep>
struct kernel_value {
  using type = Rep;
};

template <>
struct kernel_value<__int128_t> {
  using type = two_limb;
};

template <typename Rep>
using kernel_value_t = typename kernel_value<Rep>::type;

template <typename Rep>
inline constexpr bool is_map_key_rep = std::is_same_v<Rep, int32_t> || std::is_same_v<Rep, int64_t>;

template <typename Rep>
inline constexpr bool is_map_value_rep =
  std::is_same_v<Rep, int32_t> || std::is_same_v<Rep, int64_t> || std::is_same_v<Rep, float> ||
  std::is_same_v<Rep, double> || std::is_same_v<Rep, two_limb>;

/**
 * @brief Adapter dispatched by `cudf::double_type_dispatcher` over the logical key and
 *        aggregate types.
 *
 * Reduces the logical types to their physical reps and forwards to the functor's typed call.
 * Combinations outside the instantiated kernel matrix throw. They are unreachable, because
 * construction and schema validation admit only instantiated combinations.
 */
template <typename Functor>
struct map_type_adapter {
  template <typename KeyLogical, typename ValueLogical, typename... Args>
  auto operator()(Functor functor, Args&&... args) const
    -> decltype(functor.template operator()<int64_t, two_limb>(std::forward<Args>(args)...))
  {
    using key_rep   = storage_rep_t<KeyLogical>;
    using value_rep = kernel_value_t<storage_rep_t<ValueLogical>>;
    if constexpr (is_map_key_rep<key_rep> && is_map_value_rep<value_rep>) {
      return functor.template operator()<key_rep, value_rep>(std::forward<Args>(args)...);
    } else {
      throw std::logic_error("bounded_hash_aggregate: unreachable type dispatch");
    }
  }
};

/** @brief Dispatch `functor.operator()<KeyRep, ValueRep>(args...)` from the logical types. */
template <typename Functor, typename... Args>
auto dispatch_map_types(cudf::data_type key_type,
                        cudf::data_type value_type,
                        Functor functor,
                        Args&&... args)
{
  return cudf::double_type_dispatcher(key_type,
                                      value_type,
                                      map_type_adapter<Functor>{},
                                      std::move(functor),
                                      std::forward<Args>(args)...);
}

/** @brief Functor testing a logical type for INT32/INT64 physical storage. */
struct integer_storage_fn {
  template <typename T>
  [[nodiscard]] bool operator()() const
  {
    return is_map_key_rep<storage_rep_t<T>>;
  }
};

[[nodiscard]] inline bool has_integer_storage(cudf::data_type type)
{
  return cudf::type_dispatcher(type, integer_storage_fn{});
}

/** @brief Functor testing a logical type for a fixed 8-byte storage rep (`int64_t` or `double`). */
struct eight_byte_storage_fn {
  template <typename T>
  [[nodiscard]] bool operator()() const
  {
    using rep = storage_rep_t<T>;
    return std::is_same_v<rep, int64_t> || std::is_same_v<rep, double>;
  }
};

[[nodiscard]] inline bool has_8byte_storage(cudf::data_type type)
{
  return cudf::type_dispatcher(type, eight_byte_storage_fn{});
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
