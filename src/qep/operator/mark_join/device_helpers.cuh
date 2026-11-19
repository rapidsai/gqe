/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda/std/limits>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/accumulate.cuh>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <thrust/iterator/transform_iterator.h>

#include <cstdint>
#include <type_traits>

namespace gqe::qep::mark_join_detail {

/**
 * @brief Hash a numeric column value, optionally checking for nulls.
 * @tparam hash_function Numeric value hasher constructed with a uint32_t seed.
 * @tparam has_nulls Whether to hash null elements to a fixed value without reading their payload.
 */
template <template <typename> class hash_function, bool has_nulls = false>
struct element_hasher {
 public:
  __device__ element_hasher(uint32_t seed = cudf::DEFAULT_HASH_SEED) : _seed{seed} {}

  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    if constexpr (has_nulls) {
      if (col.is_null(row_index)) {
        return cuda::std::numeric_limits<cudf::hash_value_type>::max();
      }
    }
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  __device__ cudf::hash_value_type operator()(cudf::column_device_view const& col,
                                              cudf::size_type row_index)
  {
    CUDF_UNREACHABLE("Unsupported datatype");
  }

  uint32_t _seed;
};

/**
 * @brief Combine numeric key hashes, reserving the highest bit for a match mark.
 * @tparam hash_function Numeric value hasher constructed with a uint32_t seed.
 * @tparam has_nulls Whether to inspect key validity while hashing each element.
 */
template <template <typename> class hash_function, bool has_nulls = false>
class device_row_hasher {
 public:
  __host__ __device__ device_row_hasher(cudf::table_device_view const& table,
                                        uint32_t seed = cudf::DEFAULT_HASH_SEED)
    : _table{table}, _seed{seed}
  {
  }
  __device__ cudf::hash_value_type operator()(cudf::size_type row_index) const noexcept
  {
    constexpr auto bit_mask_31 = 0x7FFFFFFF;
    auto it                    = thrust::make_transform_iterator(
      _table.begin(), [=, this](cudf::column_device_view const& col) {
        return cudf::type_dispatcher(
          col.type(), element_hasher<hash_function, has_nulls>{_seed}, col, row_index);
      });

    // Hash each element and combine all the hash values together
    auto hash =
      cudf::detail::accumulate(it, it + _table.num_columns(), _seed, [](auto hash, auto h) {
        return cudf::hashing::detail::hash_combine(hash, h);
      });
    // This ignores the most significant bit so we can use it as storage for marks.
    return hash & bit_mask_31;
  }

  uint32_t _seed;
  cudf::table_device_view _table;
};

/** @brief Compare numeric values with native equality without inspecting validity. */
class equality_comparator {
 public:
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  __device__ bool operator()(cudf::column_device_view const& lhs,
                             cudf::size_type lhs_index,
                             cudf::column_device_view const& rhs,
                             cudf::size_type rhs_index)
  {
    return lhs.element<T>(lhs_index) == rhs.element<T>(rhs_index);
  }

  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  __device__ bool operator()(cudf::column_device_view const& lhs,
                             cudf::size_type lhs_index,
                             cudf::column_device_view const& rhs,
                             cudf::size_type rhs_index)
  {
    CUDF_UNREACHABLE("Unsupported datatype");
  }
};

}  // namespace gqe::qep::mark_join_detail
