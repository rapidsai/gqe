/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "query_common.hpp"

#include <cassert>
#include <iostream>

template <typename T,
          std::enable_if_t<std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>, bool> = true>
__device__ static T atomicAdd(T* addr, T val)
{
  static_assert(sizeof(T) == sizeof(unsigned long long int));
  return atomicAdd(reinterpret_cast<unsigned long long int*>(addr), val);
}

__device__ static int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
  using cas_type = unsigned long long int;
  static_assert(sizeof(int64_t) == sizeof(cas_type));
  return int64_t(atomicCAS(reinterpret_cast<cas_type*>(address), cas_type(compare), cas_type(val)));
}

template <typename T,
          T (*fn)(T, T),
          std::enable_if_t<sizeof(T) == sizeof(unsigned long long int), bool> = true>
__device__ static T atomicFn(T* address, T const val)
{
  static_assert(sizeof(T) == sizeof(unsigned long long int));
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old             = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed        = old;
    auto new_value = fn(val, reinterpret_cast<T&>(assumed));
    old = atomicCAS(address_as_ull, assumed, reinterpret_cast<unsigned long long int&>(new_value));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return reinterpret_cast<T&>(old);
}

template <typename T,
          T (*fn)(T, T),
          std::enable_if_t<sizeof(T) == sizeof(unsigned int), bool> = true>
__device__ static T atomicFn(T* address, T const val)
{
  static_assert(sizeof(T) == sizeof(unsigned int));
  unsigned int* address_as_int = reinterpret_cast<unsigned int*>(address);
  unsigned int old             = *address_as_int;
  unsigned int assumed;

  do {
    assumed        = old;
    auto new_value = fn(val, reinterpret_cast<T&>(assumed));
    old            = atomicCAS(address_as_int, assumed, reinterpret_cast<unsigned int&>(new_value));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return reinterpret_cast<T&>(old);
}

template <typename T>
__device__ static T atomicProduct(T* address, T const val)
{
  constexpr auto fn = [](T x, T y) { return x * y; };
  return atomicFn<T, fn>(address, val);
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
__device__ static T atomicMin(T* address, T const val)
{
  constexpr auto fn = [](T x, T y) { return min(x, y); };
  return atomicFn<T, fn>(address, val);
}

template <typename T, std::enable_if_t<std::is_same_v<T, float>, bool> = true>
__device__ static T atomicMin(T* address, T const val)
{
  constexpr auto fn = [](T x, T y) { return min(x, y); };
  return atomicFn<T, fn>(address, val);
}

template <typename T, std::enable_if_t<std::is_same_v<T, int64_t>, bool> = true>
__device__ static T atomicMin(T* address, T const val)
{
  static_assert(sizeof(long long int) == sizeof(int64_t));
  return atomicMin(reinterpret_cast<long long int*>(address),
                   reinterpret_cast<long long int const&>(val));
}

namespace libperfect {

template <typename T>
static int clz(T value)
{
  if constexpr (sizeof(T) == sizeof(unsigned int)) {
    return __builtin_clz(value);
  } else if constexpr (sizeof(T) == sizeof(unsigned long)) {
    return __builtin_clzl(value);
  } else if constexpr (sizeof(T) == sizeof(unsigned long long)) {
    return __builtin_clzll(value);
  } else {
    return __builtin_clzll(uint64_t(value)) - (sizeof(uint64_t) - sizeof(T)) * 8;
  }
}

template <typename T>
static int ffs(T value)
{
  if constexpr (sizeof(T) == sizeof(unsigned int)) {
    return __builtin_ffs(value);
  } else if constexpr (sizeof(T) == sizeof(unsigned long)) {
    return __builtin_ffsl(value);
  } else if constexpr (sizeof(T) == sizeof(unsigned long long)) {
    return __builtin_ffsll(value);
  } else {
    static_assert(sizeof(T) < 1);
  }
}

// Check that two 2D structures are equal at different rows.
template <typename table_type, typename left_row_index, typename right_row_index>
__device__ static bool equal_row(const table_type& left_table,
                                 const left_row_index& left_index,
                                 const table_type& right_table,
                                 const right_row_index& right_index,
                                 const size_t column_count)
{
  for (uint i = 0; i < column_count; i++) {
    if (left_table[i][left_index] != right_table[i][right_index]) { return false; }
  }
  return true;
}

template <typename T>
__global__ static void fill_memory_kernel(T* data, T value, size_t count)
{
  auto block_count           = gridDim.x;
  auto threads_per_block     = blockDim.x;
  auto block_index           = blockIdx.x;
  auto thread_in_block_index = threadIdx.x;
  auto thread_in_grid_index  = threads_per_block * block_index + thread_in_block_index;
  auto threads_per_grid      = threads_per_block * block_count;

  for (auto i = thread_in_grid_index; i < count; i += threads_per_grid) {
    data[i] = value;
  }
}

template <typename T>
constexpr T div_round_up(const T numerator, const T denominator)
{
  static_assert(std::is_integral<T>::value, "Integral required.");
  return (numerator + denominator - 1) / denominator;
}

template <typename T>
void fill_memory(T* data, T value, size_t count, rmm::cuda_stream_view stream)
{
  auto constexpr threads_per_block = 32;
  fill_memory_kernel<<<div_round_up(count, static_cast<size_t>(threads_per_block)),
                       threads_per_block,
                       0,
                       stream.value()>>>(data, value, count);
}

template <typename T>
__device__ static T reduce_add_sync(unsigned mask, T value)
{
  static_assert(std::is_same_v<T, unsigned> || std::is_same_v<T, int>);
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(mask, value);
#else
  value += __shfl_xor_sync(mask, value, 1);
  value += __shfl_xor_sync(mask, value, 2);
  value += __shfl_xor_sync(mask, value, 4);
  value += __shfl_xor_sync(mask, value, 8);
  value += __shfl_xor_sync(mask, value, 16);
  return value;
#endif
}

template <typename T>
__device__ static T reduce_min_sync(unsigned mask, T value)
{
  static_assert(std::is_same_v<T, unsigned> || std::is_same_v<T, int>);
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(mask, value);
#else
  value = min(value, __shfl_xor_sync(mask, value, 1));
  value = min(value, __shfl_xor_sync(mask, value, 2));
  value = min(value, __shfl_xor_sync(mask, value, 4));
  value = min(value, __shfl_xor_sync(mask, value, 8));
  value = min(value, __shfl_xor_sync(mask, value, 16));
  return value;
#endif
}

}  // namespace libperfect
