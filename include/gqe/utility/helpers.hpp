/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#ifdef __CUDACC__
#define GQE_HOST_DEVICE __host__ __device__
#else
#define GQE_HOST_DEVICE
#endif

#include <gqe/utility/logger.hpp>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

namespace gqe::utility {

/**
 * @brief Helper function for `std::visit`.
 *
 * @details See the [C++ reference](https://en.cppreference.com/w/cpp/utility/variant/visit) for
 * details.
 */
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

/**
 * @brief Explicit deduction guide for `std::visit`.
 *
 * @details Not needed as of C++20. See the [C++
 * reference](https://en.cppreference.com/w/cpp/utility/variant/visit) for details.
 */
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/**
 * @brief Helper function for integer division by rounding up to next-higher integer.
 */
template <typename type, std::enable_if_t<std::is_integral_v<type>>* = nullptr>
constexpr GQE_HOST_DEVICE type divide_round_up(type dividend, type divisor)
{
  assert(divisor != 0);

  return dividend / divisor + (dividend % divisor != 0);
}

/**
 * @brief Helper function for converting a vector of smart pointers to raw pointers.
 */
template <typename SmartPtr>
inline std::vector<typename SmartPtr::element_type*> to_raw_ptrs(std::vector<SmartPtr> const& ptrs)
{
  std::vector<typename SmartPtr::element_type*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

/**
 * @brief Helper function for converting a vector of smart pointers to const raw pointers.
 */
template <typename SmartPtr>
inline std::vector<typename SmartPtr::element_type const*> to_const_raw_ptrs(
  std::vector<SmartPtr> const& ptrs)
{
  std::vector<typename SmartPtr::element_type const*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

// Return all parquet files in `path` including its subdirectories
inline std::vector<std::string> get_parquet_files(std::string path)
{
  std::vector<std::string> parquet_files;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension().string() == ".parquet")
      parquet_files.push_back(entry.path());
  }
  return parquet_files;
}

// Wrapper for logging the execution time
template <typename Func, typename... Args>
inline void time_function(Func func, Args&&... args)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  func(std::forward<Args>(args)...);
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
  GQE_LOG_INFO("Query execution time: {} ms.", duration.count());
}

/**
 * @brief Convert a type to an unsigned integer type with the same size.
 */
template <typename T>
struct make_unsigned {
  using type = std::make_unsigned_t<T>;
};

template <>
struct make_unsigned<float> {
  using type = uint32_t;
};

template <>
struct make_unsigned<double> {
  using type = uint64_t;
};

template <typename T>
using make_unsigned_t = typename make_unsigned<T>::type;

}  // namespace gqe::utility
