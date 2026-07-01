/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifdef __CUDACC__
#define GQE_HOST_DEVICE __host__ __device__
#else
#define GQE_HOST_DEVICE
#endif

#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/cuda_device.hpp>

#include <cassert>
#include <cctype>
#include <chrono>
#include <concepts>
#include <filesystem>
#include <format>
#include <limits>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace gqe::utility {

/**
 * @brief Cast an integer value to a (possibly narrower) integer type, throwing
 * `std::invalid_argument` if the value falls outside the requested range.
 *
 * @tparam T Target integer type.
 * @tparam U Source integer type. Must be wide enough to represent the full
 *         range of `T` so the default bounds are well-formed.
 * @param raw The value to cast.
 * @param context A free-form prefix included in the error message
 *        (e.g. "row count").
 * @param lower Inclusive lower bound. Defaults to `numeric_limits<T>::min()`.
 * @param upper Inclusive upper bound. Defaults to `numeric_limits<T>::max()`.
 * @return The value cast to `T`.
 * @throws std::invalid_argument if `raw` is not in `[lower, upper]`.
 */
template <std::integral T, std::integral U>
  requires(std::in_range<U>(std::numeric_limits<T>::min()) &&
           std::in_range<U>(std::numeric_limits<T>::max()))
[[nodiscard]] T checked_cast(U raw,
                             std::string_view context,
                             T lower = std::numeric_limits<T>::min(),
                             T upper = std::numeric_limits<T>::max())
{
  if (raw < lower || raw > upper)
    throw std::invalid_argument(
      std::format("{}: value {} is out of range (expected {}..{})", context, raw, lower, upper));
  return static_cast<T>(raw);
}

/**
 * @brief Return a copy of `s` with every ASCII character converted to lowercase.
 */
[[nodiscard]] inline std::string to_lower(std::string_view s)
{
  std::string n;
  n.reserve(s.size());
  std::for_each(
    s.begin(), s.end(), [&n](unsigned char c) { n.push_back(static_cast<char>(std::tolower(c))); });
  return n;
}

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
template <typename type1,
          typename type2,
          std::enable_if_t<std::is_integral_v<type1>>* = nullptr,
          std::enable_if_t<std::is_integral_v<type2>>* = nullptr>
[[nodiscard]] constexpr GQE_HOST_DEVICE type1 divide_round_up(type1 dividend, type2 divisor)
{
  assert(divisor != 0);

  return dividend / divisor + (dividend % divisor != 0);
}

/**
 * @brief Helper function for converting a vector of smart pointers to raw pointers.
 */
template <typename SmartPtr>
[[nodiscard]] inline std::vector<typename SmartPtr::element_type*> to_raw_ptrs(
  std::vector<SmartPtr> const& ptrs)
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
[[nodiscard]] inline std::vector<typename SmartPtr::element_type const*> to_const_raw_ptrs(
  std::vector<SmartPtr> const& ptrs)
{
  std::vector<typename SmartPtr::element_type const*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

/**
 * @brief Helper function to convert a weak pointer to a shared pointer.
 *
 * @tparam T Type of the shared pointer
 * @param weak_ptr Weak pointer to convert
 * @return Shared pointer to the object
 * @throw std::runtime_error if the weak pointer expired
 */
template <typename T>
[[nodiscard]] inline std::shared_ptr<T> lock_or_throw(std::weak_ptr<T> weak_ptr)
{
  auto shared_ptr = weak_ptr.lock();
  if (!shared_ptr) { throw std::runtime_error("Weak pointer expired"); }
  return shared_ptr;
}

/**
 * @brief Helper function to compare the data stored in each element (pointer) of the vectors.
 *
 * @tparam T Type of pointers
 * @param v1 First vector to compare
 * @param v2 Second vector to compare
 * @return true if the content of each vector at each index are the same, false otherwise
 */
template <typename T>
[[nodiscard]] inline bool compare_pointer_vectors(const std::vector<T*>& v1,
                                                  const std::vector<T*>& v2)
{
  return equal(begin(v1), end(v1), begin(v2), end(v2), [](const T* lhs, const T* rhs) {
    return *lhs == *rhs;
  });
}

// Return all parquet files in `path` including its subdirectories.
// Throws std::invalid_argument if `path` is not a directory or no parquet files are found.
[[nodiscard]] inline std::vector<std::string> get_parquet_files(std::string path)
{
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec))
    throw std::invalid_argument(std::format("path '{}' is not an existing directory", path));

  std::vector<std::string> parquet_files;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension().string() == ".parquet")
      parquet_files.push_back(entry.path());
  }
  if (parquet_files.empty())
    throw std::invalid_argument(std::format("no .parquet files found under path '{}'", path));
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

/**
 * @brief RAII guard for std::counting_semaphore.
 *
 * Acquires the semaphore on construction and releases it on destruction.
 */
template <std::ptrdiff_t LeastMaxValue = std::counting_semaphore<>::max()>
class semaphore_acquire_guard {
  semaphore_acquire_guard()                                          = delete;
  semaphore_acquire_guard(const semaphore_acquire_guard&)            = delete;
  semaphore_acquire_guard& operator=(const semaphore_acquire_guard&) = delete;

 public:
  explicit semaphore_acquire_guard(std::counting_semaphore<LeastMaxValue>& semaphore)
    : _semaphore(&semaphore)
  {
    _semaphore->acquire();
  }

  ~semaphore_acquire_guard()
  {
    if (_semaphore) { _semaphore->release(); }
  }

  semaphore_acquire_guard(semaphore_acquire_guard&& other) noexcept : _semaphore(other._semaphore)
  {
    other._semaphore = nullptr;
  }

  semaphore_acquire_guard& operator=(semaphore_acquire_guard&& other) noexcept
  {
    if (this != &other) { std::swap(_semaphore, other._semaphore); }
    return *this;
  }

  /**
   * @brief Check if the guard is holding the semaphore.
   *
   * Note: Used for testing.
   *
   * @return true if the semaphore is held, false otherwise
   */
  [[nodiscard]] bool is_valid() const noexcept { return _semaphore != nullptr; }

 private:
  std::counting_semaphore<LeastMaxValue>* _semaphore;
};

}  // namespace gqe::utility
