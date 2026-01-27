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

#ifdef __CUDACC__
#define GQE_HOST_DEVICE __host__ __device__
#else
#define GQE_HOST_DEVICE
#endif

#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/cuda_device.hpp>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <semaphore>
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
template <typename type1,
          typename type2,
          std::enable_if_t<std::is_integral_v<type1>>* = nullptr,
          std::enable_if_t<std::is_integral_v<type2>>* = nullptr>
constexpr GQE_HOST_DEVICE type1 divide_round_up(type1 dividend, type2 divisor)
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

/**
 * @brief Helper function to convert a weak pointer to a shared pointer.
 *
 * @tparam T Type of the shared pointer
 * @param weak_ptr Weak pointer to convert
 * @return Shared pointer to the object
 * @throw std::runtime_error if the weak pointer expired
 */
template <typename T>
inline std::shared_ptr<T> lock_or_throw(std::weak_ptr<T> weak_ptr)
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
inline bool compare_pointer_vectors(const std::vector<T*>& v1, const std::vector<T*>& v2)
{
  return equal(begin(v1), end(v1), begin(v2), end(v2), [](const T* lhs, const T* rhs) {
    return *lhs == *rhs;
  });
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
