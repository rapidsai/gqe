/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <vector>

namespace gqe {
namespace utility {

/**
 * @brief Helper function for converting a vector of smart pointers to raw pointers.
 */
template <typename smart_ptr_type>
inline std::vector<typename smart_ptr_type::element_type*> to_raw_ptrs(
  std::vector<smart_ptr_type> const& ptrs)
{
  std::vector<typename smart_ptr_type::element_type*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

/**
 * @brief Helper function for converting a vector of smart pointers to const raw pointers.
 */
template <typename smart_ptr_type>
inline std::vector<typename smart_ptr_type::element_type const*> to_const_raw_ptrs(
  std::vector<smart_ptr_type> const& ptrs)
{
  std::vector<typename smart_ptr_type::element_type const*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(const char* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};

}  // namespace utility
}  // namespace gqe

// Taken from cuCollections
#define STRINGIFY_DETAIL(x) #x
#define GQE_STRINGIFY(x)    STRINGIFY_DETAIL(x)

#define GQE_CUDA_TRY(...)                                             \
  GET_GQE_CUDA_TRY_MACRO(__VA_ARGS__, GQE_CUDA_TRY_2, GQE_CUDA_TRY_1) \
  (__VA_ARGS__)
#define GET_GQE_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define GQE_CUDA_TRY_2(_call, _exception_type)                                                    \
  do {                                                                                            \
    cudaError_t const error = (_call);                                                            \
    if (cudaSuccess != error) {                                                                   \
      cudaGetLastError();                                                                         \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + GQE_STRINGIFY(__LINE__) + \
                            ": " + cudaGetErrorName(error) + " " + cudaGetErrorString(error)};    \
    }                                                                                             \
  } while (0);
#define GQE_CUDA_TRY_1(_call) GQE_CUDA_TRY_2(_call, gqe::utility::cuda_error)  // namespace gqe
