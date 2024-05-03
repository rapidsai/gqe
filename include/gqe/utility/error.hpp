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

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace gqe {

/**
 * @brief Exception thrown when a CUDA error is encountered.
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

}  // namespace gqe

#define GQE_STRINGIFY_DETAIL(x) #x
#define GQE_STRINGIFY(x)        GQE_STRINGIFY_DETAIL(x)

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
#define GQE_CUDA_TRY_1(_call) GQE_CUDA_TRY_2(_call, gqe::cuda_error)

// Adapted from cudf/utilities/error.hpp
/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing `std::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws std::logic_error
 * GQE_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * GQE_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `std::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define GQE_EXPECTS(...)                                           \
  GET_GQE_EXPECTS_MACRO(__VA_ARGS__, GQE_EXPECTS_3, GQE_EXPECTS_2) \
  (__VA_ARGS__)

#define GET_GQE_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define GQE_EXPECTS_3(_condition, _reason, _exception_type)                     \
  do {                                                                          \
    static_assert(std::is_base_of_v<std::exception, _exception_type>);          \
    (_condition) ? static_cast<void>(0)                                         \
                 : throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/ \
      {"GQE failure at: " __FILE__ ":" GQE_STRINGIFY(__LINE__) ": " _reason};   \
  } while (0)

#define GQE_EXPECTS_2(_condition, _reason) GQE_EXPECTS_3(_condition, _reason, std::logic_error)
