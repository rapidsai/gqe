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

#include <format>
#include <iostream>
#include <stdexcept>

namespace gqe {

/**
 * @brief Prints a stacktrace at the current IP location.
 * @param fd The file descriptor that will receive the stack trace.
 * @remark Requires compilation with -rdynamic to resolve symbols.
 */
void print_stacktrace(int fd) noexcept;

#define ___GQE_MAKE_ERROR_CLASS(error_name)                                 \
  struct error_name : public std::runtime_error {                           \
    error_name(const char* message) : std::runtime_error(message) {}        \
    error_name(std::string const& message) : error_name{message.c_str()} {} \
  };

___GQE_MAKE_ERROR_CLASS(cu_error);
___GQE_MAKE_ERROR_CLASS(cuda_error);
___GQE_MAKE_ERROR_CLASS(mpi_error);
___GQE_MAKE_ERROR_CLASS(nvml_error);

}  // namespace gqe

#if GQE_ENABLE_STACK_TRACE
#define ___GQE_PRINT_STACK_TRACE() print_stacktrace(STDOUT_FILENO)
#else
#define ___GQE_PRINT_STACK_TRACE()
#endif

#define ___GQE_THROW_CU_ERROR(error, file, line, throw_exception)                        \
  do {                                                                                   \
    char const* err_name = nullptr;                                                      \
    char const* err_str  = nullptr;                                                      \
    cuGetErrorName(error, &err_name);                                                    \
    cuGetErrorString(error, &err_str);                                                   \
    auto output_str =                                                                    \
      std::format("CUDA driver error at: {} {} : {} {}", file, line, err_name, err_str); \
                                                                                         \
    ___GQE_PRINT_STACK_TRACE();                                                          \
                                                                                         \
    if constexpr (throw_exception) {                                                     \
      throw gqe::cu_error(output_str);                                                   \
    } else {                                                                             \
      std::cout << output_str << std::endl;                                              \
      std::abort();                                                                      \
    }                                                                                    \
  } while (0)

#define GQE_CU_TRY_NO_THROW(_call)                                                            \
  do {                                                                                        \
    CUresult error = (func);                                                                  \
    if (::CUDA_SUCCESS != error) { ___GQE_THROW_CU_ERROR(error, __FILE__, __LINE__, false); } \
  } while (0)

#ifdef __EXCEPTIONS
#define GQE_CU_TRY(_call)                                                                    \
  do {                                                                                       \
    CUresult error = (func);                                                                 \
    if (::CUDA_SUCCESS != error) { ___GQE_THROW_CU_ERROR(error, __FILE__, __LINE__, true); } \
  } while (0)
#else
#define GQE_CU_TRY(_call) GQE_CU_TRY_NO_THROW(_call)
#endif

#define ___GQE_THROW_CUDA_ERROR(error, file, line, throw_exception)      \
  do {                                                                   \
    cudaGetLastError();                                                  \
    auto output_str = std::format("CUDA driver error at: {} {} : {} {}", \
                                  file,                                  \
                                  line,                                  \
                                  cudaGetErrorName(error),               \
                                  cudaGetErrorString(error));            \
                                                                         \
    ___GQE_PRINT_STACK_TRACE();                                          \
                                                                         \
    if constexpr (throw_exception) {                                     \
      throw gqe::cuda_error(output_str);                                 \
    } else {                                                             \
      std::cout << output_str << std::endl;                              \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

#define GQE_CUDA_TRY_NO_THROW(_call)                                                           \
  do {                                                                                         \
    cudaError_t const error = (_call);                                                         \
    if (::cudaSuccess != error) { ___GQE_THROW_CUDA_ERROR(error, __FILE__, __LINE__, false); } \
  } while (0)

#ifdef __EXCEPTIONS
#define GQE_CUDA_TRY(_call)                                                                   \
  do {                                                                                        \
    cudaError_t const error = (_call);                                                        \
    if (::cudaSuccess != error) { ___GQE_THROW_CUDA_ERROR(error, __FILE__, __LINE__, true); } \
  } while (0)
#else
#define GQE_CUDA_TRY(_call) GQE_CUDA_TRY_NO_THROW(_call)
#endif

#define ___GQE_THROW_MPI_ERROR(error, file, line, throw_exception)                             \
  do {                                                                                         \
    int len;                                                                                   \
    char estring[MPI_MAX_ERROR_STRING];                                                        \
    MPI_Error_string(error, estring, &len);                                                    \
    auto output_str = std::format("MPI error at: {} {} : {}({})", file, line, estring, error); \
                                                                                               \
    ___GQE_PRINT_STACK_TRACE();                                                                \
                                                                                               \
    if constexpr (throw_exception) {                                                           \
      throw gqe::mpi_error(output_str);                                                        \
    } else {                                                                                   \
      std::cout << output_str << std::endl;                                                    \
      std::abort();                                                                            \
    }                                                                                          \
  } while (0)

#define GQE_MPI_TRY_NO_THROW(_call)                                                          \
  do {                                                                                       \
    int const error = (_call);                                                               \
    if (MPI_SUCCESS != status) { ___GQE_THROW_MPI_ERROR(error, __FILE__, __LINE__, false); } \
  } while (0)

#ifdef __EXCEPTIONS
#define GQE_MPI_TRY(_call)                                                                  \
  do {                                                                                      \
    int const status = (_call);                                                             \
    if (MPI_SUCCESS != status) { ___GQE_THROW_MPI_ERROR(error, __FILE__, __LINE__, true); } \
  } while (0)
#else
#define GQE_MPI_TRY(_call) GQE_MPI_TRY_NO_THROW(_call)
#endif

#define ___GQE_THROW_NVML_ERROR(error, file, line, throw_exception)                           \
  do {                                                                                        \
    auto output_str =                                                                         \
      std::format("NVML error at: {} : {} : {}", __FILE__, __LINE__, nvmlErrorString(error)); \
                                                                                              \
    ___GQE_PRINT_STACK_TRACE();                                                               \
                                                                                              \
    if constexpr (throw_exception) {                                                          \
      throw gqe::nvml_error(output_str);                                                      \
    } else {                                                                                  \
      std::cout << output_str << std::endl;                                                   \
      std::abort();                                                                           \
    }                                                                                         \
  } while (0)

#define GQE_NVML_TRY_NO_THROW(_call)                                                          \
  do {                                                                                        \
    nvmlReturn_t const error = (_call);                                                       \
    if (NVML_SUCCESS != error) { ___GQE_THROW_NVML_ERROR(error, __FILE__, __LINE__, false); } \
  } while (0)

#ifdef __EXCEPTIONS
#define GQE_NVML_TRY(_call)                                                                  \
  do {                                                                                       \
    nvmlReturn_t const error = (_call);                                                      \
    if (NVML_SUCCESS != error) { ___GQE_THROW_NVML_ERROR(error, __FILE__, __LINE__, true); } \
  } while (0)
#else
#define GQE_NVML_TRY(_call) GQE_NVML_TRY_NO_THROW(_call)
#endif

/**
 * @brief A non-throwing version of GQE_EXPECTS. See \ref GQE_EXPECTS(...) for details.
 *
 * Intended to be used in destructors or when exceptions are disabled.
 */
#define GQE_EXPECTS_NO_THROW(...)                                                             \
  GET_GQE_EXPECTS_MACRO_NO_THROW(__VA_ARGS__, GQE_EXPECTS_3_NO_THROW, GQE_EXPECTS_2_NO_THROW) \
  (__VA_ARGS__)

#define GET_GQE_EXPECTS_MACRO_NO_THROW(_1, _2, _3, NAME, ...) NAME

#define GQE_EXPECTS_3_NO_THROW(_condition, _reason, _exception_type)                      \
  do {                                                                                    \
    if (!(_condition)) {                                                                  \
      ___GQE_PRINT_STACK_TRACE();                                                         \
                                                                                          \
      std::cout << std::format("GQE failure at: {}: {}: {}", __FILE__, __LINE__, _reason) \
                << std::endl;                                                             \
      std::abort();                                                                       \
    }                                                                                     \
  } while (0)

#define GQE_EXPECTS_2_NO_THROW(_condition, _reason) \
  GQE_EXPECTS_3_NO_THROW(_condition, _reason, std::logic_error)

// Adapted from cudf/utilities/error.hpp
/**
 * @def GQE_EXPECTS(...)
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
#ifdef __EXCEPTIONS
#define GQE_EXPECTS(...)                                           \
  GET_GQE_EXPECTS_MACRO(__VA_ARGS__, GQE_EXPECTS_3, GQE_EXPECTS_2) \
  (__VA_ARGS__)
#else
#define GQE_EXPECTS(...) GQE_EXPECTS_NO_THROW(_call)
#endif

#define GET_GQE_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define GQE_EXPECTS_3(_condition, _reason, _exception_type)                      \
  do {                                                                           \
    static_assert(std::is_base_of_v<std::exception, _exception_type>);           \
    if (!(_condition)) {                                                         \
      ___GQE_PRINT_STACK_TRACE();                                                \
                                                                                 \
      throw _exception_type(                                                     \
        std::format("GQE failure at: {}: {}: {}", __FILE__, __LINE__, _reason)); \
    }                                                                            \
  } while (0)

#define GQE_EXPECTS_2(_condition, _reason) GQE_EXPECTS_3(_condition, _reason, std::logic_error)
