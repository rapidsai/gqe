/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace gqe {

/**
 * @brief Implements SQL LIKE pattern matching using a Shift-And style algorithm on a
 * cudf::strings_column_view (ascii strings).
 *
 * This function evaluates whether each string in the input column matches the given
 * pattern and returns a boolean column indicating the result of the match.
 *
 * The pattern may include:
 * - `%` (multi-character wildcard)
 * - `_` (single-character wildcard)
 * - An optional escape character
 *
 * The implementation uses a GPU-optimized Shift-And bitmask algorithm to efficiently
 * handle multiple wildcard segments, making it suitable for large-scale string matching.
 * However, the Shift-And bitmask algorithm has a limitation which requires the max
 * length of strings in the middle of multi-char wildcard should be at most 64 chars
 * due to the current max supported Shift-And state is uint64_t, so when the max length
 * of middle patterns is greater than 64, it will fall back to use cudf::like()
 *
 * @param input The column of strings to evaluate.
 * @param pattern The LIKE pattern string (may include wildcards).
 * @param escape_character A string scalar representing the escape character to be used
 *        in the pattern (if empty, no escaping is applied).
 * @param stream CUDA stream on which to perform the operation.
 * @param mr Device memory resource used to allocate the result.
 *
 * @return A boolean column (same number of rows as `input`) where each element is true
 *         if the corresponding string matches the LIKE pattern, and false otherwise.
 */
std::unique_ptr<cudf::column> like(
  cudf::strings_column_view const& input,
  std::string const& pattern,
  cudf::string_scalar const& escape_character = cudf::string_scalar(""),
  rmm::cuda_stream_view stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr           = cudf::get_current_device_resource_ref());

/**
 * @brief Implements SQL LIKE pattern matching using a Shift-And style algorithm on a
 * cudf::strings_column_view with bytewise comparison (optimized for patterns without '_').
 *
 * This function provides efficient bytewise pattern matching for UTF-8 strings:
 * - Patterns must not contain '_' wildcard, otherwise it will fall back to `like_utf8()`
 * - Literal UTF-8 characters in patterns are matched byte-by-byte (works correctly)
 * - The `%` wildcard matches any sequence of bytes
 * - Maximum pattern length of 64 bytes
 *
 * **Wildcard Handling**:
 * - Patterns WITHOUT `_` wildcards: Fast bytewise matching using 256-entry lookup table
 * - Patterns WITH `_` wildcards: Automatically falls back to `like_utf8()` for correct
 *   UTF-8 character-aware matching
 *
 * **Performance**: For patterns without `_`, this is the fastest option. For patterns
 * with `_`, it transparently uses `like_utf8()` which has hash map overhead but ensures
 * correct UTF-8 character matching.
 *
 * @param input The column of strings to evaluate.
 * @param pattern The LIKE pattern string (must start and end with '%').
 * @param escape_character A string scalar representing the escape character.
 * @param stream CUDA stream on which to perform the operation.
 * @param mr Device memory resource used to allocate the result.
 *
 * @return A boolean column indicating which strings match the pattern.
 */
std::unique_ptr<cudf::column> like_utf8_bytewise(
  cudf::strings_column_view const& input,
  std::string const& pattern,
  cudf::string_scalar const& escape_character = cudf::string_scalar(""),
  rmm::cuda_stream_view stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr           = cudf::get_current_device_resource_ref());
}  // end of namespace gqe
