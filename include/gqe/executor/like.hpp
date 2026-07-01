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

/**
 * @brief Implements SQL LIKE pattern matching using a Shift-And style algorithm on a
 * cudf::strings_column_view with character-aware UTF-8 matching.
 *
 * This function provides character-aware UTF-8 pattern matching, where:
 * - The `_` wildcard matches exactly one UTF-8 character (regardless of byte length)
 * - The `%` wildcard matches any sequence of UTF-8 characters
 * - Pattern matching is done at the character level, not byte level
 *
 * **Limitations**:
 * - Only supports patterns in the form `%X%` or `%X%Y%...` (must start and end with `%`)
 * - If the num of middle patterns is 0 (e.g. %%), we fall back to use cudf::like()
 * - Maximum pattern length of 64 UTF-8 characters
 * - Uses hash maps, so performance may be lower than bytewise `like()` for simple patterns
 *
 * @param input The column of strings to evaluate.
 * @param pattern The LIKE pattern string (must start and end with '%').
 * @param escape_character A string scalar representing the escape character.
 * @param stream CUDA stream on which to perform the operation.
 * @param mr Device memory resource used to allocate the result.
 *
 * @return A boolean column indicating which strings match the pattern.
 */
std::unique_ptr<cudf::column> like_utf8(
  cudf::strings_column_view const& input,
  std::string const& pattern,
  cudf::string_scalar const& escape_character = cudf::string_scalar(""),
  rmm::cuda_stream_view stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr           = cudf::get_current_device_resource_ref());

}  // end of namespace gqe
