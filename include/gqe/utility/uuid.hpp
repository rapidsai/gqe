/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace gqe::utility {

/**
 * @brief A 128-bit universally unique identifier (UUID v4).
 *
 * Stored as a 16-byte value type. The Boost UUID implementation is an
 * internal detail; only the @c .cpp includes Boost headers.
 *
 * # Thread Safety
 *
 * @ref generate() is thread-safe (uses a `thread_local` random generator).
 * All other operations are trivially safe for concurrent reads.
 */
class uuid {
 public:
  /** @brief Generate a random UUID v4. */
  [[nodiscard]] static uuid generate();

  /**
   * @brief Construct a UUID from 16 raw bytes.
   * @param bytes Exactly 16 bytes in network byte order.
   */
  [[nodiscard]] static uuid from_bytes(std::span<std::byte const, 16> bytes);

  /** @brief Return the 16 raw bytes of the UUID. */
  [[nodiscard]] std::span<std::byte const, 16> bytes() const;

  /**
   * @brief Parse a UUID from its standard hyphenated string representation.
   * @throws std::invalid_argument if the string is not a valid UUID.
   */
  [[nodiscard]] static uuid from_string(std::string_view str);

  /** @brief Return the standard hyphenated string representation. */
  [[nodiscard]] std::string to_string() const;

  bool operator==(uuid const&) const = default;

 private:
  std::array<std::byte, 16> _data{};
};

}  // namespace gqe::utility

template <>
struct std::hash<gqe::utility::uuid> {
  std::size_t operator()(gqe::utility::uuid const& u) const noexcept;
};

template <>
struct fmt::formatter<gqe::utility::uuid> : fmt::formatter<std::string> {
  auto format(gqe::utility::uuid const& u, fmt::format_context& ctx) const
    -> fmt::format_context::iterator;
};
