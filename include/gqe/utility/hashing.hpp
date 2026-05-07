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

#include <cstdint>

namespace gqe::utility {

extern const uint32_t crc32_table[256];

template <typename T>
uint32_t crc32(const T* data, std::size_t size_t)
{
  uint32_t crc32 = 0xFFFFFFFFu;

  for (std::size_t i = 0; i < size_t; i++) {
    const uint32_t index = (crc32 ^ data[i]) & 0xff;
    crc32 = (crc32 >> 8) ^ crc32_table[index];  // crc32_table is an array of 256 32-bit constants
  }

  // Finalize the CRC-32 value by inverting all the bits
  crc32 ^= 0xFFFFFFFFu;
  return crc32;
}

}  // namespace gqe::utility
