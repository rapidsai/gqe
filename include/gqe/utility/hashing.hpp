/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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