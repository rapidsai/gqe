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

#include <cstddef>
#include <string>
#include <unordered_map>

namespace gqe {

namespace utility {

/**
 * @brief Return the Linux meminfo map.
 *
 * @throw std::logic_error if the meminfo file cannot be parsed.
 *
 * Linux provides meminfo for the system in `/proc/meminfo` and per NUMA node in
 * `/sys/devices/system/node/NODE_ID/meminfo`. This function parses a meminfo file and returns the
 * contents as a name-to-value map.
 */
std::unordered_map<std::string, std::size_t> get_meminfo(const std::string& meminfo_path);

}  // namespace utility

}  // namespace gqe
