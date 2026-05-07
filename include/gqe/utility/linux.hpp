/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/types.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>

namespace gqe {

namespace utility {

/**
 * @brief Bind the current thread to the specified CPU affinity.
 *
 * @param affinity The CPU set specifying which CPUs the thread can run on.
 */
void set_thread_affinity(cpu_set const& affinity);

/**
 * @brief Convenience function for calling set_thread_affinity() to apply full CPU mask.
 *
 */
void set_thread_affinity_fullmask();

/**
 * @brief Get the current thread cpu affinity mask.
 *
 * @param affinity Output parameter set to the current cpu affinity mask.
 */
void get_thread_affinity(cpu_set& affinity);

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
