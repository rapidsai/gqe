/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
 * @brief Capture the process-wide CPU mask used by `set_thread_affinity_fullmask`.
 *
 * The mask is read from the calling thread via `sched_getaffinity` on the first call to this
 * function or to `set_thread_affinity_fullmask`, and cached for the lifetime of the process.
 * Later calls have no effect.
 *
 * Executables call this at startup, before any thread is pinned. If the first read happens on an
 * already-pinned thread, the cached mask is that thread's narrow mask, and
 * `set_thread_affinity_fullmask` can never widen affinity again.
 *
 * If the read fails, a warning is logged and the cached mask stays empty.
 */
void capture_process_affinity();

/**
 * @brief Set the current thread's affinity to the process-wide CPU mask.
 *
 * The mask is the one cached by `capture_process_affinity`. If it has not been captured yet, it
 * is captured from the calling thread first. If the capture failed, the thread keeps its current
 * affinity.
 */
void set_thread_affinity_fullmask();

/**
 * @brief Get the current thread cpu affinity mask.
 *
 * @param affinity Output parameter set to the current cpu affinity mask.
 */
void get_thread_affinity(cpu_set& affinity);

/**
 * @brief Return CPU-bearing NUMA nodes allowed by the memory cpuset and membind policy.
 *
 * Excludes CPU-less GPU memory nodes. CPU affinity does not narrow the result because
 * memory on nodes outside the execution affinity remains usable.
 *
 * @return The matching nodes, or an empty set if NUMA is unavailable or no permitted nodes have
 * CPUs.
 * @throw std::system_error if an allowed-memory or membind query fails.
 * @throw std::runtime_error if the strict membind policy uses relative node IDs.
 */
[[nodiscard]] cpu_set cpu_bearing_numa_nodes();

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
