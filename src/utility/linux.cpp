/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/utility/linux.hpp>

#include <gqe/utility/logger.hpp>

// Include before numaif.h so its macros do not replace the kernel's enum names.
#include <linux/mempolicy.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>

namespace gqe {

namespace utility {

void set_thread_affinity(cpu_set const& affinity)
{
  auto result =
    sched_setaffinity(0, cpu_set::byte_count, reinterpret_cast<cpu_set_t const*>(affinity.bits()));
  if (result != 0) {
    GQE_LOG_WARN("Failed to set CPU affinity for worker thread: {}", strerror(errno));
  }
}

namespace {

cpu_set const& process_affinity()
{
  // Guaranteed thread-safe in C++11 Standard section 6.7
  // See section on Static block variables:
  // https://en.cppreference.com/w/cpp/language/storage_duration.html
  //
  // A failed read leaves the cached mask empty, and `set_thread_affinity_fullmask` then keeps each
  // thread's current affinity. Retrying later would not help: the failure is not transient, and a
  // later read could come from an already-pinned thread.
  static cpu_set const full_mask = []() {
    cpu_set mask;
    if (sched_getaffinity(0, cpu_set::byte_count, reinterpret_cast<cpu_set_t*>(mask.bits())) != 0) {
      GQE_LOG_WARN("Failed to capture the process CPU affinity, so it will not be widened: {}",
                   strerror(errno));
      mask.zero();
    }
    return mask;
  }();
  return full_mask;
}

}  // namespace

void capture_process_affinity() { std::ignore = process_affinity(); }

void set_thread_affinity_fullmask()
{
  auto const& full_mask = process_affinity();
  if (full_mask.empty()) { return; }
  set_thread_affinity(full_mask);
}

void get_thread_affinity(cpu_set& affinity)
{
  auto result =
    sched_getaffinity(0, cpu_set::byte_count, reinterpret_cast<cpu_set_t*>(affinity.bits()));
  if (result != 0) {
    GQE_LOG_WARN("Failed to get CPU affinity for worker thread: {}", strerror(errno));
  }
}

template <typename CharT, typename Traits>
void check_stream(const std::basic_istream<CharT, Traits>& istream)
{
  if (istream.fail()) {
    throw std::logic_error("Looks like there's a bug in the /proc/meminfo parser.");
  }
}

std::unordered_map<std::string, std::size_t> get_meminfo(const std::string& meminfo_path)
{
  const std::unordered_map<std::string, std::size_t> unit_size = {{"kB", 1024UL}};

  std::unordered_map<std::string, std::size_t> info_map;
  std::ifstream meminfo(meminfo_path);

  std::string name, value_str;
  std::size_t value;
  std::string unit;
  bool has_unit;

  while (std::getline(meminfo, name, ':') && std::getline(meminfo, value_str)) {
    std::stringstream ss(value_str);

    check_stream(ss >> value);

    if (ss >> unit) {
      has_unit = true;
    } else {
      has_unit = false;
    }

    info_map[name] = has_unit ? value * unit_size.at(unit) : value;
  }

  return info_map;
}

namespace {

using numa_mask = std::unique_ptr<bitmask, decltype(&numa_bitmask_free)>;

}  // namespace

cpu_set cpu_bearing_numa_nodes()
{
  cpu_set numa_node_set;
  if (numa_available() < 0) { return numa_node_set; }

  numa_mask const cpus{numa_allocate_cpumask(), numa_bitmask_free};
  numa_mask const mems_allowed{numa_allocate_nodemask(), numa_bitmask_free};
  numa_mask const membind{numa_allocate_nodemask(), numa_bitmask_free};
  if (cpus == nullptr || mems_allowed == nullptr || membind == nullptr) { return numa_node_set; }

  if (::get_mempolicy(
        nullptr, mems_allowed->maskp, mems_allowed->size + 1, nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
    throw std::system_error(
      errno, std::generic_category(), "Failed to query allowed NUMA memory nodes");
  }
  int mode = MPOL_DEFAULT;
  if (::get_mempolicy(&mode, membind->maskp, membind->size + 1, nullptr, 0) != 0) {
    throw std::system_error(
      errno, std::generic_category(), "Failed to query the current NUMA memory policy");
  }
  auto const is_bound = (mode & ~MPOL_MODE_FLAGS) == MPOL_BIND;
  if (is_bound && (mode & MPOL_F_RELATIVE_NODES)) {
    throw std::runtime_error("Cannot discover usable NUMA nodes under a relative membind policy");
  }

  for (int numa_node = 0; numa_node <= numa_max_node(); ++numa_node) {
    if (!numa_bitmask_isbitset(mems_allowed.get(), static_cast<unsigned int>(numa_node)) ||
        (is_bound && !numa_bitmask_isbitset(membind.get(), static_cast<unsigned int>(numa_node)))) {
      continue;
    }
    if (numa_node_to_cpus(numa_node, cpus.get()) == 0 && numa_bitmask_weight(cpus.get()) > 0) {
      numa_node_set.add(numa_node);
    }
  }

  return numa_node_set;
}

}  // namespace utility

}  // namespace gqe
