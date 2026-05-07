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

#include <gqe/utility/linux.hpp>

#include <gqe/utility/logger.hpp>

#include <sched.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace gqe {

namespace utility {

void set_thread_affinity(cpu_set const& affinity)
{
  auto result = sched_setaffinity(
    0, CPU_ALLOC_SIZE(cpu_set::max_count), reinterpret_cast<cpu_set_t const*>(affinity.bits()));
  if (result != 0) {
    GQE_LOG_WARN("Failed to set CPU affinity for worker thread: {}", strerror(errno));
  }
}

void set_thread_affinity_fullmask()
{
  // Guaranteed thread-safe in C++11 Standard section 6.7
  // See section on Static block variables:
  // https://en.cppreference.com/w/cpp/language/storage_duration.html
  static cpu_set full_mask = []() {
    cpu_set mask;
    mask.zero();
    long num_processors = sysconf(_SC_NPROCESSORS_CONF);
    for (long i = 0; i < num_processors; ++i) {
      mask.add(i);
    }
    return mask;
  }();
  set_thread_affinity(full_mask);
}

void get_thread_affinity(cpu_set& affinity)
{
  auto result = sched_getaffinity(
    0, CPU_ALLOC_SIZE(cpu_set::max_count), reinterpret_cast<cpu_set_t*>(affinity.bits()));
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

}  // namespace utility

}  // namespace gqe
