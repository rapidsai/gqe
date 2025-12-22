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

#include <gqe/device_properties.hpp>
#include <gqe/query_context.hpp>
#include <gqe/types.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>

#include <bitset>
#include <cstring>  // memcpy
#include <sstream>

#include <boost/container_hash/hash.hpp>
#include <sched.h>   // CPU_SET
#include <unistd.h>  // sysconf

namespace gqe {

compression_format compression_format_from_string(std::string const& format_str)
{
  if (format_str == "none") {
    return compression_format::none;
  } else if (format_str == "ans") {
    return compression_format::ans;
  } else if (format_str == "lz4") {
    return compression_format::lz4;
  } else if (format_str == "snappy") {
    return compression_format::snappy;
  } else if (format_str == "gdeflate") {
    return compression_format::gdeflate;
  } else if (format_str == "deflate") {
    return compression_format::deflate;
  } else if (format_str == "cascaded") {
    return compression_format::cascaded;
  } else if (format_str == "zstd") {
    return compression_format::zstd;
  } else if (format_str == "gzip") {
    return compression_format::gzip;
  } else if (format_str == "bitcomp") {
    return compression_format::bitcomp;
  } else if (format_str == "best_compression_ratio") {
    return compression_format::best_compression_ratio;
  } else if (format_str == "best_decompression_speed") {
    return compression_format::best_decompression_speed;
  } else {
    throw std::invalid_argument("Unrecognized compression format: " + format_str);
  }
}

cpu_set::cpu_set()
{
  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(CPU_ALLOC_SIZE(max_count), cpu_set);
  _cpu_set = cpu_set;
}

cpu_set::cpu_set(int cpu_id)
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(CPU_ALLOC_SIZE(max_count), cpu_set);
  CPU_SET_S(cpu_id, CPU_ALLOC_SIZE(max_count), cpu_set);

  _cpu_set = cpu_set;
}

cpu_set::cpu_set(const cpu_set_t& other, const int32_t num_cpus)
{
  assert(num_cpus <= cpu_set::max_count);

  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(CPU_ALLOC_SIZE(max_count), cpu_set);

  auto bytes = CPU_ALLOC_SIZE(num_cpus);
  std::memcpy(cpu_set, &other, bytes);

  _cpu_set = cpu_set;
}

cpu_set::cpu_set(const cpu_set& other)
{
  auto cpu_set = CPU_ALLOC(max_count);
  std::memcpy(cpu_set, other._cpu_set, CPU_ALLOC_SIZE(max_count));
  _cpu_set = cpu_set;
}

cpu_set::~cpu_set()
{
  CPU_FREE(_cpu_set);
  _cpu_set = nullptr;
}

cpu_set& cpu_set::operator=(const cpu_set& other)
{
  if (this != &other) { std::memcpy(_cpu_set, other._cpu_set, CPU_ALLOC_SIZE(max_count)); }

  return *this;
}

cpu_set& cpu_set::add(int cpu_id) noexcept
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  CPU_SET_S(cpu_id, CPU_ALLOC_SIZE(max_count), _cpu_set);

  return *this;
}

bool cpu_set::contains(int cpu_id) const noexcept
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  return CPU_ISSET_S(cpu_id, CPU_ALLOC_SIZE(max_count), _cpu_set);
}

int cpu_set::count() const noexcept { return CPU_COUNT_S(CPU_ALLOC_SIZE(max_count), _cpu_set); }

const unsigned long* cpu_set::bits() const noexcept
{
  // "man 3 CPU_SET" says that the allocation is rounded up to the next
  // multiple of sizeof(unsigned long).
  return reinterpret_cast<const unsigned long*>(_cpu_set);
}

std::string cpu_set::pretty_print() const
{
  // "man 3 CPU_SET" says that the allocation is rounded up to the next
  // multiple of sizeof(unsigned long).
  std::size_t dwords = CPU_ALLOC_SIZE(max_count) / sizeof(unsigned long);
  auto bits          = this->bits();

  std::stringstream ss;
  std::size_t i = dwords - 1;
  do {
    ss << std::bitset<sizeof(unsigned long)>(bits[i]);
  } while (i-- > 0);

  return ss.str();
}

bool cpu_set::operator==(cpu_set const& other) const noexcept
{
  return CPU_EQUAL_S(CPU_ALLOC_SIZE(max_count), _cpu_set, other._cpu_set);
}

page_kind::page_kind() {}

page_kind::page_kind(type type) : _type(type) {}

page_kind::type page_kind::value() const noexcept { return _type; };

std::size_t page_kind::size() const noexcept
{
  switch (_type) {
    case type::system_default:
    case type::small:
    case type::transparent_huge: return ::sysconf(_SC_PAGESIZE);
    case type::huge2mb: return std::size_t{1} << 21;
    case type::huge1gb: return std::size_t{1} << 30;
  };

  // Should never be reached
  return 0;
}

bool memory_kind::is_gpu_accessible(memory_kind::type type)
{
  // Check if zero copy is legal
  auto check_pageable_access =
    device_properties::instance().get<device_properties::property::pageableMemoryAccess>();

  return std::visit(
    utility::overloaded{
      [&](memory_kind::system) { return check_pageable_access; },
      [&](memory_kind::numa) { return check_pageable_access; },
      [&](memory_kind::pinned) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [&](memory_kind::numa_pinned) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      },
      [](memory_kind::device) { return true; },
      [&](memory_kind::managed) -> bool {
        return device_properties::instance().get<device_properties::property::managedMemory>();
      },
      [&](memory_kind::boost_shared) -> bool {
        return device_properties::instance().get<device_properties::property::unifiedAddressing>();
      }},
    type);
}

bool memory_kind::is_cpu_accessible(memory_kind::type type)
{
  return std::visit(utility::overloaded{
                      [&](memory_kind::system) { return true; },
                      [&](memory_kind::numa) { return true; },
                      [&](memory_kind::pinned) { return true; },
                      [&](memory_kind::numa_pinned) { return true; },
                      [&](memory_kind::managed) { return true; },
                      [&](memory_kind::boost_shared) { return true; },
                      [&](memory_kind::device) { return false; },
                    },
                    type);
}

bool memory_kind::system::operator==(system const&) const = default;

bool memory_kind::numa::operator==(numa const& other) const
{
  return numa_node_set == other.numa_node_set && page_kind == other.page_kind;
}

bool memory_kind::pinned::operator==(pinned const& other) const = default;

bool memory_kind::numa_pinned::operator==(numa_pinned const& other) const
{
  return numa_node_set == other.numa_node_set && page_kind == other.page_kind;
}

bool memory_kind::device::operator==(device const& other) const
{
  return device_id == other.device_id;
}

bool memory_kind::managed::operator==(managed const& other) const = default;

bool memory_kind::boost_shared::operator==(boost_shared const& other) const = default;

std::size_t memory_kind::type_hash::operator()(memory_kind::type const& type) const
{
  std::size_t h = boost::hash_value(type.index());
  std::visit(
    utility::overloaded{[](system const&) {},
                        [&h](numa const& n) {
                          boost::hash_combine(h, static_cast<int>(n.page_kind));
                          for (int i = 0; i < cpu_set::max_count; ++i) {
                            if (n.numa_node_set.contains(i)) { boost::hash_combine(h, i); }
                          }
                        },
                        [](pinned const&) {},
                        [&h](numa_pinned const& n) {
                          boost::hash_combine(h, static_cast<int>(n.page_kind));
                          for (int i = 0; i < cpu_set::max_count; ++i) {
                            if (n.numa_node_set.contains(i)) { boost::hash_combine(h, i); }
                          }
                        },
                        [&h](device const& d) { boost::hash_combine(h, d.device_id.value()); },
                        [](managed const&) {},
                        [](boost_shared const&) {}},
    type);
  return h;
}

}  // namespace gqe
