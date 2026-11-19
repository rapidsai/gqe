/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/types.hpp>

#include <sched.h>

#include <bitset>
#include <cassert>
#include <climits>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gqe {

cpu_set::cpu_set()
{
  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(byte_count, cpu_set);
  _cpu_set = cpu_set;
}

cpu_set::cpu_set(int cpu_id)
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(byte_count, cpu_set);
  CPU_SET_S(cpu_id, byte_count, cpu_set);

  _cpu_set = cpu_set;
}

cpu_set::cpu_set(const cpu_set_t& other, const int32_t num_cpus)
{
  assert(num_cpus <= cpu_set::max_count);

  auto cpu_set = CPU_ALLOC(max_count);
  CPU_ZERO_S(byte_count, cpu_set);

  auto bytes = CPU_ALLOC_SIZE(num_cpus);
  std::memcpy(cpu_set, &other, bytes);

  _cpu_set = cpu_set;
}

cpu_set::cpu_set(const cpu_set& other)
{
  auto cpu_set = CPU_ALLOC(max_count);
  std::memcpy(cpu_set, other._cpu_set, byte_count);
  _cpu_set = cpu_set;
}

cpu_set::~cpu_set()
{
  CPU_FREE(_cpu_set);
  _cpu_set = nullptr;
}

cpu_set& cpu_set::operator=(const cpu_set& other)
{
  if (this != &other) { std::memcpy(_cpu_set, other._cpu_set, byte_count); }

  return *this;
}

cpu_set& cpu_set::add(int cpu_id) noexcept
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  CPU_SET_S(cpu_id, byte_count, _cpu_set);

  return *this;
}

cpu_set& cpu_set::zero() noexcept
{
  CPU_ZERO(_cpu_set);

  return *this;
}

bool cpu_set::contains(int cpu_id) const noexcept
{
  assert(cpu_id >= 0 && cpu_id < max_count);

  return CPU_ISSET_S(cpu_id, byte_count, _cpu_set);
}

int cpu_set::count() const noexcept { return CPU_COUNT_S(byte_count, _cpu_set); }

bool cpu_set::empty() const noexcept { return count() == 0; }

int cpu_set::front() const
{
  for (int i = 0; i < max_count; ++i) {
    if (contains(i)) { return i; }
  }
  throw std::logic_error("Undefined behavior: cpu_set::front() called on empty set");
}

int cpu_set::back() const
{
  for (int i = max_count - 1; i >= 0; --i) {
    if (contains(i)) { return i; }
  }
  throw std::logic_error("Undefined behavior: cpu_set::back() called on empty set");
}

const unsigned long* cpu_set::bits() const noexcept
{
  // "man 3 CPU_SET" says that the allocation is rounded up to the next
  // multiple of sizeof(unsigned long).
  return reinterpret_cast<const unsigned long*>(_cpu_set);
}

unsigned long* cpu_set::bits() noexcept
{
  // "man 3 CPU_SET" says that the allocation is rounded up to the next
  // multiple of sizeof(unsigned long).
  return reinterpret_cast<unsigned long*>(_cpu_set);
}

std::string cpu_set::pretty_print() const
{
  // "man 3 CPU_SET" says that the allocation is rounded up to the next
  // multiple of sizeof(unsigned long).
  auto bits = this->bits();

  std::stringstream ss;
  std::size_t i = qword_count - 1;
  do {
    ss << std::bitset<sizeof(unsigned long) * CHAR_BIT>(bits[i]);
  } while (i-- > 0);

  return ss.str();
}

bool cpu_set::operator==(cpu_set const& other) const noexcept
{
  return CPU_EQUAL_S(byte_count, _cpu_set, other._cpu_set);
}

}  // namespace gqe
