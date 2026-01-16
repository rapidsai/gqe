/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/memory_resource/numa_memory_resource.hpp>

#include <gqe/types.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <linux/mman.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <new>  // std::bad_alloc
#include <stdexcept>
#include <string>

namespace {

std::size_t round_to_next_page(std::size_t size, std::size_t page_size) noexcept
{
  std::size_t align_mask = ~(page_size - std::size_t{1});
  return (size + page_size - std::size_t{1}) & align_mask;
}

}  // namespace

namespace gqe {

namespace memory_resource {

numa_memory_resource::numa_memory_resource(page_kind::type page_kind, bool pinned)
  : rmm::mr::device_memory_resource(), _numa_node_set(), _page_kind(page_kind), _pinned(pinned)
{
  _numa_node_set.add(0);
}

numa_memory_resource::numa_memory_resource(cpu_set numa_node_set,
                                           page_kind::type page_kind,
                                           bool pinned)
  : rmm::mr::device_memory_resource(),
    _numa_node_set(numa_node_set),
    _page_kind(page_kind),
    _pinned(pinned)
{
}

void* numa_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view)
{
  if (0 == bytes) { return nullptr; }

  int hugetlbfs_flags = 0;
  switch (_page_kind.value()) {
    case page_kind::type::system_default:
    case page_kind::type::small:
    case page_kind::type::transparent_huge: hugetlbfs_flags = 0; break;
    case page_kind::type::huge2mb: hugetlbfs_flags = MAP_HUGETLB | MAP_HUGE_2MB; break;
    case page_kind::type::huge1gb: hugetlbfs_flags = MAP_HUGETLB | MAP_HUGE_1GB; break;
  };

  void* ptr = ::mmap(
    nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | hugetlbfs_flags, -1, 0);
  if (ptr == MAP_FAILED) {
    GQE_LOG_ERROR("mmap failed with: {}", std::strerror(errno));
    throw std::bad_alloc();
  }

  int advice_flag = 0;
  switch (_page_kind.value()) {
    case page_kind::type::system_default: advice_flag = 0; break;
    case page_kind::type::small: advice_flag = MADV_NOHUGEPAGE; break;
    case page_kind::type::transparent_huge: advice_flag = MADV_HUGEPAGE; break;
    case page_kind::type::huge2mb:
    case page_kind::type::huge1gb: advice_flag = 0; break;
  };

  if (::madvise(ptr, bytes, advice_flag) == -1) {
    GQE_LOG_ERROR("madvise failed with: {}", std::strerror(errno));
    throw std::bad_alloc();
  }

  auto aligned_bytes = round_to_next_page(bytes, _page_kind.size());

  GQE_LOG_TRACE("mbind on NUMA nodes: {}", _numa_node_set.pretty_print());

  if (::mbind(ptr,
              aligned_bytes,
              MPOL_BIND,
              _numa_node_set.bits(),
              _numa_node_set.max_count,
              MPOL_MF_STRICT) == -1) {
    GQE_LOG_ERROR("mbind failed with: {}", std::strerror(errno));
    throw std::bad_alloc();
  }

  if (_pinned) {
    // Register the memory with CUDA for pinned access
    auto status = cudaHostRegister(ptr, aligned_bytes, cudaHostRegisterDefault);
    if (cudaSuccess != status) {
      GQE_LOG_ERROR("cudaHostRegister failed with: {}", cudaGetErrorString(status));
      ::munmap(ptr, aligned_bytes);
      throw std::bad_alloc();
    }
  }

  return ptr;
}

void numa_memory_resource::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view)
{
  if (ptr == nullptr || bytes == 0) { return; }

  if (_pinned) {
    // Unregister the memory from CUDA
    auto status = cudaHostUnregister(ptr);
    if (cudaSuccess != status) {
      GQE_LOG_ERROR("cudaHostUnregister failed with: {}", cudaGetErrorString(status));
    }
  }

  // munmap sometimes fails if bytes aren't page aligned
  auto aligned_bytes = round_to_next_page(bytes, _page_kind.size());

  if (::munmap(ptr, aligned_bytes) == -1) {
    GQE_LOG_ERROR("munmap failed with: {}", std::strerror(errno));
    throw std::bad_alloc();
  }
}

std::pair<std::size_t, std::size_t> available_numa_node_memory(int numa_node)
{
  long long free_bytes  = 0;
  long long total_bytes = numa_node_size64(numa_node, &free_bytes);
  if (total_bytes == -1) {
    throw std::runtime_error("Failed to get memory size for NUMA node " +
                             std::to_string(numa_node));
  }
  return {static_cast<std::size_t>(free_bytes), static_cast<std::size_t>(total_bytes)};
}

std::pair<std::size_t, std::size_t> available_numa_node_memory(const cpu_set& numa_node_set)
{
  std::size_t total_free  = 0;
  std::size_t total_bytes = 0;
  for (int i = 0; i < cpu_set::max_count; ++i) {
    if (numa_node_set.contains(i)) {
      auto [free, total] = available_numa_node_memory(i);
      total_free += free;
      total_bytes += total;
    }
  }
  return {total_free, total_bytes};
}

}  // namespace memory_resource

}  // namespace gqe
