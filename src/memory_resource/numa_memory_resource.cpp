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

#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/linux.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <new>  // std::bad_alloc
#include <sstream>
#include <string>

#include <linux/mman.h>
#include <numaif.h>
#include <sys/mman.h>

namespace {

std::size_t round_to_next_page(std::size_t size, std::size_t page_size) noexcept
{
  std::size_t align_mask = ~(page_size - std::size_t{1});
  return (size + page_size - std::size_t{1}) & align_mask;
}

}  // namespace

namespace gqe {

namespace memory_resource {

numa_memory_resource::numa_memory_resource(page_kind::type page_kind)
  : rmm::mr::device_memory_resource(), _numa_node_set(), _page_kind(page_kind)
{
  _numa_node_set.add(0);
}

numa_memory_resource::numa_memory_resource(cpu_set numa_node_set, page_kind::type page_kind)
  : rmm::mr::device_memory_resource(), _numa_node_set(numa_node_set), _page_kind(page_kind)
{
}

bool numa_memory_resource::supports_streams() const noexcept { return false; }

bool numa_memory_resource::supports_get_mem_info() const noexcept { return true; }

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
    nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | hugetlbfs_flags, 0, 0);
  if (ptr == MAP_FAILED) {
    GQE_LOG_ERROR("mmap failed with: ", std::strerror(errno));
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
    GQE_LOG_ERROR("madvise failed with: ", std::strerror(errno));
    throw std::bad_alloc();
  }

  auto aligned_bytes = round_to_next_page(bytes, _page_kind.size());

  if (::mbind(ptr,
              aligned_bytes,
              MPOL_BIND,
              _numa_node_set.bits(),
              _numa_node_set.max_count,
              MPOL_MF_STRICT) == -1) {
    GQE_LOG_ERROR("mbind failed with: ", std::strerror(errno));
    throw std::bad_alloc();
  }

  return ptr;
}

void numa_memory_resource::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view)
{
  if (ptr == nullptr || bytes == 0) { return; }

  // munmap sometimes fails if bytes aren't page aligned
  auto aligned_bytes = round_to_next_page(bytes, _page_kind.size());

  if (::munmap(ptr, aligned_bytes) == -1) {
    GQE_LOG_ERROR("munmap failed with: ", std::strerror(errno));
    throw std::bad_alloc();
  }
}

std::pair<std::size_t, std::size_t> numa_memory_resource::do_get_mem_info(
  rmm::cuda_stream_view) const
{
  std::size_t free = 0, total = 0;

  for (int node = 0; node < _numa_node_set.max_count; ++node) {
    if (_numa_node_set.contains(node)) {
      std::stringstream path;
      path << "/sys/devices/system/node/node" << node << "/meminfo";

      const auto info_map = utility::get_meminfo(path.str());

      std::stringstream free_name, total_name;
      free_name << "Node " << node << " MemFree";
      total_name << "Node " << node << " MemTotal";

      free += info_map.at(free_name.str());
      total += info_map.at(total_name.str());
    }
  }

  return std::make_pair(free, total);
}

}  // namespace memory_resource

}  // namespace gqe
