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

#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/utility/linux.hpp>

#include <new>
#include <utility>

namespace gqe {

namespace memory_resource {

bool system_memory_resource::supports_streams() const noexcept { return false; }

bool system_memory_resource::supports_get_mem_info() const noexcept { return true; }

void* system_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view)
{
  return ::operator new(bytes, _allocation_alignment);
}

void system_memory_resource::do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view)
{
  ::operator delete(ptr, _allocation_alignment);
}

std::pair<std::size_t, std::size_t> system_memory_resource::do_get_mem_info(
  rmm::cuda_stream_view) const
{
  const auto info_map = utility::get_meminfo("/proc/meminfo");
  return std::make_pair(info_map.at("MemFree"), info_map.at("MemTotal"));
}

}  // namespace memory_resource

}  // namespace gqe
