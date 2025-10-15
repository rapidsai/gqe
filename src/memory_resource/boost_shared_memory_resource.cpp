/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

namespace gqe {

namespace memory_resource {

boost_shared_memory_resource::boost_shared_memory_resource()
{
  _segment =
    boost::interprocess::managed_shared_memory(boost::interprocess::open_only, "gqe_shared_memory");

  // Register the memory with CUDA for pinned access
  // We need to pin the entire memory, instead of just allocations,
  // because only one process actually allocates memory.
  GQE_CUDA_TRY(
    cudaHostRegister(_segment.get_address(), _segment.get_size(), cudaHostRegisterDefault));
}

void* boost_shared_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view)
{
  if (0 == bytes) { return nullptr; }

  std::size_t constexpr allocation_alignment = 256;

  void* ptr;
  ptr = _segment.allocate_aligned(bytes, allocation_alignment);

  if (ptr == nullptr) {
    GQE_LOG_ERROR("Failed to allocate boost shared memory of requested size: {}", bytes);
    throw std::bad_alloc();
  }

  return ptr;
}

void boost_shared_memory_resource::do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view)
{
  if (nullptr == ptr) { return; }

  _segment.deallocate(ptr);
}

boost_shared_memory_resource::~boost_shared_memory_resource()
{
  cudaHostUnregister(_segment.get_address());
}

}  // namespace memory_resource

}  // namespace gqe
