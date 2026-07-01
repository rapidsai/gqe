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

#include <gqe/memory_resource/boost_shared_memory_resource.hpp>

#include <gqe/node_manager/context.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cassert>

namespace gqe {

namespace memory_resource {

boost_shared_memory_resource::boost_shared_memory_resource()
{
  _segment = boost::interprocess::managed_shared_memory(boost::interprocess::open_only,
                                                        gqe::node_manager::shared_memory_name);

  // Register the memory with CUDA for pinned access
  // We need to pin the entire memory, instead of just allocations,
  // because only one process actually allocates memory.
  GQE_CUDA_TRY(
    cudaHostRegister(_segment.get_address(), _segment.get_size(), cudaHostRegisterDefault));
}

void* boost_shared_memory_resource::allocate(cuda::stream_ref,
                                             std::size_t bytes,
                                             std::size_t alignment)
{
  if (0 == bytes) { return nullptr; }

  void* ptr;
  ptr = _segment.allocate_aligned(bytes, alignment);

  if (ptr == nullptr) {
    GQE_LOG_ERROR("Failed to allocate boost shared memory of requested size: {}", bytes);
    throw std::bad_alloc();
  }

  return ptr;
}

void boost_shared_memory_resource::deallocate(cuda::stream_ref,
                                              void* ptr,
                                              std::size_t,
                                              std::size_t alignment) noexcept
{
  if (nullptr == ptr) { return; }
  assert(rmm::is_pointer_aligned(ptr, alignment));

  _segment.deallocate(ptr);
}

boost_shared_memory_resource::~boost_shared_memory_resource()
{
  cudaHostUnregister(_segment.get_address());
}

}  // namespace memory_resource

}  // namespace gqe
