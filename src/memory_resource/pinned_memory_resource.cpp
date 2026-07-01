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

#include <gqe/memory_resource/pinned_memory_resource.hpp>

#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <rmm/aligned.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <concepts>
#include <new>

namespace gqe {

namespace memory_resource {

namespace {
using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
static_assert(std::constructible_from<any_device_resource, pinned_memory_resource>);
}  // namespace

void* pinned_memory_resource::allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
{
  assert(alignment > 0);

  if (0 == bytes) { return nullptr; }

  void* ptr{nullptr};
  auto status = cudaMallocHost(&ptr, bytes);
  if (cudaSuccess != status) { throw std::bad_alloc{}; }

  assert(rmm::is_pointer_aligned(ptr, alignment));

  return ptr;
}

void pinned_memory_resource::deallocate(cuda::stream_ref,
                                        void* ptr,
                                        std::size_t,
                                        std::size_t alignment) noexcept
{
  if (nullptr == ptr) { return; }
  assert(rmm::is_pointer_aligned(ptr, alignment));

  auto const status = cudaFreeHost(ptr);
  if (cudaSuccess != status) {
    GQE_LOG_ERROR(
      "cudaFreeHost failed: {} ({})", cudaGetErrorName(status), cudaGetErrorString(status));
  }
}

}  // namespace memory_resource

}  // namespace gqe
