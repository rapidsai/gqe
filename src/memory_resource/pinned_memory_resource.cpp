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

#include <gqe/memory_resource/pinned_memory_resource.hpp>

#include <gqe/utility/error.hpp>

#include <rmm/detail/aligned.hpp>

#include <cuda_runtime_api.h>

namespace gqe {

namespace memory_resource {

void* pinned_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view)
{
  if (0 == bytes) { return nullptr; }

  void* ptr{nullptr};
  auto status = cudaMallocHost(&ptr, bytes);
  if (cudaSuccess != status) { throw std::bad_alloc{}; }

  assert(rmm::is_pointer_aligned(ptr, _allocation_alignment));

  return ptr;
}

void pinned_memory_resource::do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view)
{
  if (nullptr == ptr) { return; }

  GQE_CUDA_TRY(cudaFreeHost(ptr));
}

}  // namespace memory_resource

}  // namespace gqe
