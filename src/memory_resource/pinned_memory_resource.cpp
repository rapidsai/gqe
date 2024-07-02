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

#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/utility/error.hpp>

#include <cuda_runtime_api.h>
#include <rmm/detail/aligned.hpp>

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
