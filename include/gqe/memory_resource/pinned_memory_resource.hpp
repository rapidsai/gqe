/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>

#include <cstddef>

namespace gqe {

namespace memory_resource {

/**
 * @brief Pinned memory resource for CUDA pinned host memory.
 *
 * Calls `cudaMallocHost` and `cudaFreeHost` to allocate and deallocate memory.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
class pinned_memory_resource {
 public:
  pinned_memory_resource()                                         = default;
  ~pinned_memory_resource()                                        = default;
  pinned_memory_resource(pinned_memory_resource const&)            = default;
  pinned_memory_resource(pinned_memory_resource&&)                 = default;
  pinned_memory_resource& operator=(pinned_memory_resource const&) = default;
  pinned_memory_resource& operator=(pinned_memory_resource&&)      = default;

  /**
   * @brief Allocate pinned host memory.
   *
   * The stream argument is ignored; `cudaMallocHost` is synchronous.
   *
   * Requests requiring alignment stricter than `rmm::CUDA_ALLOCATION_ALIGNMENT` are rejected.
   */
  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate(cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t /*bytes*/,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudf::get_default_stream().value()}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudf::get_default_stream().value()}, ptr, bytes, alignment);
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.
   *
   * Pinned host memory is reachable from both host and device.
   */
  friend void get_property(pinned_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(pinned_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  [[nodiscard]] bool operator==(pinned_memory_resource const&) const noexcept { return true; }
  [[nodiscard]] bool operator!=(pinned_memory_resource const&) const noexcept { return false; }
};

}  // namespace memory_resource

}  // namespace gqe
