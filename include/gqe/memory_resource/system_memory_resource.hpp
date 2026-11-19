/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>

#include <cstddef>

namespace gqe {

namespace memory_resource {

/**
 * @brief System memory resource for pageable memory.
 *
 * Calls `new` and `delete` to allocate and deallocate memory. On systems with
 * Address Translation Services / Heterogeneous Memory Management the resulting
 * pointers are also accessible from the device, which is why this resource
 * advertises both `host_accessible` and `device_accessible`.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
class system_memory_resource {
 public:
  system_memory_resource()                                         = default;
  ~system_memory_resource()                                        = default;
  system_memory_resource(system_memory_resource const&)            = default;
  system_memory_resource(system_memory_resource&&)                 = default;
  system_memory_resource& operator=(system_memory_resource const&) = default;
  system_memory_resource& operator=(system_memory_resource&&)      = default;

  /**
   * @brief Allocate `bytes` of pageable host memory.
   *
   * The stream argument is ignored (allocation is synchronous).
   */
  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return ::operator new(bytes, std::align_val_t{alignment});
  }

  /**
   * @brief Deallocate memory previously returned from `allocate`.
   *
   * The stream argument is ignored (deallocation is synchronous).
   */
  void deallocate(cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t /*bytes*/,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    ::operator delete(ptr, std::align_val_t{alignment});
  }

  /**
   * @brief Synchronous allocation overload required by `cuda::mr::synchronous_resource`.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return ::operator new(bytes, std::align_val_t{alignment});
  }

  /**
   * @brief Synchronous deallocation overload required by `cuda::mr::synchronous_resource`.
   */
  void deallocate_sync(void* ptr,
                       std::size_t /*bytes*/,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    ::operator delete(ptr, std::align_val_t{alignment});
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property.
   */
  friend void get_property(system_memory_resource const&, cuda::mr::host_accessible) noexcept {}

  /**
   * @brief Enables the `cuda::mr::device_accessible` property.
   *
   * Pageable host memory is reachable from the device on ATS/HMM-capable
   * systems, which is the only configuration where this resource is intended
   * to be installed as the current device resource.
   */
  friend void get_property(system_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  [[nodiscard]] bool operator==(system_memory_resource const&) const noexcept { return true; }
  [[nodiscard]] bool operator!=(system_memory_resource const&) const noexcept { return false; }
};

}  // namespace memory_resource

}  // namespace gqe
