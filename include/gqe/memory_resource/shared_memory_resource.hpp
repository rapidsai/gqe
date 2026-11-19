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

#pragma once

#include <gqe/storage/shared_memory.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>

#include <cstddef>
namespace gqe {

namespace memory_resource {

/**
 * @brief Memory resource over an inter-process CPU shared memory segment.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
class shared_memory_resource {
 public:
  explicit shared_memory_resource(shared_segment* non_owning_segment);
  ~shared_memory_resource() noexcept = default;

  shared_memory_resource(shared_memory_resource const&)            = delete;
  shared_memory_resource(shared_memory_resource&&)                 = delete;
  shared_memory_resource& operator=(shared_memory_resource const&) = delete;
  shared_memory_resource& operator=(shared_memory_resource&&)      = delete;

  shared_segment& segment() { return *_non_owning_segment; }

  /**
   * @brief Allocate from the shared-memory segment.
   *
   * @throw std::bad_alloc If the segment has no block of the requested size.
   * @throw shared_memory_unrecoverable_error If an arena's allocator lock is unrecoverable. The
   * process-wide poisoned flag is set as well.
   */
  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Return a block to the shared-memory segment.
   *
   * If the arena's allocator lock is unrecoverable, the block is leaked and the process-wide
   * poisoned flag is set (see @ref is_shared_memory_poisoned). Throwing would terminate the
   * process, because deallocation is `noexcept`.
   */
  void deallocate(cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t /*bytes*/,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudf::get_default_stream().get()}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudf::get_default_stream().get()}, ptr, bytes, alignment);
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` and `cuda::mr::device_accessible` properties.
   *
   * GPU callers must supply a segment whose mapping is registered with CUDA. The segment owns
   * registration for the mapping's lifetime. CPU-only callers can borrow an unregistered segment.
   */
  friend void get_property(shared_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(shared_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  [[nodiscard]] bool operator==(shared_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
  [[nodiscard]] bool operator!=(shared_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  shared_segment* _non_owning_segment{};
};

}  // namespace memory_resource

}  // namespace gqe
