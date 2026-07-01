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

#pragma once

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>

#include <cstddef>

namespace gqe {
/**
 * @brief PGAS memory resource allows suballocations within this resource to be accessed by RMA with
 * NVSHMEM. NVSHMEM will have undefined behavior if API calls to RMA routines use suballocations in
 * different pgas_memory_resource objects.
 *
 * NVSHMEM requires initialization before creating resource of this type.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
class pgas_memory_resource {
 public:
  pgas_memory_resource(std::size_t bytes);
  pgas_memory_resource(pgas_memory_resource const&)            = delete;
  pgas_memory_resource& operator=(pgas_memory_resource const&) = delete;
  void* get_local_base_ptr() const { return _local_base_ptr; }
  std::size_t get_bytes() const { return _bytes; }
  /**
   * @brief Finalize the PGAS memory resource.
   *
   * This function will free the symmetric memory allocated by this resource. Function ensures that
   * all processes have finished RMA operations before freeing the memory.
   */
  void finalize();

  /**
   * @brief Allocates memory of size at least \p bytes as a NVSHMEM symmetric object.
   *
   * The returned pointer has 256-byte alignment. Requests above that alignment are rejected.
   *
   * The stream argument is ignored (NVSHMEM allocations are synchronous).
   */
  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocates the NVSHMEM symmetric object.
   *
   * The stream argument is ignored. The supplied pointer must be the one returned by `allocate`.
   */
  void deallocate(cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t /*bytes*/,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
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
   * @brief Enables the `cuda::mr::device_accessible` property.
   */
  friend void get_property(pgas_memory_resource const&, cuda::mr::device_accessible) noexcept {}

  [[nodiscard]] bool operator==(pgas_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
  [[nodiscard]] bool operator!=(pgas_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  void* _local_base_ptr;  // Pointer to the local base of the symmetric memory
  bool _allocated;        // Whether the memory has been allocated
  std::size_t _bytes;     // Total size of the symmetric memory
};

}  // namespace gqe
