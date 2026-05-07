/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/memory_resource/numa_pool_memory_resource.hpp>
#include <gqe/types.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace gqe {

namespace memory_resource {

class imported_numa_pool_resource;

/**
 * @brief Owner-side NUMA pool memory resource based on cudaMemPool_t.
 *
 * Memory allocation behavior:
 *
 *   [unexported] --> export_pool() --> [exported]
 *
 *   unexported:
 *     allocate(size): allowed
 *     deallocate(ptr, size): immediate
 *
 *   exported:
 *     allocate(size): allowed
 *     deallocate(ptr, size): deferred until finalize()
 *
 * Allocations are owned by the exporting process, when using cudaMemPool it is the responsibility
 * of the application to free on importing process before free on the owner process. Subsequent
 * allocations after free on the owner process can use the freed memory, bad ordering can lead to
 * RAW hazard.
 *
 * References (CUDA Runtime API, Stream Ordered Memory Allocator):
 * - cudaMemPoolImportFromShareableHandle:
 *   imported pools cannot create new allocations.
 *   https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 * - cudaMemPoolImportPointer:
 *   imported memory must be freed in importers before freeing in exporter.
 *   https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
 */
class shared_numa_pool_resource : public rmm::mr::device_memory_resource {
 public:
  enum class export_state { unexported, exported };

  explicit shared_numa_pool_resource(int numa_node                       = 0,
                                     std::size_t initial_size            = 0,
                                     std::optional<std::size_t> max_size = std::nullopt);

  ~shared_numa_pool_resource() noexcept override = default;

  shared_numa_pool_resource(shared_numa_pool_resource const&)            = delete;
  shared_numa_pool_resource& operator=(shared_numa_pool_resource const&) = delete;
  shared_numa_pool_resource(shared_numa_pool_resource&&)                 = delete;
  shared_numa_pool_resource& operator=(shared_numa_pool_resource&&)      = delete;

  [[nodiscard]] export_state state() const;

  [[nodiscard]] remote_numa_pool_handle export_pool();

  [[nodiscard]] remote_pool_pointer export_pointer(void* ptr) const;

  /**
   * @brief Finalize the owner-side shared NUMA pool resource.
   *
   * Flushes deferred owner deallocations scheduled after export_pool().
   * finalize() performs synchronization before and after enqueuing deferred
   * frees on cudf default stream.
   *
   * @note Making allocations on this memory resource after finalize() is undefined
   * behavior.
   */
  void finalize();

  [[nodiscard]] static imported_numa_pool_resource import_pool(
    remote_numa_pool_handle const& remote_handle);

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override;

  mutable std::mutex _state_latch;
  export_state _state{export_state::unexported};
  std::unique_ptr<numa_pool_memory_resource> _owner_pool;
  std::unordered_map<void*, std::size_t> _deferred_deallocations;
};

/**
 * @brief Importer-side NUMA pool pointer-import manager.
 *
 * This class imports a remote pool and supports importing pointers from that pool.
 */
class imported_numa_pool_resource {
 public:
  explicit imported_numa_pool_resource(remote_numa_pool_handle const& remote_handle);

  ~imported_numa_pool_resource() = default;

  imported_numa_pool_resource(imported_numa_pool_resource const&)            = delete;
  imported_numa_pool_resource& operator=(imported_numa_pool_resource const&) = delete;
  imported_numa_pool_resource(imported_numa_pool_resource&&)                 = delete;
  imported_numa_pool_resource& operator=(imported_numa_pool_resource&&)      = delete;

  /**
   * @brief Import a pointer and return an owning handle.
   *
   * Destroying the handle releases the imported pointer via cudaFreeAsync on
   * `stream`.
   */
  [[nodiscard]] owning_pointer import_pointer(
    remote_pool_pointer const& remote_pointer,
    rmm::cuda_stream_view stream = cudf::get_default_stream());

 private:
  imported_numa_pool_handle _imported_pool;
};

}  // namespace memory_resource

}  // namespace gqe
