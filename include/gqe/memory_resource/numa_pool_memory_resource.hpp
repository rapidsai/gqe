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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace gqe {

namespace memory_resource {

class numa_pool_handle_impl;

/**
 * @brief Opaque handle wrapping a CUDA memory pool.
 *
 */
class numa_pool_handle {
 public:
  numa_pool_handle();
  ~numa_pool_handle();

  numa_pool_handle(numa_pool_handle&&) noexcept;
  numa_pool_handle& operator=(numa_pool_handle&&) noexcept;

  numa_pool_handle(numa_pool_handle const&)            = delete;
  numa_pool_handle& operator=(numa_pool_handle const&) = delete;

 private:
  std::unique_ptr<numa_pool_handle_impl> _impl;

  friend class numa_pool_memory_resource;
};

/**
 * @brief NUMA pool memory resource backed by a cudaMemPool.
 *
 * The pool is configured to allow hardware decompression and host NUMA allocations
 *
 * @note The CUDA device must be set (via cudaSetDevice) before constructing
 * this resource. Pool capabilities are validated against the current device.
 */
class numa_pool_memory_resource : public rmm::mr::device_memory_resource {
 public:
  /**
   * @brief Construct a NUMA pool memory resource.
   *
   * @note The current CUDA device is used for pool capability validation.
   * Ensure the desired device is set via cudaSetDevice before calling.
   *
   * @param[in] numa_node The host NUMA node to use for the pool location.
   * @param[in] initial_size Initial pool size in bytes.
   * @param[in] max_size Maximum pool size in bytes. If nullopt, defaults to 0
   *            (system-dependent value).
   */
  explicit numa_pool_memory_resource(int numa_node                       = 0,
                                     std::size_t initial_size            = 0,
                                     std::optional<std::size_t> max_size = std::nullopt);
  ~numa_pool_memory_resource() override;

  numa_pool_memory_resource(numa_pool_memory_resource const&)                = delete;
  numa_pool_memory_resource& operator=(numa_pool_memory_resource const&)     = delete;
  numa_pool_memory_resource(numa_pool_memory_resource&&) noexcept            = default;
  numa_pool_memory_resource& operator=(numa_pool_memory_resource&&) noexcept = default;

  /**
   * @brief Return the opaque handle to the underlying CUDA memory pool.
   */
  [[nodiscard]] numa_pool_handle const& pool_handle() const noexcept;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override;

  numa_pool_handle _pool_handle;
};

}  // namespace memory_resource

}  // namespace gqe
