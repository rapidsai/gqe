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

#include <boost/interprocess/managed_shared_memory.hpp>
#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>

#include <cstddef>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief Boost shared memory resource for inter-process CPU shared memory.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
class boost_shared_memory_resource {
 public:
  boost_shared_memory_resource();
  ~boost_shared_memory_resource();

  boost_shared_memory_resource(boost_shared_memory_resource const&)            = delete;
  boost_shared_memory_resource(boost_shared_memory_resource&&)                 = delete;
  boost_shared_memory_resource& operator=(boost_shared_memory_resource const&) = delete;
  boost_shared_memory_resource& operator=(boost_shared_memory_resource&&)      = delete;

  boost::interprocess::managed_shared_memory& segment() { return _segment; }

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
   * The shared-memory segment is registered with CUDA via `cudaHostRegister` in the constructor,
   * so allocations are reachable from both host and device.
   */
  friend void get_property(boost_shared_memory_resource const&, cuda::mr::host_accessible) noexcept
  {
  }
  friend void get_property(boost_shared_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] bool operator==(boost_shared_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
  [[nodiscard]] bool operator!=(boost_shared_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  boost::interprocess::managed_shared_memory _segment;
};

}  // namespace memory_resource

}  // namespace gqe
