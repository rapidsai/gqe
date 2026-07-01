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

#include <gqe/types.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>

#include <cstddef>
#include <utility>

namespace gqe {

namespace memory_resource {

/**
 * @brief NUMA memory resource for pageable memory.
 *
 * A NUMA (non-uniform memory access) memory architecture is commonly associated with multi-socket
 * systems. However, systems with GPU coherency integrate GPU device memory into the NUMA address
 * space. Examples include P9+V100 and G+H. Thus, on these systems, GPU memory can be allocated with
 * the NUMA memory resource.
 *
 * Furthermore, chiplet CPU architectures such as the AMD EPYC 9004 series divide a single CPU
 * socket into multiple NUMA nodes. Thus, the NUMA memory resource is relevant also for
 * single-socket systems.
 *
 * == Configuration ==
 *
 * The memory resource can be configured with a NUMA node set and a page kind. The node set
 * determines which NUMA nodes the memory resource consists of. Node binding is strict and thus
 * allocations will fail if memory cannot be allocated on the specified NUMA nodes. The page kind
 * determines the page kind that will be allocated. Huge pages require special setup, as documented
 * in @ref gqe::page_kind::type.
 *
 * Models the CCCL `cuda::mr::async_resource` concept.
 */
template <bool DeviceAccessible = false>
class basic_numa_memory_resource {
 public:
  /**
   * @brief Create a NUMA memory resource on the default NUMA node.
   *
   * @param[in] page_kind The page kind. Defaults to the system default page kind.
   */
  basic_numa_memory_resource(page_kind::type page_kind = page_kind::system_default);

  /**
   * @brief Create a NUMA memory resource using the specified NUMA nodes.
   *
   * @param[in] numa_node_set The NUMA nodes the memory resource will consist of.
   * @param[in] page_kind The page kind. Defaults to the system default page kind.
   */
  basic_numa_memory_resource(cpu_set numa_node_set,
                             page_kind::type page_kind = page_kind::system_default);

  ~basic_numa_memory_resource()                                            = default;
  basic_numa_memory_resource(basic_numa_memory_resource const&)            = default;
  basic_numa_memory_resource(basic_numa_memory_resource&&)                 = default;
  basic_numa_memory_resource& operator=(basic_numa_memory_resource const&) = default;
  basic_numa_memory_resource& operator=(basic_numa_memory_resource&&)      = default;

  void* allocate(cuda::stream_ref /*stream*/,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);

  void deallocate(cuda::stream_ref /*stream*/,
                  void* ptr,
                  std::size_t bytes,
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
   * @brief Enables the `cuda::mr::host_accessible` property and conditionally advertises
   * `cuda::mr::device_accessible`.
   *
   * `numa_memory_resource` memory is always host-accessible. Device accessibility is only valid
   * for the `DeviceAccessible=true` specialization, matching CUDA registration behavior in the
   * implementation.
   * `cudaHostRegister`/`cudaHostUnregister` behavior in the implementation.
   */
  friend void get_property(basic_numa_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(basic_numa_memory_resource const&, cuda::mr::device_accessible) noexcept
    requires(DeviceAccessible)
  {
  }

  [[nodiscard]] bool operator==(basic_numa_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
  [[nodiscard]] bool operator!=(basic_numa_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  cpu_set _numa_node_set; /**< The NUMA nodes on which memory will be allocated. */
  page_kind
    _page_kind; /**< The page type to allocate, e.g., 4 KB small pages or 2 MB huge pages. */
};

using numa_memory_resource            = basic_numa_memory_resource<false>;
using numa_device_accessible_resource = basic_numa_memory_resource<true>;

/**
 * @brief Returns the available and total memory in bytes for a NUMA node.
 *
 * @param[in] numa_node The NUMA node ID.
 * @return The available and total memory in bytes for the NUMA node as a std::pair.
 * @throws std::runtime_error if the NUMA node information cannot be read.
 */
std::pair<std::size_t, std::size_t> available_numa_node_memory(int numa_node);

/**
 * @brief Returns the aggregate available and total memory in bytes for all NUMA nodes in a cpu_set.
 *
 * @param[in] numa_node_set The set of NUMA nodes.
 * @return The aggregate available and total memory in bytes as a std::pair.
 * @throws std::runtime_error if any NUMA node information cannot be read.
 */
std::pair<std::size_t, std::size_t> available_numa_node_memory(const cpu_set& numa_node_set);

}  // namespace memory_resource

}  // namespace gqe
