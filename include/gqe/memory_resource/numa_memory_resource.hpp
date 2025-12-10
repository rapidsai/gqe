/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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
 */
class numa_memory_resource : public rmm::mr::device_memory_resource {
 public:
  /**
   * @brief Create a NUMA memory resource on the default NUMA node.
   *
   * @param[in] page_kind The page kind. Defaults to the system default page kind.
   * @param[in] pinned Whether to register the memory with CUDA for pinned access. Defaults to
   * false.
   */
  numa_memory_resource(page_kind::type page_kind = page_kind::system_default, bool pinned = false);

  /**
   * @brief Create a NUMA memory resource using the specified NUMA nodes.
   *
   * @param[in] numa_node_set The NUMA nodes the memory resource will consist of.
   * @param[in] page_kind The page kind. Defaults to the system default page kind.
   * @param[in] pinned Whether to register the memory with CUDA for pinned access. Defaults to
   * false.
   */
  numa_memory_resource(cpu_set numa_node_set,
                       page_kind::type page_kind = page_kind::system_default,
                       bool pinned               = false);

  ~numa_memory_resource() override = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view) override;

  cpu_set _numa_node_set; /**< The NUMA nodes on which memory will be allocated. */
  page_kind
    _page_kind; /**< The page type to allocate, e.g., 4 KB small pages or 2 MB huge pages. */
  bool _pinned; /**< Whether the memory is registered with CUDA for pinned access. */
};

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
