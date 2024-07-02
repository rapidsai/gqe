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
   */
  numa_memory_resource(page_kind::type page_kind = page_kind::system_default);

  /**
   * @brief Create a NUMA memory resource using the specified NUMA nodes.
   *
   * @param[in] numa_node_set The NUMA nodes the memory resource will consist of.
   * @param[in] page_kind The page kind. Defaults to the system default page kind.
   */
  numa_memory_resource(cpu_set numa_node_set,
                       page_kind::type page_kind = page_kind::system_default);

  ~numa_memory_resource() override = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override;

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view) override;

  cpu_set _numa_node_set; /**< The NUMA nodes on which memory will be allocated. */
  page_kind
    _page_kind; /**< The page type to allocate, e.g., 4 KB small pages or 2 MB huge pages. */
};

}  // namespace memory_resource

}  // namespace gqe
