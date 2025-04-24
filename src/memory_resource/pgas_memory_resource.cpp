/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/utility/logger.hpp>
#include <nvshmem.h>

namespace gqe {

pgas_memory_resource::pgas_memory_resource(std::size_t bytes) : _bytes{bytes}
{
  // _local_base_ptr must be 256 bytes aligned, which is required by RMM. See
  // https://docs.rapids.ai/api/rmm/stable/librmm_docs/memory_resources/#_CPPv4N3rmm2mr22device_memory_resourceE
  std::size_t constexpr allocation_alignment = 256;
  _local_base_ptr                            = nvshmem_align(allocation_alignment, _bytes);
  if (!_local_base_ptr) { throw std::runtime_error("pgas_memory_resource allocation failed."); }
}

void* pgas_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view)
{
  if (_allocated) {
    GQE_LOG_ERROR(
      "pgas_memory_resource is intended to be used as Upstream resource for a pool allocator "
      "with a fixed initial allocation");
    throw std::runtime_error("pgas_memory_resource does not support allocation more than once.");
  }
  if (bytes > _bytes) {
    throw std::runtime_error(
      "Requested allocation size is greater than the underlying PGAS memory.");
  }
  _allocated = true;
  return _local_base_ptr;
}

void pgas_memory_resource::do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view)
{
  if (ptr != _local_base_ptr) {
    throw std::runtime_error(
      "Invalid pointer deallocation. Supplied pointer should be the same as the one returned by "
      "do_allocate.");
  }
  _allocated = false;
}

void pgas_memory_resource::finalize()
{
  nvshmem_barrier_all();  // Ensure all processes have finished RMA operations
  nvshmem_free(_local_base_ptr);
}

}  // namespace gqe