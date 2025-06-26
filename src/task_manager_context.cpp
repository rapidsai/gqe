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
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace gqe {
task_manager_context::task_manager_context(std::unique_ptr<rmm::mr::device_memory_resource> mr,
                                           device_properties device_prop)
  : _device_properties{device_prop}
{
  _mr = std::move(mr);
  rmm::mr::set_current_device_resource(_mr.get());
}

task_manager_context::~task_manager_context()
{
  // We need to call both these APIs because of a bug in RMM.
  // https://github.com/rapidsai/rmm/issues/1953
  rmm::mr::set_current_device_resource(nullptr);
  rmm::mr::reset_current_device_resource_ref();
}

multi_process_task_manager_context::multi_process_task_manager_context(
  std::optional<std::unique_ptr<gqe::pgas_memory_resource>> upstream_mr,
  device_properties device_prop)
{
  // Initialize MPI and NVSHMEM
  _mpi_comm = MPI_COMM_WORLD;
  GQE_MPI_TRY(MPI_Init(NULL, NULL));
  GQE_MPI_TRY(MPI_Comm_rank(_mpi_comm, &_mpi_rank));
  GQE_MPI_TRY(MPI_Comm_size(_mpi_comm, &_mpi_size));

  // Get node-local rank and size
  int node_rank;
  int node_size;
  MPI_Comm node_comm;
  // Split the MPI_COMM_WORLD into node_comm based on the node rank
  GQE_MPI_TRY(MPI_Comm_split_type(_mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm));
  // Get the local rank of the process in the node_comm
  GQE_MPI_TRY(MPI_Comm_rank(node_comm, &node_rank));
  GQE_MPI_TRY(MPI_Comm_size(node_comm, &node_size));
  GQE_MPI_TRY(MPI_Comm_free(&node_comm));

  // Set CUDA device based on local rank
  int num_gpus;
  GQE_CUDA_TRY(cudaGetDeviceCount(&num_gpus));
  if (node_size > num_gpus) {
    throw std::runtime_error("Not enough GPUs available. Node process count " +
                             std::to_string(node_size) + " >= number of GPUs " +
                             std::to_string(num_gpus));
  }
  GQE_CUDA_TRY(cudaSetDevice(node_rank));

  // Initialize NVSHMEM
  assert(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED);

  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &_mpi_comm;
  auto status   = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  if (status != NVSHMEMX_SUCCESS) { throw std::runtime_error("Failed to initialize NVSHMEM"); }

  GQE_LOG_INFO("NVSHMEM team initialized with {} PEs", nvshmem_n_pes());

  auto pool_size = gqe::utility::default_device_memory_pool_size();
  void* local_base_ptr;
  {
    using upstream_mr_type = gqe::pgas_memory_resource;
    using mr_type          = rmm::mr::pool_memory_resource<upstream_mr_type>;

    // Initialize memory resource
    if (!upstream_mr.has_value()) {
      auto pgas_mr      = std::make_shared<gqe::pgas_memory_resource>(pool_size);
      _upstream_pgas_mr = pgas_mr.get();
      local_base_ptr    = pgas_mr->get_local_base_ptr();
      _mr               = std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(
        pgas_mr, pool_size, pool_size);
    } else {
      _upstream_pgas_mr = upstream_mr.value().get();
      local_base_ptr    = upstream_mr.value()->get_local_base_ptr();
      auto pool_size    = upstream_mr.value()->get_bytes();
      _mr               = std::make_unique<rmm::mr::owning_wrapper<mr_type, upstream_mr_type>>(
        std::shared_ptr<gqe::pgas_memory_resource>(std::move(upstream_mr.value())),
        pool_size,
        pool_size);
    }
  }

  _base_ptrs.resize(_mpi_size);
  GQE_MPI_TRY(MPI_Allgather(&local_base_ptr,
                            sizeof(void*),
                            MPI_CHAR,
                            _base_ptrs.data(),
                            sizeof(void*),
                            MPI_CHAR,
                            _mpi_comm));

  rmm::mr::set_current_device_resource(_mr.get());
}

void multi_process_task_manager_context::finalize()
{
  _upstream_pgas_mr->finalize();
  nvshmemx_hostlib_finalize();
  GQE_MPI_TRY(MPI_Finalize());
}

void* multi_process_task_manager_context::get_translated_ptr(void* ptr, int32_t destination_rank)
{
  return static_cast<std::byte*>(_base_ptrs[destination_rank]) +
         (reinterpret_cast<std::uintptr_t>(ptr) -
          reinterpret_cast<std::uintptr_t>(_base_ptrs[_mpi_rank]));
}
}  // namespace gqe
