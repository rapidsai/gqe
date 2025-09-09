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

#include <gqe/communicator.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace gqe {

void nvshmem_communicator::init()
{
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &_mpi_comm;
  auto status   = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  if (status != NVSHMEMX_SUCCESS) { throw std::runtime_error("Failed to initialize NVSHMEM"); }

  GQE_LOG_INFO("NVSHMEM team initialized with {} PEs", nvshmem_n_pes());

  // Get node-local rank and size
  auto mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  auto npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

  // Set CUDA device based on local rank
  int num_gpus_per_node;
  GQE_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  int max_ranks_per_gpu = (num_gpus_per_node + npes_node - 1) / num_gpus_per_node;
  GQE_LOG_INFO("Setting CUDA device to {}", mype_node / max_ranks_per_gpu);
  _device_id = rmm::cuda_device_id(mype_node / max_ranks_per_gpu);
  GQE_CUDA_TRY(cudaSetDevice(_device_id.value()));

  _rank             = nvshmem_my_pe();
  _size             = nvshmem_n_pes();
  _ranks_per_device = max_ranks_per_gpu;
}

void nvshmem_communicator::finalize() { nvshmemx_hostlib_finalize(); }

void nvshmem_communicator::barrier_world() { nvshmem_barrier_all(); }

}  // namespace gqe