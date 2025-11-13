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

#include <gqe/communicator.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/mpi_helpers.hpp>
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

void nvshmem_communicator::propagate_error(std::exception_ptr local_exception) const
{
  bool local_error = (local_exception != nullptr);
  bool global_error;
  GQE_MPI_TRY(MPI_Allreduce(&local_error, &global_error, 1, MPI_C_BOOL, MPI_LOR, mpi_comm()));

  if (global_error) {
    if (local_error) { std::rethrow_exception(local_exception); }
    throw std::runtime_error("Aborting stage: failure detected on another rank");
  }
  return;
}

}  // namespace gqe
