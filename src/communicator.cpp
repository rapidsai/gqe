/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/communicator.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <nvshmem.h>
#include <nvshmemx.h>

namespace gqe {

void nvshmem_communicator::init()
{
  // Set the CUDA device before NVSHMEM init. NVSHMEM allocates its symmetric
  // heap on the current device during init, so the device must be selected first.
  auto num_gpus_per_node = utility::get_device_count();
  int device             = _init_rank % num_gpus_per_node;
  GQE_LOG_INFO("Setting CUDA device to {}", device);
  GQE_CUDA_TRY(cudaSetDevice(device));

  nvshmemx_init_attr_t attr;
  nvshmemx_set_attr_uniqueid_args(_init_rank, _init_nranks, &_uid, &attr);
  auto status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  if (status != NVSHMEMX_SUCCESS) { throw std::runtime_error("Failed to initialize NVSHMEM"); }

  GQE_LOG_INFO("NVSHMEM team initialized with {} PEs", nvshmem_n_pes());

  auto npes_node        = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int max_ranks_per_gpu = (num_gpus_per_node + npes_node - 1) / num_gpus_per_node;

  _device_id        = rmm::cuda_device_id(device);
  _rank             = nvshmem_my_pe();
  _size             = nvshmem_n_pes();
  _ranks_per_device = max_ranks_per_gpu;

  // Allocate symmetric heap scratch for error propagation
  _err_local  = make_nvshmem_unique<uint8_t>();
  _err_global = make_nvshmem_unique<uint8_t>();
}

void nvshmem_communicator::finalize()
{
  _err_local.reset();
  _err_global.reset();
  nvshmemx_hostlib_finalize();
}

void nvshmem_communicator::barrier_world() { nvshmem_barrier_all(); }

void nvshmem_communicator::propagate_error(std::exception_ptr local_exception) const
{
  uint8_t val = (local_exception != nullptr) ? 1 : 0;
  GQE_CUDA_TRY(cudaMemcpy(_err_local.get(), &val, 1, cudaMemcpyHostToDevice));
  nvshmem_uint8_or_reduce(NVSHMEM_TEAM_WORLD, _err_global.get(), _err_local.get(), 1);
  uint8_t result;
  GQE_CUDA_TRY(cudaMemcpy(&result, _err_global.get(), 1, cudaMemcpyDeviceToHost));

  if (result != 0) {
    if (local_exception != nullptr) { std::rethrow_exception(local_exception); }
    throw std::runtime_error("Aborting stage: failure detected on another rank");
  }
}

}  // namespace gqe
