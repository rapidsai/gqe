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

#include <cstdint>
#include <mpi.h>
#include <rmm/cuda_device.hpp>

namespace gqe {
/**
 * @brief Abstract base class for communicators.
 */
struct communicator {
 public:
  virtual ~communicator() = default;
  virtual void init()     = 0;
  /**
   * @brief Finalize the communicator. Using the communicator after this call is undefined behavior.
   */
  virtual void finalize() = 0;
  /**
   * @brief Block until all ranks in the world have completed previously issues communication
   * operations.
   */
  virtual void barrier_world() = 0;
  /**
   * @brief Get the rank of the current process.
   */
  virtual int32_t rank() const = 0;
  /**
   * @brief Get the total number of ranks in the world.
   */
  virtual int32_t world_size() const = 0;
  /**
   * @brief Get the number of ranks per device.
   */
  virtual int32_t num_ranks_per_device() const = 0;

  /**
   * @brief Get the device id of the current process.
   */
  virtual rmm::cuda_device_id device_id() const = 0;

  /**
   * @brief Error propagation across ranks.
   */
  virtual void propagate_error(std::exception_ptr local_exception) const = 0;
};

struct nvshmem_communicator : public communicator {
  nvshmem_communicator(MPI_Comm mpi_comm) : _mpi_comm(mpi_comm) {}
  void init() override;
  void finalize() override;
  void barrier_world() override;
  int32_t rank() const override { return _rank; }
  int32_t world_size() const override { return _size; }
  int32_t num_ranks_per_device() const override { return _ranks_per_device; }
  rmm::cuda_device_id device_id() const override { return _device_id; }
  MPI_Comm mpi_comm() const { return _mpi_comm; }
  void propagate_error(std::exception_ptr local_exception) const override;

 private:
  int32_t _rank;
  int32_t _size;
  int32_t _ranks_per_device;
  MPI_Comm _mpi_comm;
  rmm::cuda_device_id _device_id;
};
}  // namespace gqe
