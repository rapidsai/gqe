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

 private:
  int32_t _rank;
  int32_t _size;
  int32_t _ranks_per_device;
  MPI_Comm _mpi_comm;
  rmm::cuda_device_id _device_id;
};
}  // namespace gqe