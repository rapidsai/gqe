/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/device_properties.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <memory>
#include <mpi.h>
#include <optional>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace gqe {

/**
 * @brief Implements an wrapper around rmm::cuda_stream to allow for cooperative stream sharing
 * among multiple threads.
 */
struct shared_stream {
  std::mutex mtx;
  rmm::cuda_stream stream;
};

/**
 * @brief Task manager context for query execution
 *
 * The task manager context centralizes all important resources and parameters that
 * are relevant for execution across queries on a node.
 *
 */
struct task_manager_context {
  /**
   * @brief Constructs a task manager context.
   * @note If no memory resource is supplied, then the manager context will use a
   * rmm::pool_memory_resource reserving ~90% of current free memory on device.
   */
  explicit task_manager_context(
    device_properties device_prop                                      = device_properties(),
    std::optional<std::unique_ptr<rmm::mr::device_memory_resource>> mr = std::nullopt);
  task_manager_context(const task_manager_context&) = delete;
  task_manager_context(task_manager_context&&)      = default;
  task_manager_context& operator=(const task_manager_context&) = delete;
  task_manager_context& operator=(task_manager_context&&) = default;
  device_properties const& get_device_properties() const { return _device_properties; };

 protected:
  device_properties _device_properties;

  // we need to keep _upstream_mr alive for the memory pool. rmm does not provide a way to have an
  // owned upstream resource. Shared semantics for memory resource will be supported in the future.
  // Ref https://github.com/rapidsai/rmm/issues/1878
  std::unique_ptr<rmm::mr::device_memory_resource> _upstream_mr;
  std::unique_ptr<rmm::mr::device_memory_resource> _mr;

 public:
  shared_stream copy_engine_stream;
};

struct multi_process_task_manager_context : public task_manager_context {
  explicit multi_process_task_manager_context(
    device_properties device_prop                                         = device_properties(),
    std::optional<std::unique_ptr<gqe::pgas_memory_resource>> upstream_mr = std::nullopt);
  multi_process_task_manager_context(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context(multi_process_task_manager_context&&)      = default;
  multi_process_task_manager_context& operator=(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context& operator=(multi_process_task_manager_context&&) = default;

  /**
   * @brief Translate a pointer on the current rank to that on destination_rank
   *
   * @param[in] ptr Pointer on current rank
   * @param[in] destination_rank Destination rank for ptr translation
   */
  void* get_translated_ptr(void* ptr, int32_t destination_rank);

  /**
   * @brief Get the MPI rank of the current process
   *
   * @return int32_t MPI rank
   */
  int32_t get_mpi_rank() const { return _mpi_rank; }

  /**
   * @brief Get the MPI communicator size
   *
   * @return int32_t MPI size
   */
  int32_t get_mpi_size() const { return _mpi_size; }

  /**
   * @brief Finalize the task manager context.
   *
   * This function will finalize the MPI environment and NVSHMEM communicator. Using NVSHMEM
   * functions after this call is undefined behavior. This function should be called only once.
   *
   * Need for this functions arises because destructor should be noexcept.
   */
  void finalize();

 private:
  int32_t _mpi_rank;
  int32_t _mpi_size;
  MPI_Comm _mpi_comm;
  std::vector<void*> _base_ptrs;
};

}  // namespace gqe
