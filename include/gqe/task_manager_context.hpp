/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cudf/table/table.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/scheduler.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <memory>
#include <mpi.h>
#include <nvshmem.h>
#include <optional>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <unordered_set>

#include <gqe/communicator.hpp>
#include <grpc/grpc.h>
#include <grpcpp/server.h>

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
   * @warning Constructing an object of this type will set the current device resource to @param mr
   */
  explicit task_manager_context(std::unique_ptr<rmm::mr::device_memory_resource> mr =
                                  std::make_unique<rmm::mr::cuda_async_memory_resource>());

  /**
   * @brief Destructs a task manager context.
   * @warning Destroying this object will reset the current device resource to the default memory
   * resource provided by RMM
   */
  ~task_manager_context();
  task_manager_context(const task_manager_context&)            = delete;
  task_manager_context(task_manager_context&&)                 = default;
  task_manager_context& operator=(const task_manager_context&) = delete;
  task_manager_context& operator=(task_manager_context&&)      = default;
  virtual void finalize();

 protected:
  std::unique_ptr<rmm::mr::device_memory_resource> _mr;

 public:
  shared_stream copy_engine_stream;
};

struct multi_process_task_manager_context : public task_manager_context {
  explicit multi_process_task_manager_context(
    std::unique_ptr<gqe::communicator> comm,
    std::unique_ptr<gqe::scheduler> scheduler,
    std::unique_ptr<gqe::task_migration_client> migration_client,
    std::unique_ptr<gqe::task_migration_service> migration_service,
    gqe::rpc_server&& server,
    std::unique_ptr<gqe::pgas_memory_resource> upstream_mr);

  static std::unique_ptr<multi_process_task_manager_context> default_init(MPI_Comm mpi_comm);

  multi_process_task_manager_context(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context(multi_process_task_manager_context&&)      = default;
  multi_process_task_manager_context& operator=(const multi_process_task_manager_context&) = delete;

  /**
   * @brief Finalize the task manager context.
   *
   * This function will finalize the MPI environment and NVSHMEM communicator. Using NVSHMEM
   * functions after this call is undefined behavior. This function should be called only once.
   *
   * Need for this functions arises because destructor should be noexcept.
   */
  void finalize() override;

  std::unique_ptr<gqe::communicator> comm;
  std::unique_ptr<gqe::scheduler> scheduler;
  std::unique_ptr<gqe::task_migration_client> migration_client;
  std::unique_ptr<gqe::task_migration_service> migration_service;

 private:
  gqe::rpc_server _rpc_server;
  gqe::pgas_memory_resource* _upstream_pgas_mr;
};

}  // namespace gqe
