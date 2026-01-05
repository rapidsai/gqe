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

#pragma once

#include <gqe/communicator.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/pgas_memory_resource.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/scheduler.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/table/table.hpp>
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <mpi.h>
#include <nvshmem.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace gqe {

/**
 * @brief Implements an wrapper around rmm::cuda_stream to allow for cooperative stream sharing
 * among multiple threads.
 */
struct shared_stream {
  std::unique_ptr<std::mutex> mtx = std::make_unique<std::mutex>();
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
   * @param params The optimization parameters. The context stores a reference; the caller must
   *               ensure the parameters outlive this context.
   * @param mr The memory resource to use for device memory allocations.
   *
   * FIXME: Single source of truth for optimization parameters. There is another instance in
   * `query_context`.
   */
  explicit task_manager_context(optimization_parameters params = gqe::optimization_parameters{},
                                std::unique_ptr<rmm::mr::device_memory_resource> mr =
                                  std::make_unique<rmm::mr::cuda_async_memory_resource>());

  /**
   * @brief Destructs a task manager context.
   * @warning Destroying this object will reset the current device resource to the default memory
   * resource provided by RMM
   */
  virtual ~task_manager_context();
  task_manager_context(const task_manager_context&)            = delete;
  task_manager_context(task_manager_context&&)                 = default;
  task_manager_context& operator=(const task_manager_context&) = delete;
  task_manager_context& operator=(task_manager_context&&)      = default;
  virtual void finalize();

  /**
   * @brief Get a memory resource reference for a memory kind.
   *
   * If the memory resource has not been created yet, it will be created lazily.
   * These memory resources are primarily intended for use by in-memory tables stored in the GQE
   * catalog.
   *
   * # Thread Safety
   *
   * This method is thread-safe.
   *
   * @param memory_kind The memory kind to get or create.
   *
   * @return A reference to the memory resource.
   */
  rmm::device_async_resource_ref get_table_memory_resource(memory_kind::type const& memory_kind);

  /**
   * @brief Get a raw pointer to the memory resource for a memory kind.
   *
   * If the memory resource has not been created yet, it will be created lazily.
   *
   * # Thread Safety
   *
   * This method is thread-safe.
   *
   * @param memory_kind The memory kind to get or create.
   *
   * @return A raw pointer to the memory resource. The pointer is valid as long as this
   *         task_manager_context exists.
   */
  rmm::mr::device_memory_resource* get_table_memory_resource_ptr(
    memory_kind::type const& memory_kind);

 private:
  /**
   * @brief Create a table memory resource.
   *
   * This is an internal method that creates a table memory resource for a given memory kind.
   *
   * # Thread Safety
   *
   * Warning: This method is _not_ thread-safe, to avoid acquiring the _table_memory_resource_latch
   * twice.
   *
   * @param memory_kind The memory kind to create a resource for.
   * @return A pointer to the created memory resource.
   */
  rmm::mr::device_memory_resource* create_table_memory_resource_unsafe(
    memory_kind::type const& memory_kind);

 protected:
  optimization_parameters _optimization_parameters;
  std::unique_ptr<rmm::mr::device_memory_resource> _mr;

  /*
   * @brief Storage for table memory resources.
   *
   * The storage contains lazily initialized memory resources. Each memory resource is keyed by a
   * `memory_kind::type`, which determines its configuration.
   *
   * # Thread Safety
   *
   * All accesses must obtain the `_memory_resource_latch` before accessing the memory resources.
   *
   * # Mutable
   *
   * Both the map and latch are mutable to allow lazy initialization in the const
   * get_memory_resource method.
   */
  std::unordered_map<memory_kind::type,
                     std::unique_ptr<rmm::mr::device_memory_resource>,
                     memory_kind::type_hash>
    _table_memory_resources;
  std::unique_ptr<std::mutex>
    _table_memory_resource_latch;  ///< Latch for thread-safe access to the memory resources.

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
    optimization_parameters params,
    std::unique_ptr<gqe::pgas_memory_resource> upstream_mr);

  static std::unique_ptr<multi_process_task_manager_context> default_init(
    MPI_Comm mpi_comm,
    gqe::SCHEDULER_TYPE type       = gqe::SCHEDULER_TYPE::ROUND_ROBIN,
    optimization_parameters params = optimization_parameters{});

  multi_process_task_manager_context(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context(multi_process_task_manager_context&&)      = default;
  multi_process_task_manager_context& operator=(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context& operator=(multi_process_task_manager_context&&) = default;

  /**
   * @brief Update the scheduler.
   * @param type The type of scheduler to update to.
   */
  void update_scheduler(gqe::SCHEDULER_TYPE type);

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
