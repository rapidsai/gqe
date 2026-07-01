/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/node_manager/context.hpp>
#include <gqe/node_manager/spawn.hpp>
#include <gqe/types.hpp>

#include <grpcpp/channel.h>

#include <arrow/record_batch.h>
#include <arrow/status.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace grpc {
class ClientContext;
}

namespace gqe::physical {
class relation;
}

namespace gqe::node_manager {

/** @brief Configuration for the node manager Flight SQL server. */
struct configuration {
  std::string listen_address = "0.0.0.0";  ///< Network address to listen on.
  int port                   = 50051;      ///< Flight SQL server port.
  int num_gpus               = 1;          ///< Number of GPUs / task managers to spawn.
  std::string task_manager_binary;         ///< Path to the gqe_task_manager executable.
  int task_manager_base_port      = 0;     ///< Base gRPC port for task managers (0 = port + 1000).
  gqe::storage_kind::type storage = gqe::storage_kind::boost_shared_memory{};  ///< Storage backend.
  std::chrono::seconds query_timeout{600};  ///< gRPC deadline for ExecutePlan.
  gqe::optimization_parameters opt_params =
    gqe::make_optimization_parameters();  ///< Optimizer and execution parameters.
};

class service;

/**
 * @brief Orchestrates query execution across task managers.
 *
 * The node manager spawns task manager subprocesses (one per GPU), bootstraps
 * NVSHMEM via gRPC, and distributes query plans to task managers for execution.
 * Flight SQL client interaction is handled by @ref service.
 */
class node_manager {
 public:
  explicit node_manager(configuration config);
  ~node_manager();

  /// Start the Flight SQL server and launch task manager subprocesses.
  [[nodiscard]] arrow::Status serve();

  /// Shutdown: cancel in-flight RPCs, terminate task managers, stop Flight SQL server.
  void shutdown();

 private:
  /// Result of executing a plan on task managers.
  struct execution_result {
    std::shared_ptr<arrow::RecordBatch> batch;         ///< Query result (null for writes).
    std::optional<gqe::table_statistics> write_stats;  ///< Write statistics (absent for queries).
  };

  void cancel_in_flight_rpcs();

  /**
   * @brief Spawn task manager subprocesses and bootstrap NVSHMEM.
   *
   * Forks the task manager processes, waits for their gRPC servers to become
   * reachable, then calls @ref gqe::node_manager::bootstrap_nvshmem to
   * initialise the NVSHMEM collective.
   *
   * @pre Caller must hold @c _task_manager_mutex.
   */
  void launch_task_managers_unsafe();

  void restart_task_managers();
  void monitor_task_managers(std::stop_token token);

  /**
   * @brief Execute a Substrait SELECT plan.
   *
   * Parses, optimises, and executes the plan on task managers.
   *
   * @param data Raw bytes of the serialised Substrait plan.
   * @param size Size of @p data in bytes.
   * @return The query result as an Arrow RecordBatch, or nullptr if the plan
   *         produced no output.
   * @throws std::logic_error if the plan is a write operation.
   */
  [[nodiscard]] std::shared_ptr<arrow::RecordBatch> execute_substrait_query(void const* data,
                                                                            std::size_t size);

  /**
   * @brief Execute a Substrait write/DDL plan.
   *
   * Parses, optimises, and executes the plan on task managers. Updates the
   * catalog statistics if write statistics are produced.
   *
   * @param data Raw bytes of the serialised Substrait plan.
   * @param size Size of @p data in bytes.
   * @return Write statistics if the plan produced them, or std::nullopt for
   *         DDL operations that don't report row counts.
   * @throws std::logic_error if the plan is not a write operation.
   */
  [[nodiscard]] std::optional<gqe::table_statistics> execute_substrait_statement(void const* data,
                                                                                 std::size_t size);

  /**
   * @brief Execute a serialized `proto::PhysicalRelation` SELECT plan.
   *
   * Deserialises the physical relation tree and dispatches it directly to
   * task managers, bypassing Substrait parsing and logical optimisation.
   *
   * @param data Raw bytes of the serialised `proto::PhysicalRelation`.
   * @param size Size of @p data in bytes.
   * @return The query result as an Arrow RecordBatch, or nullptr if the plan
   *         produced no output.
   * @throws std::runtime_error if the bytes do not parse as a
   *         `proto::PhysicalRelation`.
   * @throws std::logic_error if the plan is a write operation.
   */
  [[nodiscard]] std::shared_ptr<arrow::RecordBatch> execute_physical_plan_query(void const* data,
                                                                                std::size_t size);

  /**
   * @brief Dispatch a physical plan to all task managers and collect results.
   *
   * @param physical_plan The physical plan to execute.
   * @return Combined query results and/or write statistics from all ranks.
   */
  [[nodiscard]] execution_result execute_on_task_managers(
    std::shared_ptr<gqe::physical::relation> physical_plan);

  configuration _config;
  std::unique_ptr<context> _ctx;           // Must outlive _catalog.
  std::unique_ptr<gqe::catalog> _catalog;  // Must outlive _service.
  std::unique_ptr<service> _service;

  /**
   * @brief Protects @c _task_manager_processes and @c _task_manager_channels.
   *
   * The monitor thread may replace both vectors when restarting crashed task
   * managers, while Flight SQL handler threads read @c _task_manager_channels
   * to dispatch queries. Without this mutex, a handler thread could read a
   * partially-swapped channel vector during a restart.
   *
   * Acquired by @c serve() at startup, @c monitor_task_managers() on the
   * monitor thread, and @c execute_on_task_managers() on Flight SQL handler
   * threads (to snapshot channels).
   */
  std::mutex _task_manager_mutex;
  process_group _task_manager_processes;  ///< Guarded by @c _task_manager_mutex.
  std::vector<std::shared_ptr<grpc::Channel>>
    _task_manager_channels;  ///< Guarded by @c _task_manager_mutex.
  std::jthread _monitor_thread;

  /**
   * @brief Interruptible sleep for the monitor thread.
   *
   * The monitor thread polls task manager health every second via
   * @c _monitor_cv.wait_for() with the jthread's stop token. On shutdown,
   * @c _monitor_thread.request_stop() unblocks the wait immediately so the
   * monitor thread can exit without waiting for the polling interval to expire.
   */
  std::mutex _monitor_mutex;
  std::condition_variable_any _monitor_cv;

  /**
   * @brief Protects @c _in_flight_rpcs.
   *
   * Flight SQL handler threads register their @c grpc::ClientContext pointers
   * before dispatching RPCs and unregister them afterwards. On task manager
   * crash or shutdown, @c cancel_in_flight_rpcs() iterates the vector to
   * cancel outstanding RPCs. Without this mutex, a crash during registration
   * could corrupt the vector or miss an in-flight RPC.
   *
   * Acquired by @c execute_on_task_managers() to register/unregister RPCs,
   * and by @c cancel_in_flight_rpcs() on the monitor thread and during
   * shutdown.
   */
  std::mutex _in_flight_mutex;
  std::vector<grpc::ClientContext*> _in_flight_rpcs;  ///< Guarded by @c _in_flight_mutex.

  /**
   * @brief Serialises plan execution. @c gqe::catalog is unsynchronised; any
   * concurrent path that reads or mutates it would race on @c _table_entries.
   *
   * The optimizer and plan builder use shared catalog state (@c _table_entries)
   * that is not thread-safe, and @c _config.opt_params must not be modified
   * while a query is being planned or dispatched. This mutex ensures only one
   * query, statement, or SET/SHOW handler runs at a time.
   *
   * Acquired by @c execute_substrait_query(), @c execute_substrait_statement(),
   * @c execute_physical_plan_query(), and the set/get session option callbacks
   * in @c serve().
   */
  std::mutex _execute_mutex;
};

}  // namespace gqe::node_manager
