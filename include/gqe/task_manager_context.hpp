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
#include <gqe/storage/shared_memory.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cuda/memory_resource>
#include <cudf/table/table.hpp>
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <unordered_map>

namespace gqe {

struct peer_info;  // defined in gqe/rpc/task_migration.hpp

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
   * @param mr The owning, type-erased memory resource to use for device memory allocations.
   *
   * FIXME: Single source of truth for optimization parameters. There is another instance in
   * `query_context`.
   */
  explicit task_manager_context(
    optimization_parameters params = gqe::make_optimization_parameters(),
    cuda::mr::any_resource<cuda::mr::device_accessible> mr =
      cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_async_memory_resource{}});

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
   * @brief Get the process-local mapping of the shared-memory segment.
   *
   * The segment is opened lazily and owned by the context. Objects that borrow this reference must
   * not outlive the context.
   *
   * # Thread Safety
   *
   * This method is thread-safe.
   *
   * @return A reference to the shared-memory segment.
   */
  shared_segment& shared_memory_segment();

  /**
   * @brief Get the statistics adaptor that wraps the query memory resource.
   *
   * The adaptor shares its counters with `_mr`. The plan executor pushes a counter frame when a
   * query starts and pops it when the query ends, so the frame's peak is the query's peak above
   * the bytes allocated at its start. The pointer stays valid until the context is destroyed or the
   * query memory resource is replaced.
   *
   * @return The adaptor, or null unless `track_query_memory` was set when the context was created.
   */
  [[nodiscard]] rmm::mr::statistics_resource_adaptor* query_memory_statistics() noexcept
  {
    return _query_memory_statistics ? &*_query_memory_statistics : nullptr;
  }

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
   * @return A non-owning reference to the freshly created (and now stored) memory resource.
   */
  rmm::device_async_resource_ref create_table_memory_resource_unsafe(
    memory_kind::type const& memory_kind);

 protected:
  /**
   * @brief Install `mr` as the query memory resource and the current device resource.
   *
   * With `track_query_memory` set, `mr` is wrapped in an `rmm::mr::statistics_resource_adaptor`,
   * and a copy that shares its counters is kept in `_query_memory_statistics`.
   *
   * @param mr The owning, type-erased query memory resource.
   */
  void set_query_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  optimization_parameters _optimization_parameters;
  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  std::optional<rmm::mr::statistics_resource_adaptor>
    _query_memory_statistics;  ///< Shares counters with `_mr`; empty when peak tracking is off.

  std::unique_ptr<shared_segment> _shared_memory_segment;
  std::unique_ptr<std::mutex> _shared_memory_segment_latch =
    std::make_unique<std::mutex>();  ///< Latch for thread-safe access to shared memory segment.

  /*
   * @brief Storage for table memory resources.
   *
   * The storage contains lazily initialized type-erased memory resources. Each entry is keyed by
   * a `memory_kind::type`, which determines its configuration.
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
                     cuda::mr::any_resource<cuda::mr::device_accessible>,
                     memory_kind::type_hash>
    _table_memory_resources;
  std::unique_ptr<std::mutex>
    _table_memory_resource_latch;  ///< Latch for thread-safe access to the memory resources.

 public:
  shared_stream copy_engine_stream;

  /**
   * @brief Semaphore to limit concurrent batched memcpy API calls.
   *
   * # Problem Description
   *
   * This is a heuristical performance optimization for the batched memcpy API.
   * It is based on the empirical observation that there are two competing
   * factors:
   *
   * - The CUDA driver (appears to) schedules batch copy requests in round-robin fashion. This can
   *   lead to head-of-queue blocking for kernels, as these wait on dependencies to become ready. In
   *   some cases, a single batch copy is broken up into multiple smaller copies.
   * - Scheduling a batch copy has high overhead on the host.
   *
   * These factors must be balanced. The overhead means that multiple host
   * threads must schedule and copy concurrently to hide the overhead. However,
   * the round-robin scheduling means that fewer concurrent copies result in
   * less head-of-queue blocking.
   *
   * # Solution
   *
   * The semaphore limits the number of copy requests participating in
   * round-robin scheduling at any given time. The semaphore value `2` is the
   * smallest possible value that overlaps host overhead with a copy.
   */
  std::counting_semaphore<2> batched_memcpy_semaphore{2};

  /**
   * @brief Semaphore to limit concurrent decompress API calls.
   *
   * See `batched_memcpy_semaphore` for more details.
   */
  std::counting_semaphore<2> decompress_semaphore{2};

  /**
   * @brief Optimization parameters used to configure the context.
   *
   * Used during serialization and deserialization to compute the data path.
   *
   * # Thread Safety
   *
   * Warning: this method is _not_ thread-safe. The returned reference stays valid only until the
   * next @ref set_optimization_parameters call. See that method for the external mutual exclusion
   * the caller must provide.
   */
  [[nodiscard]] optimization_parameters const& get_optimization_parameters() const noexcept
  {
    return _optimization_parameters;
  }

  /**
   * @brief Replace the active optimization parameters.
   *
   * The node manager calls this when a session option is set, so that everything reading these
   * parameters -- notably the serialize/deserialize data-path derivation -- observes the current
   * session settings rather than the startup snapshot taken at construction. Already-created memory
   * resources are cached and are not resized by a later change.
   *
   * # Thread Safety
   *
   * Warning: this method is _not_ thread-safe, and the parameters carry no internal
   * synchronization. The caller must serialize this write against every concurrent
   * @ref get_optimization_parameters reader, including readers reached indirectly through
   * @ref gqe::catalog.
   *
   * @pre No thread is reading the parameters, directly or indirectly, for the duration of the call.
   */
  void set_optimization_parameters(optimization_parameters params) noexcept
  {
    _optimization_parameters = std::move(params);
  }
};

struct multi_process_task_manager_context : public task_manager_context {
  /**
   * @brief Create a context from already initialized parts.
   *
   * If this throws, `comm` is leaked instead of destroyed, as in @ref default_init.
   */
  explicit multi_process_task_manager_context(
    std::unique_ptr<gqe::communicator> comm,
    std::unique_ptr<gqe::scheduler> scheduler,
    std::unique_ptr<gqe::task_migration_client> migration_client,
    std::unique_ptr<gqe::task_migration_service> migration_service,
    gqe::rpc_server&& server,
    optimization_parameters params,
    std::unique_ptr<gqe::pgas_memory_resource> upstream_mr);

  /**
   * @brief Create a fully initialised multi-process task manager context.
   *
   * Initialises NVSHMEM via the UID bootstrap path (`NVSHMEMX_INIT_WITH_UNIQUEID`),
   * allocates the PGAS memory pool, creates the scheduler, and wires up the
   * task-migration RPC infrastructure.
   *
   * @param rank   The NVSHMEM rank assigned to this process (0-based).
   * @param nranks Total number of ranks in the world.
   * @param uid    128-byte unique ID shared by all ranks.
   * @param device The device this process uses. The caller decides it once per process.
   * @param peers  Pre-built peer list with PGAS base pointers and gRPC addresses. If it is empty,
   *               the context has no migration client until @ref connect_peers is called.
   * @param type   Scheduler type (default: round-robin).
   * @param params Optimization parameters for execution.
   * @return A fully initialised context, ready for query execution.
   *
   * If this fails after NVSHMEM is initialized, the NVSHMEM objects created so far are leaked
   * instead of freed. Freeing them is collective, so freeing them on one failing rank would block
   * forever. The caller is expected to exit after a failure.
   */
  static std::unique_ptr<multi_process_task_manager_context> default_init(
    int rank,
    int nranks,
    nvshmemx_uniqueid_t const& uid,
    rmm::cuda_device_id device,
    std::span<peer_info const> peers,
    gqe::SCHEDULER_TYPE type       = gqe::SCHEDULER_TYPE::ROUND_ROBIN,
    optimization_parameters params = make_optimization_parameters());

  multi_process_task_manager_context(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context(multi_process_task_manager_context&&)      = default;
  multi_process_task_manager_context& operator=(const multi_process_task_manager_context&) = delete;
  multi_process_task_manager_context& operator=(multi_process_task_manager_context&&) = default;

  /**
   * @brief Create the migration client for the given peers.
   *
   * This completes a context that `default_init` created without peers. In a task manager, the
   * peers' PGAS base pointers and migration addresses are known only after every rank has
   * initialized NVSHMEM.
   *
   * @param peers Peer list with PGAS base pointers and gRPC addresses, including this rank.
   *
   * @pre The context has no migration client yet. Enforced by `GQE_EXPECTS`.
   */
  void connect_peers(std::span<peer_info const> peers);

  /**
   * @brief Update the scheduler.
   * @param type The type of scheduler to update to.
   */
  void update_scheduler(gqe::SCHEDULER_TYPE type);

  /**
   * @brief Finalize the task manager context.
   *
   * This function will finalize the NVSHMEM communicator. Using NVSHMEM functions after this call
   * is undefined behavior. This function should be called only once.
   *
   * Need for this functions arises because destructor should be noexcept.
   */
  void finalize() override;

  std::unique_ptr<gqe::communicator> comm;
  std::unique_ptr<gqe::scheduler> scheduler;
  std::unique_ptr<gqe::task_migration_client> migration_client;
  std::unique_ptr<gqe::task_migration_service> migration_service;

  /**
   * @brief Get the PGAS base pointer for this rank's symmetric heap allocation.
   *
   * @return The local base pointer of the PGAS memory resource.
   */
  void* pgas_base_ptr() const;

  /**
   * @brief Get the task-migration gRPC server address.
   *
   * @return The address in "ip:port" format.
   */
  std::string migration_grpc_address() const;

 private:
  gqe::rpc_server _rpc_server;
  gqe::pgas_memory_resource* _upstream_pgas_mr;
};

}  // namespace gqe
