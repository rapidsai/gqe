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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/physical/plan.hpp>
#include <gqe/storage/descriptor.hpp>
#include <gqe/utility/uuid.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gqe {
struct task_manager_context;
struct peer_info;
}  // namespace gqe

namespace gqe::task_manager {

class owner_thread;

/**
 * @brief Lifecycle state of a task manager.
 *
 * The state only moves forward: `starting` → `nvshmem_ready` → `ready`. Any state can move to
 * `stopping`.
 */
enum class task_manager_state : std::uint8_t {
  starting,       ///< Waiting for `GenerateNvshmemUid` or `InitNvshmem`.
  nvshmem_ready,  ///< NVSHMEM and the execution context are initialized, waiting for `SetPeerInfo`.
  ready,          ///< Serving `ExecutePlan` and `RefreshStatsTables`.
  stopping,       ///< Shutting down, after a failed bootstrap or at server shutdown.
};

/**
 * @brief gRPC service implementation for a task manager process.
 *
 * Receives physical plans from the node manager, deserializes them, and
 * delegates execution to a callback. Also handles the NVSHMEM UID bootstrap
 * RPCs (`GenerateNvshmemUid`, `InitNvshmem`, `SetPeerInfo`).
 *
 * # Design
 *
 * Handlers run on gRPC's thread pool and do no GPU work. A handler checks the state, reserves the
 * task manager, and queues a job on the @ref owner_thread. It then waits for the job to finish.
 * Every callback runs inside such a job, so all initialization and every operation run on the owner
 * thread. `HealthCheck` is the exception and answers from the state alone.
 *
 * ## Operations
 *
 * `ExecutePlan` and `RefreshStatsTables` build their responses through
 * @ref gqe::node_manager::respond_to_operation, which reports unrecoverable shared memory and
 * sticky CUDA errors to the node manager (see gqe/node_manager/fault_boundary.hpp). Their jobs also
 * deserialize the request, so that a deserialization failure is reported the same way.
 *
 * ## Admission
 *
 * A task manager runs at most one job at a time. A handler reserves the task manager by setting a
 * busy flag with a compare-and-swap, and the owner thread clears it when the job ends. If the flag
 * is already set, the handler rejects the request with `FAILED_PRECONDITION` instead of queueing
 * it. The node manager runs one operation at a time, so a rejection means that a previous operation
 * is still running after the node manager gave up on it. Queueing would hide that as slowness.
 *
 * ## State
 *
 * Only the owner thread changes the state. Handlers check it before reserving, and the owner thread
 * checks it again when it picks up the job.
 *
 * - `GenerateNvshmemUid` and `InitNvshmem` require `starting`.
 * - `SetPeerInfo` requires `nvshmem_ready`.
 * - `ExecutePlan` and `RefreshStatsTables` require `ready`, and are rejected with `UNAVAILABLE`
 *   otherwise. The other RPCs are rejected with `FAILED_PRECONDITION`.
 *
 * ## Fail-Stop
 *
 * NVSHMEM cannot be initialized again in a process after a failed initialization. If a bootstrap
 * RPC fails, the service moves to `stopping`, answers the request with the error, and calls the
 * @ref fail_stop_callback. The task manager then exits, and the node manager replaces it.
 *
 * # Thread Safety
 *
 * The `on_*` methods replace callbacks that the owner thread reads without synchronization.
 * Register every callback before the gRPC server starts serving this service. Replacing a callback
 * later is only safe while no job that may invoke it is queued or running.
 */
class service final : public proto::TaskManagerService::Service {
 public:
  /** @brief Result returned by the execute callback. */
  struct execution_result {
    std::string arrow_ipc;
    bool has_write_stats = false;
    proto::TableStatistics write_stats;
  };

  /** @brief Callback invoked to execute a deserialized physical plan. */
  using execute_callback =
    std::function<execution_result(physical::identified_plan plan,
                                   utility::uuid query_id,
                                   std::vector<storage::descriptor> descriptors,
                                   gqe::optimization_parameters opt_params)>;

  /** @brief One statistics table's freshly-projected contents to materialize. */
  struct stats_table_write {
    std::string table_name;  ///< Name of the shared_segment_memory statistics table to overwrite.
    std::string arrow_ipc;   ///< Projected rows as an Arrow IPC stream batch; empty => clear only.
  };

  /** @brief Callback invoked to refresh (overwrite) the statistics tables in shared memory. */
  using refresh_stats_callback = std::function<void(std::vector<stats_table_write> writes,
                                                    utility::uuid query_id,
                                                    std::vector<storage::descriptor> descriptors,
                                                    std::uint64_t epoch)>;

  /**
   * @brief Values returned by the @ref init_nvshmem_callback after NVSHMEM initialisation.
   */
  struct init_nvshmem_result {
    void* pgas_base_ptr;                 ///< Base pointer of this rank's PGAS symmetric heap.
    std::string migration_grpc_address;  ///< Task-migration gRPC address (ip:port).
  };

  /**
   * @brief Callback invoked when the InitNvshmem RPC is received.
   *
   * The implementation should initialise NVSHMEM and the execution context, then
   * return the PGAS base pointer and task-migration gRPC address so that the
   * service can populate the InitNvshmemResponse.
   *
   * @param rank      The rank assigned by the node manager.
   * @param nranks    Total number of ranks.
   * @param uid_bytes Raw bytes of the 128-byte `nvshmemx_uniqueid_t`.
   * @return @ref init_nvshmem_result with the PGAS pointer and migration address.
   */
  using init_nvshmem_callback =
    std::function<init_nvshmem_result(int rank, int nranks, std::string_view uid_bytes)>;

  /**
   * @brief Callback invoked when the SetPeerInfo RPC is received.
   *
   * The implementation should use the peer list to build the task-migration
   * client and create the plan executor.
   *
   * @param peers Complete peer list for all ranks in the world.
   */
  using set_peer_info_callback = std::function<void(std::vector<gqe::peer_info> peers)>;

  /**
   * @brief Callback invoked on the owner thread when a bootstrap RPC fails.
   *
   * The implementation must make the process exit. It must not block, because the failed RPC's
   * response is sent only after the callback returns.
   *
   * @param reason Description of the failure.
   */
  using fail_stop_callback = std::function<void(std::string_view reason)>;

  /**
   * @brief Construct a new service.
   *
   * @param owner      The thread that runs every callback. It must outlive the service.
   * @param on_execute Callback invoked for each ExecutePlan RPC.
   */
  service(owner_thread& owner, execute_callback on_execute);

  /**
   * @brief Register the callback for the InitNvshmem RPC.
   * @param cb Callback to invoke when InitNvshmem is received.
   */
  void on_init_nvshmem(init_nvshmem_callback cb);

  /**
   * @brief Register the callback for the SetPeerInfo RPC.
   * @param cb Callback to invoke when SetPeerInfo is received.
   */
  void on_set_peer_info(set_peer_info_callback cb);

  /**
   * @brief Register the callback for the RefreshStatsTables RPC.
   * @param cb Callback to invoke when RefreshStatsTables is received.
   */
  void on_refresh_stats(refresh_stats_callback cb);

  /**
   * @brief Register the callback invoked when a bootstrap RPC fails.
   * @param cb Callback that makes the process exit.
   */
  void on_fail_stop(fail_stop_callback cb);

  /**
   * @brief Move to `stopping` from the owner thread.
   *
   * The task manager calls this from a job on the owner thread before tearing down its execution
   * context, so that no handler admits another operation.
   */
  void mark_stopping();

  /** @brief The current state. */
  [[nodiscard]] task_manager_state state() const;

  /** @name gRPC method overrides */
  /** @{ */

  grpc::Status ExecutePlan(grpc::ServerContext* context,
                           proto::ExecutePlanRequest const* request,
                           proto::ExecutePlanResponse* response) override;

  /**
   * @brief Overwrite the contents of one or more shared_segment_memory statistics tables.
   *
   * Delegates to the registered @ref refresh_stats_callback. Rank 0 writes the shared-memory copy;
   * all ranks update their local row-group references.
   */
  grpc::Status RefreshStatsTables(grpc::ServerContext* context,
                                  proto::RefreshStatsRequest const* request,
                                  proto::RefreshStatsResponse* response) override;

  grpc::Status HealthCheck(grpc::ServerContext* context,
                           proto::HealthCheckRequest const* request,
                           proto::HealthCheckResponse* response) override;

  /**
   * @brief Generate a NVSHMEM unique ID.
   *
   * Called on task manager 0 only. Returns the 128-byte UID that all ranks
   * must share during `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID)`.
   */
  grpc::Status GenerateNvshmemUid(grpc::ServerContext* context,
                                  proto::GenerateUidRequest const* request,
                                  proto::GenerateUidResponse* response) override;

  /**
   * @brief Initialise NVSHMEM on this task manager.
   *
   * Receives the shared UID, rank, and world size from the node manager.
   * Delegates to the registered @ref init_nvshmem_callback and returns
   * this rank's PGAS base pointer and task-migration gRPC address.
   */
  grpc::Status InitNvshmem(grpc::ServerContext* context,
                           proto::InitNvshmemRequest const* request,
                           proto::InitNvshmemResponse* response) override;

  /**
   * @brief Distribute the complete peer list to this task manager.
   *
   * Called after all ranks have completed InitNvshmem. Delegates to the
   * registered @ref set_peer_info_callback so the task-migration client
   * can be constructed with the full set of PGAS pointers and gRPC addresses.
   */
  grpc::Status SetPeerInfo(grpc::ServerContext* context,
                           proto::SetPeerInfoRequest const* request,
                           proto::SetPeerInfoResponse* response) override;

  /** @} */

 private:
  /**
   * @brief Reserve the task manager, run `job` on the owner thread, and wait for it.
   *
   * @param rpc_name Name of the RPC, for error messages.
   * @param required The state the owner thread requires when it picks up the job.
   * @param job      The work to run. Its return value is the RPC's status.
   * @return `FAILED_PRECONDITION` if the task manager is busy. If the state is no longer
   *         `required` when the owner thread picks up the job, the same code as the handler's
   *         state check. `UNAVAILABLE` if the owner thread has stopped. Otherwise, the status
   *         returned by `job`.
   */
  grpc::Status run_on_owner(std::string_view rpc_name,
                            task_manager_state required,
                            std::function<grpc::Status()> const& job);

  /** @brief Move to `stopping` and call the fail-stop callback. Runs on the owner thread. */
  void fail_stop(std::string_view reason);

  owner_thread& _owner;
  std::atomic<task_manager_state> _state{task_manager_state::starting};
  std::atomic<bool> _is_busy{false};
  int _rank = -1;  ///< Written by the owner thread before `_state` leaves `starting`.
  execute_callback _on_execute;
  refresh_stats_callback _on_refresh_stats;
  init_nvshmem_callback _on_init_nvshmem;
  set_peer_info_callback _on_set_peer_info;
  fail_stop_callback _on_fail_stop;
};

/** @brief The name of `state`, for messages. */
[[nodiscard]] std::string_view to_string(task_manager_state state);

}  // namespace gqe::task_manager
