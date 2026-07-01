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
#include <gqe/physical/relation.hpp>
#include <gqe/storage/descriptor.hpp>
#include <gqe/utility/uuid.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gqe {
struct task_manager_context;
struct peer_info;
}  // namespace gqe

namespace gqe::task_manager {

/**
 * @brief gRPC service implementation for a task manager process.
 *
 * Receives physical plans from the node manager, deserializes them, and
 * delegates execution to a callback.  Also handles the NVSHMEM UID bootstrap
 * RPCs (GenerateNvshmemUid, InitNvshmem, SetPeerInfo).
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
    std::function<execution_result(std::shared_ptr<physical::relation> plan,
                                   utility::uuid query_id,
                                   std::vector<storage::descriptor> descriptors,
                                   gqe::optimization_parameters opt_params)>;

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
   * The implementation should initialise NVSHMEM and the execution context,
   * then return the PGAS base pointer and task-migration gRPC address so
   * that the service can populate the InitNvshmemResponse.
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
   * @brief Construct a new service.
   *
   * @param on_execute Callback invoked for each ExecutePlan RPC.
   */
  explicit service(execute_callback on_execute);

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

  /** @name gRPC method overrides */
  /** @{ */

  grpc::Status ExecutePlan(grpc::ServerContext* context,
                           proto::ExecutePlanRequest const* request,
                           proto::ExecutePlanResponse* response) override;

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
  std::optional<int> _rank;
  execute_callback _on_execute;
  init_nvshmem_callback _on_init_nvshmem;
  set_peer_info_callback _on_set_peer_info;
};

}  // namespace gqe::task_manager
