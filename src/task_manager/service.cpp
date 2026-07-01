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

#include <gqe/task_manager/service.hpp>

#include <gqe/rpc/serialization/optimization_parameters.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/rpc/serialization/storage.hpp>
#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/utility/logger.hpp>

#include <nvshmemx.h>

#include <format>

namespace gqe::task_manager {

service::service(execute_callback on_execute) : _on_execute(std::move(on_execute)) {}

void service::on_init_nvshmem(init_nvshmem_callback cb) { _on_init_nvshmem = std::move(cb); }

void service::on_set_peer_info(set_peer_info_callback cb) { _on_set_peer_info = std::move(cb); }

grpc::Status service::ExecutePlan(grpc::ServerContext* context,
                                  proto::ExecutePlanRequest const* request,
                                  proto::ExecutePlanResponse* response)
{
  if (!_rank) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        "ExecutePlan called before NVSHMEM initialization");
  }

  try {
    auto query_id = rpc::deserialize_uuid(request->query_id());
    GQE_LOG_TRACE("Task manager rank {}: received request (query {})", *_rank, query_id);
    auto plan = rpc::deserialize_physical_plan(request->physical_plan());

    // Deserialize the storage descriptors from the request.
    std::vector<storage::descriptor> descriptors;
    descriptors.reserve(request->table_catalog_size());
    for (auto const& desc : request->table_catalog()) {
      descriptors.push_back(rpc::deserialize_storage_descriptor(desc));
    }

    auto opt_params =
      rpc::deserialize_optimization_parameters(request->optimization_parameters()).ValueOrDie();
    auto result =
      _on_execute(std::move(plan), query_id, std::move(descriptors), std::move(opt_params));

    response->set_success(true);
    response->set_rank(*_rank);
    if (!result.arrow_ipc.empty()) { response->set_arrow_ipc_result(std::move(result.arrow_ipc)); }
    if (result.has_write_stats) {
      *response->mutable_write_statistics() = std::move(result.write_stats);
    }
    GQE_LOG_TRACE("Task manager rank {}: sending response (query {})", *_rank, query_id);
  } catch (std::exception const& e) {
    GQE_LOG_ERROR("Task manager rank {} failed to execute plan: {}", *_rank, e.what());
    response->set_success(false);
    response->set_rank(*_rank);
    response->set_error_message(e.what());
  }
  return grpc::Status::OK;
}

grpc::Status service::HealthCheck(grpc::ServerContext* context,
                                  proto::HealthCheckRequest const* request,
                                  proto::HealthCheckResponse* response)
{
  bool is_initialized = _rank.has_value();
  response->set_healthy(is_initialized);
  if (is_initialized) { response->set_rank(*_rank); }
  return grpc::Status::OK;
}

grpc::Status service::GenerateNvshmemUid(grpc::ServerContext* context,
                                         proto::GenerateUidRequest const* request,
                                         proto::GenerateUidResponse* response)
{
  try {
    nvshmemx_uniqueid_t uid;
    nvshmemx_get_uniqueid(&uid);
    response->set_unique_id(&uid, sizeof(uid));
    GQE_LOG_INFO("Task manager: generated NVSHMEM UID");
  } catch (std::exception const& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::format("GenerateNvshmemUid failed: {}", e.what()));
  }
  return grpc::Status::OK;
}

grpc::Status service::InitNvshmem(grpc::ServerContext* context,
                                  proto::InitNvshmemRequest const* request,
                                  proto::InitNvshmemResponse* response)
{
  try {
    auto result = _on_init_nvshmem(request->rank(), request->num_ranks(), request->unique_id());

    _rank = request->rank();

    response->set_success(true);
    response->set_pgas_base_ptr(reinterpret_cast<uint64_t>(result.pgas_base_ptr));
    response->set_grpc_address(result.migration_grpc_address);

    GQE_LOG_INFO(
      "Task manager rank {}: NVSHMEM initialized (nranks={})", _rank.value(), request->num_ranks());
  } catch (std::exception const& e) {
    GQE_LOG_ERROR("Task manager rank {}: InitNvshmem failed: {}", _rank.value_or(-1), e.what());
    response->set_success(false);
    response->set_error_message(e.what());
  }
  return grpc::Status::OK;
}

grpc::Status service::SetPeerInfo(grpc::ServerContext* context,
                                  proto::SetPeerInfoRequest const* request,
                                  proto::SetPeerInfoResponse* response)
{
  // SetPeerInfo requires _rank, which is set by InitNvshmem.
  // TODO: rethink task manager initialization across multiple RPCs.
  // The ordering dependency between them is fragile.
  if (!_rank) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "SetPeerInfo called before NVSHMEM initialization");
  }
  try {
    std::vector<gqe::peer_info> peers;
    peers.reserve(request->peers_size());
    for (auto const& proto_peer : request->peers()) {
      peers.push_back({.rank          = proto_peer.rank(),
                       .pgas_base_ptr = reinterpret_cast<void*>(proto_peer.pgas_base_ptr()),
                       .grpc_address  = proto_peer.grpc_address()});
    }
    _on_set_peer_info(std::move(peers));
    response->set_success(true);
    GQE_LOG_INFO(
      "Task manager rank {}: peer info set ({} peers)", _rank.value(), request->peers_size());
  } catch (std::exception const& e) {
    GQE_LOG_ERROR("Task manager rank {}: SetPeerInfo failed: {}", _rank.value_or(-1), e.what());
    response->set_success(false);
  }
  return grpc::Status::OK;
}

}  // namespace gqe::task_manager
