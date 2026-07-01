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

#include <gqe/node_manager/nvshmem_bootstrap.hpp>

#include <gqe/utility/logger.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <format>
#include <future>

namespace gqe::node_manager {

std::string generate_nvshmem_uid(std::shared_ptr<grpc::Channel> const& rank0_channel)
{
  auto stub = proto::TaskManagerService::NewStub(rank0_channel);
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
  proto::GenerateUidRequest req;
  proto::GenerateUidResponse resp;
  auto status = stub->GenerateNvshmemUid(&ctx, req, &resp);
  if (!status.ok()) {
    throw std::runtime_error(std::format("GenerateNvshmemUid failed: {}", status.error_message()));
  }
  GQE_LOG_INFO("Received NVSHMEM UID from task manager 0 ({} bytes)", resp.unique_id().size());
  return resp.unique_id();
}

std::vector<nvshmem_peer_result> init_nvshmem(
  std::span<std::shared_ptr<grpc::Channel> const> channels, std::string_view uid_bytes)
{
  auto const num_ranks = static_cast<int>(channels.size());
  std::vector<nvshmem_peer_result> peer_results(num_ranks);
  std::vector<std::future<void>> futures;

  // NVSHMEM init is a collective — all ranks must call it concurrently.
  for (int rank = 0; rank < num_ranks; ++rank) {
    futures.push_back(std::async(std::launch::async, [&, rank]() {
      auto stub = proto::TaskManagerService::NewStub(channels[rank]);
      grpc::ClientContext ctx;
      ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(180));
      proto::InitNvshmemRequest req;
      req.set_unique_id(std::string{uid_bytes});
      req.set_rank(rank);
      req.set_num_ranks(num_ranks);

      proto::InitNvshmemResponse resp;
      auto status = stub->InitNvshmem(&ctx, req, &resp);
      if (!status.ok()) {
        throw std::runtime_error(
          std::format("InitNvshmem failed for rank {}: {}", rank, status.error_message()));
      }
      if (!resp.success()) {
        throw std::runtime_error(
          std::format("InitNvshmem failed for rank {}: {}", rank, resp.error_message()));
      }
      peer_results[rank] = {
        .rank = rank, .pgas_base_ptr = resp.pgas_base_ptr(), .grpc_address = resp.grpc_address()};
      GQE_LOG_INFO(
        "Task manager rank {} initialized NVSHMEM (migration addr: {})", rank, resp.grpc_address());
    }));
  }
  for (auto& f : futures) {
    f.get();
  }
  return peer_results;
}

void distribute_peer_info(std::span<std::shared_ptr<grpc::Channel> const> channels,
                          std::span<nvshmem_peer_result const> peers)
{
  auto const num_ranks = static_cast<int>(channels.size());

  proto::SetPeerInfoRequest peer_req;
  for (auto const& pr : peers) {
    auto* peer = peer_req.add_peers();
    peer->set_rank(pr.rank);
    peer->set_pgas_base_ptr(pr.pgas_base_ptr);
    peer->set_grpc_address(pr.grpc_address);
  }

  for (int rank = 0; rank < num_ranks; ++rank) {
    auto stub = proto::TaskManagerService::NewStub(channels[rank]);
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
    proto::SetPeerInfoResponse resp;
    auto status = stub->SetPeerInfo(&ctx, peer_req, &resp);
    if (!status.ok()) {
      throw std::runtime_error(
        std::format("SetPeerInfo failed for rank {}: {}", rank, status.error_message()));
    }
    if (!resp.success()) {
      throw std::runtime_error(std::format("SetPeerInfo failed for rank {}", rank));
    }
  }
  GQE_LOG_INFO("All {} task managers bootstrapped via NVSHMEM UID", num_ranks);
}

void bootstrap_nvshmem(std::span<std::shared_ptr<grpc::Channel> const> channels)
{
  auto uid   = generate_nvshmem_uid(channels[0]);
  auto peers = init_nvshmem(channels, uid);
  distribute_peer_info(channels, peers);
}

}  // namespace gqe::node_manager
