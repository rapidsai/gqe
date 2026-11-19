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
#include <mutex>
#include <optional>

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

  // One context per rank, so that the failure of one rank can cancel the RPCs of the others.
  std::vector<grpc::ClientContext> contexts(num_ranks);
  std::mutex error_mutex;
  std::optional<std::string> first_error;  // Guarded by error_mutex.

  // NVSHMEM init is a collective, so the other ranks cannot finish once one has failed. They would
  // wait until their own 180 s deadline, one after the other, before the failure surfaced. Record
  // the first failure and cancel every RPC, so that the caller learns of it at once and can
  // terminate the group. The error that is reported is the first one, not the cancellations that
  // it causes.
  auto const fail = [&](std::string message) {
    {
      std::scoped_lock guard(error_mutex);
      if (!first_error) { first_error = std::move(message); }
    }
    for (auto& context : contexts) {
      context.TryCancel();
    }
  };

  // Declared after the state that the workers use, so that unwinding joins the workers (the
  // destructor of a future from `std::async` waits) before that state is destroyed.
  std::vector<std::future<void>> futures;
  futures.reserve(num_ranks);

  // NVSHMEM init is a collective — all ranks must call it concurrently.
  auto const init_rank = [&](int rank) {
    try {
      auto stub = proto::TaskManagerService::NewStub(channels[rank]);
      auto& ctx = contexts[rank];
      ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(180));
      proto::InitNvshmemRequest req;
      req.set_unique_id(std::string{uid_bytes});
      req.set_rank(rank);
      req.set_num_ranks(num_ranks);

      proto::InitNvshmemResponse resp;
      auto status = stub->InitNvshmem(&ctx, req, &resp);
      if (!status.ok()) {
        fail(std::format("InitNvshmem failed for rank {}: {}", rank, status.error_message()));
        return;
      }
      if (!resp.success()) {
        fail(std::format("InitNvshmem failed for rank {}: {}", rank, resp.error_message()));
        return;
      }
      peer_results[rank] = {
        .rank = rank, .pgas_base_ptr = resp.pgas_base_ptr(), .grpc_address = resp.grpc_address()};
      GQE_LOG_INFO(
        "Task manager rank {} initialized NVSHMEM (migration addr: {})", rank, resp.grpc_address());
    } catch (std::exception const& error) {
      // Not only an RPC failure ends the collective: an allocation or a logging failure here would
      // leave the other ranks waiting for this one until their deadline.
      fail(std::format("InitNvshmem failed for rank {}: {}", rank, error.what()));
    }
  };
  try {
    for (int rank = 0; rank < num_ranks; ++rank) {
      futures.push_back(std::async(std::launch::async, init_rank, rank));
    }
  } catch (std::exception const& error) {
    // The workers already started would wait for a rank that never starts. Cancel them, so that
    // unwinding, which joins them, does not take their deadline.
    fail(std::format("Could not start the NVSHMEM initialization of every rank: {}", error.what()));
    throw;
  }
  // Every RPC has finished, or has been cancelled and returns promptly, once this loop is done.
  for (auto& f : futures) {
    f.get();
  }
  if (first_error) { throw std::runtime_error(*first_error); }
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
      throw std::runtime_error(
        std::format("SetPeerInfo failed for rank {}: {}", rank, resp.error_message()));
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
