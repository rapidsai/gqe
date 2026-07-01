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

#include <grpcpp/channel.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gqe::node_manager {

/**
 * @brief Result of the InitNvshmem RPC for a single rank.
 */
struct nvshmem_peer_result {
  int32_t rank;              ///< NVSHMEM rank.
  uint64_t pgas_base_ptr;    ///< PGAS base pointer as uint64_t.
  std::string grpc_address;  ///< Task-migration gRPC address (ip:port).
};

/**
 * @brief Ask rank 0 to generate an NVSHMEM unique ID.
 *
 * @param rank0_channel gRPC channel to task manager rank 0.
 * @return The 128-byte unique ID as a string.
 * @throws std::runtime_error if the RPC fails.
 */
[[nodiscard]] std::string generate_nvshmem_uid(std::shared_ptr<grpc::Channel> const& rank0_channel);

/**
 * @brief Initialise NVSHMEM on all task managers.
 *
 * Sends the InitNvshmem RPC to all ranks in parallel (NVSHMEM init is a
 * collective) and collects the PGAS base pointers and task-migration gRPC
 * addresses from each rank.
 *
 * @param channels gRPC channels to all task managers, indexed by rank.
 * @param uid_bytes The 128-byte unique ID from @ref generate_nvshmem_uid.
 * @return Per-rank results with PGAS pointers and gRPC addresses.
 * @throws std::runtime_error if any RPC fails.
 */
[[nodiscard]] std::vector<nvshmem_peer_result> init_nvshmem(
  std::span<std::shared_ptr<grpc::Channel> const> channels, std::string_view uid_bytes);

/**
 * @brief Distribute the peer list to all task managers.
 *
 * Sends the SetPeerInfo RPC to each rank with the complete set of PGAS base
 * pointers and task-migration gRPC addresses.
 *
 * @param channels gRPC channels to all task managers, indexed by rank.
 * @param peers Per-rank results from @ref init_nvshmem.
 * @throws std::runtime_error if any RPC fails.
 */
void distribute_peer_info(std::span<std::shared_ptr<grpc::Channel> const> channels,
                          std::span<nvshmem_peer_result const> peers);

/**
 * @brief Run the full NVSHMEM bootstrap sequence.
 *
 * Convenience function that calls @ref generate_nvshmem_uid,
 * @ref init_nvshmem, and @ref distribute_peer_info in order.
 *
 * @param channels gRPC channels to all task managers, indexed by rank.
 * @throws std::runtime_error if any phase fails.
 */
void bootstrap_nvshmem(std::span<std::shared_ptr<grpc::Channel> const> channels);

}  // namespace gqe::node_manager
