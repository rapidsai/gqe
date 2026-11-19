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

#include <rmm/cuda_device.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

namespace grpc {
class Server;
}  // namespace grpc

namespace gqe {
class task_manager_context;
}  // namespace gqe

namespace gqe::task_manager {

class plan_executor;

/** @brief Configuration for a task manager process. */
struct configuration {
  int grpc_port;            ///< gRPC port assigned by the node manager.
  int rank = -1;            ///< Spawn-time GPU rank, or -1 in standalone mode.
  std::string log_address;  ///< Dedicated node-manager logging endpoint.
};

/**
 * @brief A task manager process that owns one GPU.
 *
 * NVSHMEM initialisation is driven by the node manager via gRPC RPCs
 * (GenerateNvshmemUid, InitNvshmem, SetPeerInfo). The task manager starts
 * its gRPC server and waits for bootstrap commands before executing queries.
 *
 * # Design
 *
 * All initialization and every operation run on one @ref owner_thread. The task manager decides its
 * device once, as the spawn rank modulo the number of visible devices, and the owner thread enters
 * it before anything else runs. Without a spawn rank, the device is decided from the NVSHMEM rank
 * when `InitNvshmem` arrives. The execution context and the plan executor are created, used, and
 * destroyed only on the owner thread.
 *
 * If a bootstrap RPC fails, the task manager fails stop: it answers the RPC, shuts down its gRPC
 * server, and `run` throws so that the process exits with a non-zero status.
 */
class task_manager {
 public:
  explicit task_manager(configuration config);
  ~task_manager();

  /**
   * @brief Start gRPC server and enter event loop.
   *
   * Blocks until the gRPC server is shut down.
   *
   * @throw std::runtime_error if the task manager stopped after a failed bootstrap RPC.
   */
  void run();

 private:
  /** @brief Decide the device from `rank` and enter it. Runs on the owner thread. */
  void enter_device(int rank);

  configuration _config;
  std::optional<rmm::cuda_device_id> _device;  ///< Owner thread only.
  std::unique_ptr<gqe::task_manager_context>
    _ctx;  ///< Single or multi-process, depending on bootstrap. Owner thread only.
  std::unique_ptr<plan_executor> _executor;  ///< Created by `SetPeerInfo`. Owner thread only.
  std::unique_ptr<grpc::Server> _grpc_server;
  std::atomic<bool> _is_fail_stopped{false};
};

}  // namespace gqe::task_manager
