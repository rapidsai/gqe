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

#include <memory>
#include <string>

namespace grpc {
class Server;
}

namespace gqe {
class task_manager_context;
}

namespace gqe::task_manager {

/** @brief Configuration for a task manager process. */
struct configuration {
  int grpc_port;  ///< gRPC port assigned by the node manager.
};

/**
 * @brief A task manager process that owns one GPU.
 *
 * NVSHMEM initialisation is driven by the node manager via gRPC RPCs
 * (GenerateNvshmemUid, InitNvshmem, SetPeerInfo).  The task manager starts
 * its gRPC server and waits for bootstrap commands before executing queries.
 */
class task_manager {
 public:
  explicit task_manager(configuration config);
  ~task_manager();

  /**
   * @brief Start gRPC server and enter event loop.
   *
   * Blocks until the gRPC server is shut down.
   */
  void run();

 private:
  configuration _config;
  std::unique_ptr<gqe::task_manager_context>
    _ctx;  // Single or multi-process, depending on bootstrap.
  std::unique_ptr<grpc::Server> _grpc_server;
};

}  // namespace gqe::task_manager
