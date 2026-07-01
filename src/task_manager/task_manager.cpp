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

#include <gqe/task_manager/task_manager.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/task_manager/plan_executor.hpp>
#include <gqe/task_manager/service.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/logger.hpp>

#include <grpcpp/grpcpp.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>

#include <signal.h>

#include <cstring>
#include <format>
#include <thread>

namespace gqe::task_manager {

task_manager::task_manager(configuration config) : _config(std::move(config)) {}

task_manager::~task_manager()
{
  if (_grpc_server) { _grpc_server->Shutdown(); }
}

void task_manager::run()
{
  GQE_LOG_INFO("Task manager starting on port {}", _config.grpc_port);

  // The plan_executor is created after NVSHMEM init + peer info are received.
  std::unique_ptr<plan_executor> executor;

  // Rank and GPU id are assigned during the InitNvshmem RPC, before any
  // ExecutePlan can arrive.
  service service(
    [&executor](std::shared_ptr<physical::relation> plan,
                utility::uuid query_id,
                std::vector<storage::descriptor> descriptors,
                gqe::optimization_parameters opt_params) -> service::execution_result {
      if (!executor) { throw std::runtime_error("ExecutePlan called before NVSHMEM init"); }
      return (*executor)(std::move(plan), query_id, std::move(descriptors), std::move(opt_params));
    });

  // InitNvshmem callback: initialise NVSHMEM and create the execution context.
  service.on_init_nvshmem(
    [this](int rank, int nranks, std::string_view uid_bytes) -> service::init_nvshmem_result {
      cudaSetDevice(rank);

      nvshmemx_uniqueid_t uid;
      if (uid_bytes.size() != sizeof(uid)) { throw std::runtime_error("Invalid NVSHMEM UID size"); }
      std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

      // Create context with an empty peer list; peers arrive via SetPeerInfo.
      _ctx = multi_process_task_manager_context::default_init(rank, nranks, uid, {});

      // Eagerly initialise the boost shared memory resource.
      std::ignore = _ctx->get_table_memory_resource(memory_kind::boost_shared{});

      auto* mp_ctx = static_cast<multi_process_task_manager_context*>(_ctx.get());
      return {mp_ctx->pgas_base_ptr(), mp_ctx->migration_grpc_address()};
    });

  // SetPeerInfo callback: wire up the migration client and create the executor.
  service.on_set_peer_info([this, &executor](std::vector<gqe::peer_info> peers) {
    auto* mp_ctx = static_cast<multi_process_task_manager_context*>(_ctx.get());

    mp_ctx->migration_client = std::make_unique<nvshmem_task_migration_client>(
      static_cast<nvshmem_communicator*>(mp_ctx->comm.get()), std::span{peers});

    bool const multi_process = mp_ctx->comm->world_size() > 1;
    executor = std::make_unique<plan_executor>(_ctx.get(), mp_ctx->comm->rank(), multi_process);
  });

  // Start gRPC server.
  auto address = std::format("0.0.0.0:{}", _config.grpc_port);

  int bound_port = 0;
  grpc::ServerBuilder builder;
  apply_default_server_args(builder);
  builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &bound_port);
  builder.RegisterService(&service);
  _grpc_server = builder.BuildAndStart();
  if (!_grpc_server || bound_port != _config.grpc_port) {
    throw std::runtime_error(std::format("Failed to bind task manager gRPC server to {}", address));
  }

  GQE_LOG_INFO("Task manager listening on {}", address);

  // Shut down the gRPC server on SIGTERM (sent by the node manager).
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGTERM);
  sigaddset(&mask, SIGINT);
  pthread_sigmask(SIG_BLOCK, &mask, nullptr);

  std::jthread signal_thread([this, &mask](std::stop_token) {
    int sig;
    sigwait(&mask, &sig);
    GQE_LOG_CRITICAL("Task manager received signal {}, shutting down", sig);
    auto grpc_shutdown_deadline = std::chrono::system_clock::now();
    _grpc_server->Shutdown(grpc_shutdown_deadline);
  });

  _grpc_server->Wait();
}

}  // namespace gqe::task_manager
