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
#include <gqe/task_manager/owner_thread.hpp>
#include <gqe/task_manager/plan_executor.hpp>
#include <gqe/task_manager/service.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/fault_injection.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/thread_entry.hpp>

#include <grpcpp/grpcpp.h>
#include <nvshmemx.h>

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

namespace gqe::task_manager {

namespace {

/// Time the gRPC server gets to send the failed bootstrap RPC's response before it shuts down.
constexpr auto fail_stop_grace_period = std::chrono::seconds(5);

#ifdef GQE_ENABLE_FAULT_INJECTION
/**
 * @brief Test-only fault injection that fails `SetPeerInfo` on demand.
 *
 * If the trigger file named by the environment variable `GQE_TEST_FAIL_SET_PEER_INFO_FILE` (see
 * `gqe/utility/fault_injection.hpp`) has a line for @p rank, this sleeps for the line's optional
 * delay and then throws. It runs after the peer info has been applied, so the failure comes at the
 * end of bootstrap: the node manager sees a failed bootstrap, the failing task manager fail-stops,
 * and lower ranks that already received their peer info stay ready. Tests use it to hold the node
 * manager in its restart-retry loop, and the delay to make each failing restart attempt take a
 * known time.
 *
 * Compiled only with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by default.
 */
void inject_set_peer_info_failure_for_tests(int rank)
{
  auto const trigger =
    utility::fault_injection::find_trigger("GQE_TEST_FAIL_SET_PEER_INFO_FILE", rank);
  if (!trigger) { return; }

  std::this_thread::sleep_for(trigger->delay);
  throw std::runtime_error(std::format("Injected SetPeerInfo failure for rank {}", rank));
}

/**
 * @brief Test-only fault injection that fails `InitNvshmem` on demand.
 *
 * If the trigger file named by the environment variable `GQE_TEST_FAIL_INIT_NVSHMEM_FILE` (see
 * `gqe/utility/fault_injection.hpp`) has a line for @p rank, this sleeps for the line's optional
 * delay and then throws. It runs before the rank joins the NVSHMEM collective, so with several
 * ranks the other ranks wait in the collective for a rank that never arrives: the failure that
 * the node manager must report at once, instead of after the deadline of each rank in turn. Tests
 * use it to check that a bootstrap with one failed rank fails fast.
 *
 * Compiled only with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by default.
 */
void inject_init_nvshmem_failure_for_tests(int rank)
{
  auto const trigger =
    utility::fault_injection::find_trigger("GQE_TEST_FAIL_INIT_NVSHMEM_FILE", rank);
  if (!trigger) { return; }

  std::this_thread::sleep_for(trigger->delay);
  throw std::runtime_error(std::format("Injected InitNvshmem failure for rank {}", rank));
}
#endif

}  // namespace

task_manager::task_manager(configuration config) : _config(std::move(config)) {}

task_manager::~task_manager()
{
  if (_grpc_server) { _grpc_server->Shutdown(); }
}

void task_manager::enter_device(int rank)
{
  _device = rmm::cuda_device_id{rank % utility::get_device_count()};
  utility::enter_device_thread(*_device);
  GQE_LOG_INFO("Task manager rank {} uses device {}", rank, _device->value());
}

void task_manager::run()
{
  GQE_LOG_INFO("Task manager starting on port {}", _config.grpc_port);

  owner_thread owner;
  if (_config.rank >= 0) {
    owner.submit([this] { enter_device(_config.rank); }).get();
  }

  // The service runs every callback on the owner thread, and only in the state that guarantees
  // the members it uses exist.
  service service(owner,
                  [this](physical::identified_plan plan,
                         utility::uuid query_id,
                         std::vector<storage::descriptor> descriptors,
                         gqe::optimization_parameters opt_params) -> service::execution_result {
                    return (*_executor)(
                      std::move(plan), query_id, std::move(descriptors), std::move(opt_params));
                  });

  service.on_refresh_stats([this](std::vector<service::stats_table_write> writes,
                                  utility::uuid query_id,
                                  std::vector<storage::descriptor> descriptors,
                                  std::uint64_t epoch) {
    _executor->refresh_stats(std::move(writes), query_id, std::move(descriptors), epoch);
  });

  // InitNvshmem callback: initialise NVSHMEM and create the execution context.
  service.on_init_nvshmem(
    [this](int rank, int nranks, std::string_view uid_bytes) -> service::init_nvshmem_result {
      if (_config.rank >= 0 && rank != _config.rank) {
        throw std::runtime_error(
          std::format("Spawn rank {} does not match NVSHMEM rank {}", _config.rank, rank));
      }
#ifdef GQE_ENABLE_FAULT_INJECTION
      inject_init_nvshmem_failure_for_tests(rank);
#endif
      if (!_device) { enter_device(rank); }

      nvshmemx_uniqueid_t uid;
      if (uid_bytes.size() != sizeof(uid)) { throw std::runtime_error("Invalid NVSHMEM UID size"); }
      std::memcpy(&uid, uid_bytes.data(), sizeof(uid));

      // Create context with an empty peer list; peers arrive via SetPeerInfo.
      _ctx = multi_process_task_manager_context::default_init(rank, nranks, uid, *_device, {});

      // Eagerly initialise the shared memory resource.
      std::ignore = _ctx->get_table_memory_resource(memory_kind::shared_segment{});

      auto* mp_ctx = static_cast<multi_process_task_manager_context*>(_ctx.get());
      return {mp_ctx->pgas_base_ptr(), mp_ctx->migration_grpc_address()};
    });

  // SetPeerInfo callback: connect the migration client and create the executor.
  service.on_set_peer_info([this](std::vector<gqe::peer_info> peers) {
    auto* mp_ctx = static_cast<multi_process_task_manager_context*>(_ctx.get());
    mp_ctx->connect_peers(peers);

    bool const multi_process = mp_ctx->comm->world_size() > 1;
    _executor = std::make_unique<plan_executor>(_ctx.get(), mp_ctx->comm->rank(), multi_process);

#ifdef GQE_ENABLE_FAULT_INJECTION
    inject_set_peer_info_failure_for_tests(mp_ctx->comm->rank());
#endif
  });

  // A failed bootstrap leaves NVSHMEM unusable in this process. Shut down through the same path
  // as SIGTERM: the signal thread below receives the signal and shuts down the gRPC server.
  service.on_fail_stop([this](std::string_view) {
    _is_fail_stopped.store(true, std::memory_order_release);
    kill(getpid(), SIGTERM);
  });

  // Start gRPC server.
  auto address = std::format("{}:{}", grpc_localhost_addr, _config.grpc_port);

  int bound_port = 0;
  grpc::ServerBuilder builder;
  apply_default_server_args(builder);
  builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &bound_port);
  builder.RegisterService(&service);
  _grpc_server = builder.BuildAndStart();
  if (!_grpc_server || bound_port != _config.grpc_port) {
    // The server refers to `service` and `owner`, which are destroyed when this throws.
    if (_grpc_server) { _grpc_server->Shutdown(); }
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
    auto grpc_shutdown_deadline = std::chrono::system_clock::now();
    if (_is_fail_stopped.load(std::memory_order_acquire)) {
      GQE_LOG_CRITICAL("Task manager shutting down after a failed bootstrap");
      // Give the failed RPC time to send its response.
      grpc_shutdown_deadline += fail_stop_grace_period;
    } else {
      GQE_LOG_CRITICAL("Task manager received signal {}, shutting down", sig);
    }
    _grpc_server->Shutdown(grpc_shutdown_deadline);
  });

  _grpc_server->Wait();

  // Release CUDA and NVSHMEM resources on the thread that created them.
  bool const is_fail_stopped = _is_fail_stopped.load(std::memory_order_acquire);
  owner
    .submit([this, &service, is_fail_stopped] {
      service.mark_stopping();
      _executor.reset();
      if (is_fail_stopped) {
        // Freeing NVSHMEM memory is collective. The other ranks are not freeing theirs, so a
        // context left by a failed `SetPeerInfo` is released by process exit instead.
        std::ignore = _ctx.release();
      }
      _ctx.reset();
    })
    .get();
  owner.stop();

  if (is_fail_stopped) {
    throw std::runtime_error("Task manager stopped after a failed bootstrap RPC");
  }
}

}  // namespace gqe::task_manager
