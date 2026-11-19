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

#include <gqe/node_manager/fault_boundary.hpp>
#include <gqe/rpc/serialization/optimization_parameters.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/rpc/serialization/storage.hpp>
#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/task_manager/owner_thread.hpp>
#include <gqe/utility/fault_injection.hpp>
#include <gqe/utility/logger.hpp>

#include <nvshmemx.h>

#include <format>
#include <future>
#include <thread>

namespace gqe::task_manager {

namespace {

#ifdef GQE_ENABLE_FAULT_INJECTION
/**
 * @brief Test-only fault injection that stalls an RPC handler on demand.
 *
 * If the trigger file named by the environment variable @p trigger_variable (see
 * `gqe/utility/fault_injection.hpp`) has a line for @p rank, this sleeps for the line's delay on
 * the gRPC handler thread. That is before the request is queued for the owner thread, so the rank
 * has neither checked for poisoning nor started the operation, has published nothing, and still
 * answers `HealthCheck`. The operation then runs normally. Tests use it to keep an RPC running past
 * the point where the node manager gives up on it, for example when its deadline expires or the
 * node tears down.
 *
 * The variables are `GQE_TEST_STALL_EXECUTE_PLAN_FILE` for `ExecutePlan` and
 * `GQE_TEST_STALL_REFRESH_STATS_FILE` for `RefreshStatsTables`.
 *
 * Compiled only with the CMake option `GQE_ENABLE_FAULT_INJECTION`, which is off by default.
 */
void inject_stall_for_tests(char const* trigger_variable, int rank)
{
  auto const trigger = utility::fault_injection::find_trigger(trigger_variable, rank);
  if (trigger) { std::this_thread::sleep_for(trigger->delay); }
}
#endif

/**
 * @brief Status code for a request that arrives in the wrong state.
 *
 * Operations sent before the task manager is ready get `UNAVAILABLE`, as a server that is not
 * serving yet. Bootstrap RPCs sent out of order get `FAILED_PRECONDITION`.
 */
grpc::StatusCode state_rejection_code(task_manager_state required)
{
  return required == task_manager_state::ready ? grpc::StatusCode::UNAVAILABLE
                                               : grpc::StatusCode::FAILED_PRECONDITION;
}

/** @brief Reject a request unless the task manager is in state `required`. */
grpc::Status check_state(std::string_view rpc_name,
                         task_manager_state required,
                         task_manager_state current)
{
  if (current == required) { return grpc::Status::OK; }
  return grpc::Status(state_rejection_code(required),
                      std::format("{} requires the task manager to be {}, but it is {}",
                                  rpc_name,
                                  to_string(required),
                                  to_string(current)));
}

/** @brief Clears the busy flag when a job ends, including when it throws. */
class busy_release {
 public:
  explicit busy_release(std::atomic<bool>& is_busy) : _is_busy(is_busy) {}
  ~busy_release() { _is_busy.store(false, std::memory_order_release); }

  busy_release(busy_release const&)            = delete;
  busy_release& operator=(busy_release const&) = delete;

 private:
  std::atomic<bool>& _is_busy;
};

}  // namespace

std::string_view to_string(task_manager_state state)
{
  switch (state) {
    case task_manager_state::starting: return "starting";
    case task_manager_state::nvshmem_ready: return "nvshmem_ready";
    case task_manager_state::ready: return "ready";
    case task_manager_state::stopping: return "stopping";
  }
  return "unknown";
}

service::service(owner_thread& owner, execute_callback on_execute)
  : _owner(owner), _on_execute(std::move(on_execute))
{
}

void service::on_refresh_stats(refresh_stats_callback cb) { _on_refresh_stats = std::move(cb); }

void service::on_init_nvshmem(init_nvshmem_callback cb) { _on_init_nvshmem = std::move(cb); }

void service::on_set_peer_info(set_peer_info_callback cb) { _on_set_peer_info = std::move(cb); }

void service::on_fail_stop(fail_stop_callback cb) { _on_fail_stop = std::move(cb); }

void service::mark_stopping()
{
  _state.store(task_manager_state::stopping, std::memory_order_release);
}

task_manager_state service::state() const { return _state.load(std::memory_order_acquire); }

void service::fail_stop(std::string_view reason)
{
  GQE_LOG_CRITICAL("Task manager bootstrap failed, exiting: {}", reason);
  mark_stopping();
  if (_on_fail_stop) { _on_fail_stop(reason); }
}

grpc::Status service::run_on_owner(std::string_view rpc_name,
                                   task_manager_state required,
                                   std::function<grpc::Status()> const& job)
{
  bool expected = false;
  if (!_is_busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return grpc::Status(
      grpc::StatusCode::FAILED_PRECONDITION,
      std::format("{} rejected: the task manager is still running another operation", rpc_name));
  }

  // The job only runs while this handler waits for it, so it can capture locals by reference.
  grpc::Status status;
  std::future<void> done;
  try {
    done = _owner.submit([&] {
      busy_release const release(_is_busy);
      status = check_state(rpc_name, required, state());
      if (status.ok()) { status = job(); }
    });
  } catch (std::exception const& e) {
    _is_busy.store(false, std::memory_order_release);
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        std::format("{} rejected: {}", rpc_name, e.what()));
  }

  try {
    done.get();
  } catch (std::future_error const&) {
    // The owner thread stopped before running the job, so the job never cleared the flag.
    _is_busy.store(false, std::memory_order_release);
    return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                        std::format("{} not run: the task manager is shutting down", rpc_name));
  } catch (std::exception const& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::format("{} failed: {}", rpc_name, e.what()));
  }
  return status;
}

grpc::Status service::ExecutePlan(grpc::ServerContext* context,
                                  proto::ExecutePlanRequest const* request,
                                  proto::ExecutePlanResponse* response)
{
  if (auto status = check_state("ExecutePlan", task_manager_state::ready, state()); !status.ok()) {
    return status;
  }

#ifdef GQE_ENABLE_FAULT_INJECTION
  inject_stall_for_tests("GQE_TEST_STALL_EXECUTE_PLAN_FILE", _rank);
#endif

  return run_on_owner("ExecutePlan", task_manager_state::ready, [&] {
    node_manager::respond_to_operation(*response, _rank, "ExecutePlan", [&] {
      auto const query_id = rpc::deserialize_uuid(request->query_id());
      GQE_LOG_TRACE("Task manager rank {}: received request (query {})", _rank, query_id);
      auto identified_plan = rpc::deserialize_physical_plan(request->physical_plan());

      std::vector<storage::descriptor> descriptors;
      descriptors.reserve(request->table_catalog_size());
      for (auto const& desc : request->table_catalog()) {
        descriptors.push_back(rpc::deserialize_storage_descriptor(desc));
      }

      auto opt_params =
        rpc::deserialize_optimization_parameters(request->optimization_parameters()).ValueOrDie();

      auto result = _on_execute(
        std::move(identified_plan), query_id, std::move(descriptors), std::move(opt_params));
      if (!result.arrow_ipc.empty()) {
        response->set_arrow_ipc_result(std::move(result.arrow_ipc));
      }
      if (result.has_write_stats) {
        *response->mutable_write_statistics() = std::move(result.write_stats);
      }
      GQE_LOG_TRACE("Task manager rank {}: sending response (query {})", _rank, query_id);
    });
    return grpc::Status::OK;
  });
}

grpc::Status service::RefreshStatsTables(grpc::ServerContext* context,
                                         proto::RefreshStatsRequest const* request,
                                         proto::RefreshStatsResponse* response)
{
  if (auto status = check_state("RefreshStatsTables", task_manager_state::ready, state());
      !status.ok()) {
    return status;
  }

#ifdef GQE_ENABLE_FAULT_INJECTION
  inject_stall_for_tests("GQE_TEST_STALL_REFRESH_STATS_FILE", _rank);
#endif

  return run_on_owner("RefreshStatsTables", task_manager_state::ready, [&] {
    node_manager::respond_to_operation(*response, _rank, "RefreshStatsTables", [&] {
      auto const query_id = rpc::deserialize_uuid(request->query_id());

      std::vector<stats_table_write> writes;
      writes.reserve(request->writes_size());
      for (auto const& write : request->writes()) {
        writes.push_back({write.table_name(), write.arrow_ipc()});
      }

      std::vector<storage::descriptor> descriptors;
      descriptors.reserve(request->table_catalog_size());
      for (auto const& desc : request->table_catalog()) {
        descriptors.push_back(rpc::deserialize_storage_descriptor(desc));
      }

      _on_refresh_stats(std::move(writes), query_id, std::move(descriptors), request->epoch());
    });
    return grpc::Status::OK;
  });
}

grpc::Status service::HealthCheck(grpc::ServerContext* context,
                                  proto::HealthCheckRequest const* request,
                                  proto::HealthCheckResponse* response)
{
  auto const current = state();
  bool const is_healthy =
    current == task_manager_state::nvshmem_ready || current == task_manager_state::ready;
  response->set_healthy(is_healthy);
  if (is_healthy) { response->set_rank(_rank); }
  return grpc::Status::OK;
}

grpc::Status service::GenerateNvshmemUid(grpc::ServerContext* context,
                                         proto::GenerateUidRequest const* request,
                                         proto::GenerateUidResponse* response)
{
  if (auto status = check_state("GenerateNvshmemUid", task_manager_state::starting, state());
      !status.ok()) {
    return status;
  }

  return run_on_owner("GenerateNvshmemUid", task_manager_state::starting, [&] {
    nvshmemx_uniqueid_t uid;
    if (auto const result = nvshmemx_get_uniqueid(&uid); result != 0) {
      auto message =
        std::format("GenerateNvshmemUid failed: nvshmemx_get_uniqueid returned {}", result);
      fail_stop(message);
      return grpc::Status(grpc::StatusCode::INTERNAL, std::move(message));
    }
    response->set_unique_id(&uid, sizeof(uid));
    GQE_LOG_INFO("Task manager: generated NVSHMEM UID");
    return grpc::Status::OK;
  });
}

grpc::Status service::InitNvshmem(grpc::ServerContext* context,
                                  proto::InitNvshmemRequest const* request,
                                  proto::InitNvshmemResponse* response)
{
  if (auto status = check_state("InitNvshmem", task_manager_state::starting, state());
      !status.ok()) {
    return status;
  }

  return run_on_owner("InitNvshmem", task_manager_state::starting, [&] {
    try {
      auto result = _on_init_nvshmem(request->rank(), request->num_ranks(), request->unique_id());

      _rank = request->rank();
      _state.store(task_manager_state::nvshmem_ready, std::memory_order_release);

      response->set_success(true);
      response->set_pgas_base_ptr(reinterpret_cast<uint64_t>(result.pgas_base_ptr));
      response->set_grpc_address(result.migration_grpc_address);

      GQE_LOG_INFO(
        "Task manager rank {}: NVSHMEM initialized (nranks={})", _rank, request->num_ranks());
    } catch (std::exception const& e) {
      response->set_success(false);
      response->set_error_message(e.what());
      fail_stop(std::format("InitNvshmem failed on rank {}: {}", request->rank(), e.what()));
    }
    return grpc::Status::OK;
  });
}

grpc::Status service::SetPeerInfo(grpc::ServerContext* context,
                                  proto::SetPeerInfoRequest const* request,
                                  proto::SetPeerInfoResponse* response)
{
  if (auto status = check_state("SetPeerInfo", task_manager_state::nvshmem_ready, state());
      !status.ok()) {
    return status;
  }

  std::vector<gqe::peer_info> peers;
  peers.reserve(request->peers_size());
  for (auto const& proto_peer : request->peers()) {
    peers.push_back({.rank          = proto_peer.rank(),
                     .pgas_base_ptr = reinterpret_cast<void*>(proto_peer.pgas_base_ptr()),
                     .grpc_address  = proto_peer.grpc_address()});
  }

  return run_on_owner("SetPeerInfo", task_manager_state::nvshmem_ready, [&] {
    try {
      _on_set_peer_info(std::move(peers));
      _state.store(task_manager_state::ready, std::memory_order_release);
      response->set_success(true);
      GQE_LOG_INFO("Task manager rank {}: peer info set ({} peers)", _rank, request->peers_size());
    } catch (std::exception const& e) {
      response->set_success(false);
      response->set_error_message(e.what());
      fail_stop(std::format("SetPeerInfo failed on rank {}: {}", _rank, e.what()));
    }
    return grpc::Status::OK;
  });
}

}  // namespace gqe::task_manager
