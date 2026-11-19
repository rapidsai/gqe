/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/node_manager/node_manager.hpp>

#include <gqe/catalog.hpp>
#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/node_manager/log_drain.hpp>
#include <gqe/node_manager/log_stream_service.hpp>
#include <gqe/node_manager/nvshmem_bootstrap.hpp>
#include <gqe/node_manager/service.hpp>
#include <gqe/node_manager/spawn.hpp>
#include <gqe/node_manager/stats_tables.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/optimizer/optimization_configuration.hpp>
#include <gqe/optimizer/physical_transformation.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/write.hpp>
#include <gqe/rpc/serialization/optimization_parameters.hpp>
#include <gqe/rpc/serialization/physical_plan.hpp>
#include <gqe/rpc/serialization/statistics.hpp>
#include <gqe/rpc/serialization/storage.hpp>
#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/rpc/task_migration.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/fault_injection.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/serialization.hpp>
#include <gqe/utility/uuid.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <arrow/flight/types.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <signal.h>

#include <cstdlib>
#include <exception>
#include <format>
#include <future>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

namespace gqe::node_manager {

namespace {

/**
 * Visitor that collects table names from read and write relations in a physical plan.
 * The base visitor's default implementations recurse into children and subqueries.
 */
struct table_name_collector : public physical::relation_visitor {
  std::unordered_set<std::string> table_names;

  void visit(physical::read_relation* rel) override
  {
    table_names.insert(rel->table_name());
    visit_children(rel);
  }
  void visit(physical::write_relation* rel) override
  {
    table_names.insert(rel->table_name());
    visit_children(rel);
  }
};

/**
 * @brief Error from a plan execution in which some rank did not confirm that it finished.
 *
 * A rank confirms by returning a response, successful or not. A transport failure, such as an
 * expired deadline or a cancelled RPC, confirms nothing, and the rank may still be executing the
 * plan after the error is reported. The node manager therefore terminates the group before it
 * reports the error, see @ref node_manager::fail_group_unconfirmed.
 */
class unconfirmed_completion_error : public task_manager_group_failed_error {
 public:
  using task_manager_group_failed_error::task_manager_group_failed_error;
};

/**
 * @brief Return the revision of the table's published row groups, if its storage tracks one.
 *
 * Only shared-segment tables advance their revision on every publication. Other storage kinds
 * report a constant revision, so it cannot show whether a write published anything.
 */
std::optional<std::uint64_t> published_revision(gqe::catalog const& catalog,
                                                std::string const& table_name)
{
  if (!std::holds_alternative<gqe::storage_kind::shared_segment_memory>(
        catalog.storage_kind(table_name))) {
    return std::nullopt;
  }
  return catalog.table_storage(table_name)->storage_revision();
}

/// Flag a table whose failed write may have left published row groups, see catalog::mark_unusable.
void mark_unusable_after_failed_write(gqe::catalog& catalog, std::string const& table_name) noexcept
{
  try {
    if (catalog.is_unusable(table_name)) { return; }
    catalog.mark_unusable(table_name);
    GQE_LOG_ERROR(
      "A write to table '{}' failed after it may have published data. The table is unusable until "
      "it is dropped.",
      table_name);
  } catch (std::exception const& error) {
    GQE_LOG_ERROR("Cannot flag table '{}' as unusable: {}", table_name, error.what());
  } catch (...) {
    GQE_LOG_ERROR("Cannot flag table '{}' as unusable", table_name);
  }
}

/// Serialize a record batch to an Arrow IPC stream buffer.
std::string serialize_record_batch_to_ipc(arrow::RecordBatch const& batch)
{
  auto sink   = arrow::io::BufferOutputStream::Create().ValueOrDie();
  auto writer = arrow::ipc::MakeStreamWriter(sink, batch.schema()).ValueOrDie();
  auto status = writer->WriteRecordBatch(batch);
  if (!status.ok()) {
    throw std::runtime_error(std::format("Failed to write stats Arrow IPC: {}", status.ToString()));
  }
  status = writer->Close();
  if (!status.ok()) {
    throw std::runtime_error(
      std::format("Failed to close stats Arrow IPC writer: {}", status.ToString()));
  }
  auto buffer = sink->Finish().ValueOrDie();
  return std::string(reinterpret_cast<char const*>(buffer->data()), buffer->size());
}

/**
 * @brief Return a copy of `batch` with its columns renamed to `names`.
 *
 * Column types and data are preserved; only the schema field names change. `names` are the
 * query's real output column names (the Substrait RelRoot.names captured during parsing).
 *
 * The rename is guarded:
 * - If `names` is empty or its size doesn't match the column count, `batch` is returned
 *   unchanged, keeping the placeholder names the executor emits (column_0, column_1, ...).
 * - Count and names always match for the flat output schemas GQE supports; nested structs, which
 *   could make the flattened name count differ from the column count, are rejected upstream.
 *
 * @param batch The result batch whose columns to relabel.
 * @param names The output column names to apply, one per column.
 * @return A copy of `batch` with `names` applied, or `batch` unchanged if the guard trips.
 */
std::shared_ptr<arrow::RecordBatch> relabel_batch_columns(std::shared_ptr<arrow::RecordBatch> batch,
                                                          std::vector<std::string> const& names)
{
  if (!batch || names.size() != static_cast<std::size_t>(batch->num_columns())) { return batch; }

  arrow::FieldVector fields;
  fields.reserve(names.size());
  for (int i = 0; i < batch->num_columns(); ++i) {
    fields.push_back(batch->schema()->field(i)->WithName(names[i]));
  }
  return arrow::RecordBatch::Make(
    arrow::schema(std::move(fields)), batch->num_rows(), batch->columns());
}

}  // namespace

// ============================================================================
// Construction / destruction
// ============================================================================

node_manager::node_manager(configuration config, std::shared_ptr<log_drain> drain)
  : _config(std::move(config)), _log_drain(std::move(drain))
{
  if (!_log_drain) { throw std::invalid_argument("Node manager log drain cannot be null"); }
  if (_config.task_manager_base_port == 0) { _config.task_manager_base_port = _config.port + 1000; }

  _ctx     = std::make_unique<context>(_config.opt_params, _config.shmem_policy);
  _catalog = std::make_unique<gqe::catalog>(_ctx.get());

  // Always-present system tables that expose statistics to SQL clients; populated lazily on read.
  register_stats_tables(*_catalog);
}

node_manager::~node_manager() { shutdown(); }

// ============================================================================
// Serve / shutdown
// ============================================================================

exit_code::value node_manager::serve()
{
  auto code = exit_code::success;
  try {
    auto const status = serve_impl();
    if (!status.ok()) {
      GQE_LOG_ERROR(failure_log_format, status.ToString());
      code = exit_code::failure;
    }
  } catch (std::exception const& e) {
    GQE_LOG_ERROR(failure_log_format, e.what());
    code = exit_code::failure;
  }
  if (auto const cause = _teardown.cause()) {
    GQE_LOG_CRITICAL("Node manager stopped because {}: {}",
                     teardown_reason_description(_teardown.reason().value()),
                     *cause);
    return _teardown.exit_status();
  }
  return code;
}

arrow::Status node_manager::serve_impl()
{
  _log_stream_service = std::make_unique<log_stream_service>(_log_drain);
  grpc::ServerBuilder log_server_builder;
  apply_default_server_args(log_server_builder);
  int log_port = 0;
  log_server_builder.AddListeningPort(
    std::format("{}:0", grpc_localhost_addr), grpc::InsecureServerCredentials(), &log_port);
  log_server_builder.RegisterService(_log_stream_service.get());
  _log_server = log_server_builder.BuildAndStart();
  if (!_log_server || log_port == 0) {
    return arrow::Status::IOError("Failed to start node-manager log server");
  }
  _log_address = std::format("{}:{}", grpc_localhost_addr, log_port);
  GQE_LOG_INFO("Log server listening on {}", _log_address);

  {
    auto group = bring_up_group(std::stop_token{});
    // No group means node teardown began, for example because every start attempt failed.
    if (!group) { _teardown.throw_error(); }
    install_group(std::move(group));
  }

  // Create the Flight service with execution callbacks.
  _service = std::make_unique<service>(
    _catalog.get(),
    [this](void const* data, std::size_t size) {
      return _teardown.guard([&] { return execute_substrait_query(data, size); });
    },
    [this](void const* data, std::size_t size) {
      return _teardown.guard([&] { return execute_substrait_statement(data, size); });
    },
    [this](void const* data, std::size_t size) {
      return _teardown.guard([&] { return execute_physical_plan_query(data, size); });
    },
    [this](std::string const& name, arrow::flight::SessionOptionValue const& value) {
      std::scoped_lock guard(_execute_mutex);
      auto status = rpc::apply_session_option(_config.opt_params, name, value);
      if (status.ok()) {
        // Keep the node-manager context in sync with the session-updated parameters. Without this,
        // options set via SET (such as the compression format) never reach the serialize path.
        _ctx->set_optimization_parameters(_config.opt_params);
      }
      return status;
    },
    [this]() -> arrow::Result<std::map<std::string, arrow::flight::SessionOptionValue>> {
      std::scoped_lock guard(_execute_mutex);
      return rpc::optimization_parameters_to_session_options(_config.opt_params);
    });

  // Start the monitor only once _service exists, because node teardown stops it from the monitor
  // thread.
  _monitor_thread =
    std::jthread([this](std::stop_token token) { monitor_task_managers(std::move(token)); });

  // Start Flight SQL server
  auto location = arrow::flight::Location::ForGrpcTcp(_config.listen_address, _config.port);
  if (!location.ok()) { return location.status(); }

  arrow::flight::FlightServerOptions options(*location);
  options.middleware.emplace_back("teardown_gate", std::make_shared<teardown_gate>(_teardown));

  GQE_LOG_INFO("Node manager starting on {}:{} with {} GPU(s)",
               _config.listen_address,
               _config.port,
               _config.num_gpus);

  {
    std::scoped_lock guard(_server_mutex);
    ARROW_RETURN_NOT_OK(_service->Init(options));
    _server_initialized = true;
  }
  // Teardown that began before Init() found no server to stop. Teardown that begins from here on
  // stops the server, so Serve() returns.
  if (_teardown.has_begun()) {
    shutdown();
    return arrow::Status::OK();
  }
  ARROW_RETURN_NOT_OK(_service->SetShutdownOnSignals({SIGINT, SIGTERM, SIGHUP}));
  ARROW_RETURN_NOT_OK(_service->Serve());

  // Serve() returned — either a signal was received or Shutdown() was called.
  shutdown();
  return arrow::Status::OK();
}

void node_manager::shutdown()
{
  // The log server starts before the task-manager monitor. Keep shutdown idempotence independent
  // of the monitor's stop state so an early launch or bootstrap failure still releases the server.
  if (_shutdown_started.exchange(true, std::memory_order_acq_rel)) { return; }

  if (_monitor_thread.joinable()) {
    _monitor_thread.request_stop();
    _monitor_thread.join();
  }

  constexpr auto task_manager_shutdown_timeout = std::chrono::seconds(5);

  GQE_LOG_CRITICAL("Node manager shutting down");

  if (_service) {
    auto status = stop_flight_server();
    if (!status.ok()) { GQE_LOG_WARN("Flight SQL server shutdown failed: {}", status.ToString()); }
  } else {
    cancel_in_flight_rpcs();
  }

  {
    std::shared_ptr<task_manager_group> group;
    {
      std::scoped_lock guard(_task_manager_mutex);
      group = std::move(_group);
    }
    if (group) {
      GQE_LOG_INFO("Terminating task managers");
      group->terminate(task_manager_shutdown_timeout);
      GQE_LOG_INFO("All task managers terminated");
    }
  }

  if (_log_server) {
    _log_server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds{1});
    _log_server->Wait();
    _log_server.reset();
  }
}

// ============================================================================
// In-flight RPC cancellation
// ============================================================================

void node_manager::cancel_in_flight_rpcs()
{
  std::scoped_lock guard(_in_flight_mutex);
  for (auto* ctx : _in_flight_rpcs) {
    ctx->TryCancel();
  }
}

void node_manager::register_in_flight_rpc(grpc::ClientContext& ctx, [[maybe_unused]] int rank)
{
#ifdef GQE_ENABLE_FAULT_INJECTION
  // Test-only: hold the handler here, so that a test can have the server stop before this RPC is
  // registered. The trigger file lists the rank and the pause in seconds.
  if (auto const trigger = utility::fault_injection::find_trigger(
        "GQE_TEST_PAUSE_BEFORE_RPC_REGISTRATION_FILE", rank)) {
    std::this_thread::sleep_for(trigger->delay);
  }
#endif
  std::scoped_lock guard(_in_flight_mutex);
  // Teardown can begin before the monitor thread reaches `stop_flight_server`, for example while it
  // brings up a group, so the server can still be serving when it has begun. `has_begun` is a
  // lock-free atomic read, and nothing that holds `_in_flight_mutex` begins teardown.
  if (_teardown.has_begun() || _is_server_stopping.load()) {
    ctx.TryCancel();
    return;
  }
  _in_flight_rpcs.push_back(&ctx);
}

arrow::Status node_manager::stop_flight_server()
{
  std::scoped_lock guard(_server_mutex);
  {
    std::scoped_lock in_flight_guard(_in_flight_mutex);
    _is_server_stopping.store(true);
  }
  // Operations that wait for a group notice that the server is stopping.
  _group_cv.notify_all();
  cancel_in_flight_rpcs();
  if (!_server_initialized) { return arrow::Status::OK(); }
  return _service->Shutdown();
}

// ============================================================================
// Task manager subprocess management
// ============================================================================

std::shared_ptr<task_manager_group> node_manager::start_group()
{
  auto group = std::make_shared<task_manager_group>(
    _next_generation++,
    spawn_task_managers(
      _config.task_manager_binary, _config.task_manager_base_port, _config.num_gpus, _log_address));
  GQE_LOG_INFO("Bootstrapping task manager group {}", group->generation());

  try {
    bootstrap_nvshmem(group->channels());
  } catch (...) {
    // The ranks that did not fail are stuck in the NVSHMEM collective, and do not react to SIGTERM
    // until it ends, so a graceful stop would only add the grace period to every failed attempt.
    // A group that never served an operation has nothing to save.
    group->terminate(std::chrono::seconds(0));
    throw;
  }
  group->mark_ready();
  return group;
}

std::shared_ptr<task_manager_group> node_manager::bring_up_group(std::stop_token const& token)
{
  constexpr int max_attempts = 3;
  auto backoff               = std::chrono::seconds(1);
  std::string last_error;

  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    // A task manager that poisoned the segment may have died right after. Starting task managers
    // on a segment that is no longer trusted would be pointless.
    if (_teardown.begin_if_poisoned("detected before starting task managers") ||
        token.stop_requested()) {
      return nullptr;
    }
    try {
      return start_group();
    } catch (std::exception const& e) {
      last_error = e.what();
    } catch (...) {
      last_error = "unknown error";
    }
    GQE_LOG_ERROR("Starting the task managers failed (attempt {} of {}): {}",
                  attempt,
                  max_attempts,
                  last_error);
    if (attempt < max_attempts) {
      _teardown.wait(token, backoff);
      backoff *= 2;
    }
  }

  _teardown.begin(teardown_reason::task_managers_unavailable,
                  std::format("the task managers failed to start after {} attempts, last error: {}",
                              max_attempts,
                              last_error));
  return nullptr;
}

void node_manager::install_group(std::shared_ptr<task_manager_group> group)
{
  std::shared_ptr<task_manager_group> previous;
  {
    std::scoped_lock guard(_task_manager_mutex);
    previous = std::exchange(_group, std::move(group));
  }
  _group_cv.notify_all();
  // `previous` is released here, outside the lock. Operations that still hold it keep it alive.
}

void node_manager::restart_after_rank_fatal(task_manager_group& group, std::string message)
{
  _teardown.throw_if_poisoned(message);
  // The monitor thread replaces the group. Replacing it here would hold the caller's
  // _execute_mutex for the whole restart.
  group.mark_failed();
  throw task_manager_group_failed_error(std::format(
    "A task manager reported a fatal error, and the task managers are being restarted: {}",
    message));
}

void node_manager::fail_group_unconfirmed(task_manager_group& group, std::string const& error)
{
  GQE_LOG_ERROR(
    "An operation did not finish on every task manager of group {}. Terminating the group, "
    "because a rank that did not confirm may still be running it: {}",
    group.generation(),
    error);
  // Done before the caller releases _execute_mutex. Otherwise the next operation could meet a rank
  // that is still running this one, or appends of this one could race a recovery of its table. The
  // monitor thread starts the replacement.
  //
  // The ranks get no grace period: a rank that is stalled or still running the operation would use
  // it to finish the operation, and publish what the node manager has just given up on. Killing a
  // rank is safe for the shared memory, because a kill cannot write to the segment, and a lock it
  // held is reported as unrecoverable, see shared_memory_unrecoverable_error.
  group.terminate(std::chrono::seconds(0));
  throw unconfirmed_completion_error(std::format(
    "The task managers failed during this operation and are being restarted (group {}): {}",
    group.generation(),
    error));
}

void node_manager::stop_for_teardown()
{
  // Teardown runs here rather than on the Flight SQL handler thread that began it, because
  // Shutdown() waits for every in-progress handler to return. serve() then calls shutdown(), which
  // terminates the task managers instead of restarting them.
  //
  // A handler can still be waiting for a task manager, for example for a rank blocked in a
  // collective with a rank that reported node_fatal, which the handler reads only later.
  // Shutdown() would wait for that RPC's deadline (query_timeout), and a deadline passed to
  // Shutdown() would not help, because Arrow still waits for running handlers. Cancelling the RPCs
  // makes the handler return. A handler that waits for a group notices that the server is stopping
  // or that teardown began.
  GQE_LOG_CRITICAL("Node manager: stopping the Flight SQL server for node teardown");
  auto status = stop_flight_server();
  if (status.ok()) { return; }

  // serve() would never return, so finish the teardown here instead.
  int const exit_status = _teardown.exit_status();
  GQE_LOG_CRITICAL(
    "Flight SQL server shutdown for node teardown failed: {}. Terminating task managers, removing "
    "the shared memory, and exiting with status {}",
    status.ToString(),
    exit_status);
  {
    std::shared_ptr<task_manager_group> group;
    {
      std::scoped_lock guard(_task_manager_mutex);
      group = _group;
    }
    if (group) { group->terminate(); }
  }
  boost::interprocess::shared_memory_object::remove(shared_memory_name);
  std::_Exit(exit_status);
}

void node_manager::supervise_task_managers(std::stop_token const& token)
{
  std::shared_ptr<task_manager_group> group;
  {
    std::scoped_lock guard(_task_manager_mutex);
    group = _group;
  }

  if (group && group->is_ready()) {
    auto const status = group->try_wait_any();
    if (!status) { return; }
    switch (status->kind) {
      case exit_kind::exited:
        GQE_LOG_ERROR("Task manager rank {} (pid {}) exited with status {}",
                      status->rank,
                      status->pid,
                      status->code);
        break;
      case exit_kind::signaled:
        GQE_LOG_ERROR("Task manager rank {} (pid {}) killed by signal {}",
                      status->rank,
                      status->pid,
                      status->code);
        break;
    }
    group->mark_failed();
    // The operation that was running on the group cannot complete.
    cancel_in_flight_rpcs();
  }

  // The group failed: a rank exited, a rank reported a rank-fatal error, or no group was ever
  // installed. Replace it. Dispatch waits for the replacement.
  if (group) {
    GQE_LOG_WARN("Restarting all task managers (group {} failed)", group->generation());
    group->terminate();
  }
  if (auto replacement = bring_up_group(token)) { install_group(std::move(replacement)); }
}

void node_manager::monitor_task_managers(std::stop_token token)
{
  // An exception escaping this thread calls `std::terminate`, so nothing is allowed to escape. A
  // failed start is retried inside bring_up_group, and a group that could not be started tears the
  // node down.
  while (!token.stop_requested()) {
    // Catches poisoning recorded in the segment between operations, by any process, and teardown
    // that began elsewhere, including because no group could be started.
    if (_teardown.begin_if_poisoned("detected by the monitor")) {
      stop_for_teardown();
      return;
    }
    try {
      supervise_task_managers(token);
    } catch (std::exception const& e) {
      GQE_LOG_ERROR("Failed to supervise the task managers, retrying: {}", e.what());
    } catch (...) {
      GQE_LOG_ERROR("Failed to supervise the task managers with an unknown error, retrying");
    }
    _teardown.wait(token, std::chrono::seconds(1));
  }
}

std::shared_ptr<task_manager_group> node_manager::acquire_ready_group()
{
  std::unique_lock lock(_task_manager_mutex);
  while (true) {
    // Teardown and server stop are not signalled to this wait, so poll for them.
    _teardown.throw_if_begun();
    if (_is_server_stopping.load()) {
      throw std::runtime_error("The node manager is shutting down");
    }
    if (_group && _group->is_ready()) { return _group; }
    _group_cv.wait_for(lock, std::chrono::milliseconds(500));
  }
}

// ============================================================================
// Multi-GPU query execution
// ============================================================================

node_manager::execution_result node_manager::execute_on_task_managers(
  std::shared_ptr<gqe::physical::relation> physical_plan, bool* is_dispatched)
{
  if (is_dispatched != nullptr) { *is_dispatched = false; }
  auto identified_plan       = physical::make_identified_plan(physical_plan);
  auto serialized_plan       = rpc::serialize_physical_plan(identified_plan);
  auto serialized_opt_params = rpc::serialize_optimization_parameters(_config.opt_params);
  auto query_id              = utility::uuid::generate();

  // Build the table catalog: only the tables referenced by this plan.
  table_name_collector collector;
  physical_plan->accept(collector);
  auto const& referenced_tables = collector.table_names;
  for (auto const& name : referenced_tables) {
    if (_catalog->is_unusable(name)) {
      throw std::runtime_error(std::format(
        "Table '{}' is incomplete after a failed write. DROP it and load it again.", name));
    }
  }
  google::protobuf::RepeatedPtrField<proto::StorageDescriptor> table_catalog;
  for (auto const& name : referenced_tables) {
    *table_catalog.Add() = rpc::serialize_storage_descriptor(_catalog->storage_descriptor(name));
  }

  // The group stays alive for the whole operation, also if it fails and is replaced meanwhile.
  auto const group     = acquire_ready_group();
  auto const& channels = group->channels();

  struct in_flight_rpc {
    grpc::ClientContext ctx;
    std::future<rank_result<proto::ExecutePlanResponse>> future;
  };
  std::vector<in_flight_rpc> rpcs(channels.size());

  // Unregisters the RPCs on every exit, including exceptions, before `rpcs` is destroyed, so that
  // cancel_in_flight_rpcs() never uses a destroyed context.
  struct in_flight_registration {
    node_manager& self;
    std::vector<in_flight_rpc>& rpcs;
    in_flight_registration(node_manager& self, std::vector<in_flight_rpc>& rpcs)
      : self(self), rpcs(rpcs)
    {
    }
    in_flight_registration(in_flight_registration const&)            = delete;
    in_flight_registration& operator=(in_flight_registration const&) = delete;
    ~in_flight_registration()
    {
      std::scoped_lock guard(self._in_flight_mutex);
      for (auto& rpc : rpcs) {
        std::erase(self._in_flight_rpcs, &rpc.ctx);
      }
    }
  } const registration{*this, rpcs};

  // Collect results from all ranks
  std::vector<std::shared_ptr<arrow::RecordBatch>> partial_results;
  gqe::table_statistics aggregated_write_stats;
  bool has_write_stats = false;

  // See the task manager error contract in the execute_on_task_managers() Doxygen. Each holds the
  // first error of its kind (see rank_outcome).
  std::string first_error;
  std::string rank_fatal_error;
  std::string node_fatal_error;
  std::size_t num_dispatched = 0;
  std::size_t num_confirmed  = 0;
  auto const cancel_after    = [&](std::size_t rank) {
    for (std::size_t j = rank + 1; j < rpcs.size(); ++j) {
      rpcs[j].ctx.TryCancel();
    }
  };

  try {
    for (std::size_t i = 0; i < channels.size(); ++i) {
      rpcs[i].ctx.set_deadline(std::chrono::system_clock::now() + _config.query_timeout);

      register_in_flight_rpc(rpcs[i].ctx, static_cast<int>(i));

      rpcs[i].future = std::async(std::launch::async, [&, i]() {
        proto::ExecutePlanRequest request;
        *request.mutable_physical_plan()           = serialized_plan;
        *request.mutable_query_id()                = rpc::serialize_uuid(query_id);
        *request.mutable_table_catalog()           = table_catalog;
        *request.mutable_optimization_parameters() = serialized_opt_params;

        GQE_LOG_TRACE("Node manager: submitting query {} to rank {}", query_id, i);
        task_manager_client client(channels[i], static_cast<int>(i));
        auto result = client.execute_plan(rpcs[i].ctx, request);
        GQE_LOG_TRACE("Node manager: received response for query {} from rank {} (success={})",
                      query_id,
                      i,
                      result.outcome == rank_outcome::success);
        return result;
      });
      ++num_dispatched;
      if (is_dispatched != nullptr) { *is_dispatched = true; }
    }

    for (std::size_t i = 0; i < rpcs.size(); ++i) {
      auto result = rpcs[i].future.get();
      if (result.is_confirmed()) { ++num_confirmed; }
      switch (result.outcome) {
        case rank_outcome::success: break;
        case rank_outcome::node_fatal:
          if (node_fatal_error.empty()) {
            node_fatal_error = std::move(result.message);
            cancel_after(i);
          }
          continue;
        case rank_outcome::rank_fatal:
          if (rank_fatal_error.empty()) {
            rank_fatal_error = std::move(result.message);
            cancel_after(i);
          }
          continue;
        case rank_outcome::error:
        case rank_outcome::transport_failure:
          if (first_error.empty()) { first_error = std::move(result.message); }
          continue;
      }
      // Keep reading after a fatal error only to find a node-fatal error. Results are discarded.
      if (!rank_fatal_error.empty() || !node_fatal_error.empty()) { continue; }
      auto& response = result.response;
      if (response.has_write_statistics()) {
        has_write_stats = true;
        aggregated_write_stats.append_table_statistics(
          rpc::deserialize_table_statistics(response.write_statistics()));
      }
      if (!response.arrow_ipc_result().empty()) {
        // Move IPC bytes out of the response — response.arrow_ipc_result() is invalid after this.
        auto owned_buffer =
          arrow::Buffer::FromString(std::move(*response.mutable_arrow_ipc_result()));
        auto reader = arrow::ipc::RecordBatchStreamReader::Open(
                        std::make_shared<arrow::io::BufferReader>(owned_buffer))
                        .ValueOrDie();
        auto batch = reader->Next().ValueOrDie();
        if (batch) { partial_results.push_back(batch); }
      }
    }
  } catch (std::exception const& error) {
    // A rank whose response was not collected may still be executing the plan.
    if (num_confirmed < num_dispatched) { fail_group_unconfirmed(*group, error.what()); }
    throw;
  }

  if (!node_fatal_error.empty()) {
    _teardown.begin_and_throw(teardown_reason::shared_memory_unrecoverable,
                              std::move(node_fatal_error));
  }

  if (!rank_fatal_error.empty()) { restart_after_rank_fatal(*group, std::move(rank_fatal_error)); }

  // A rank may have poisoned the segment and then been cancelled. The flag in the segment still
  // records it.
  _teardown.throw_if_poisoned("node manager");

  if (!first_error.empty()) {
    if (num_confirmed < num_dispatched) { fail_group_unconfirmed(*group, first_error); }
    throw std::runtime_error(first_error);
  }

  auto write_stats = has_write_stats ? std::optional{aggregated_write_stats} : std::nullopt;

  if (partial_results.empty()) { return {nullptr, write_stats}; }

  auto table    = arrow::Table::FromRecordBatches(partial_results).ValueOrDie();
  auto combined = table->CombineChunks().ValueOrDie();
  auto reader   = std::make_shared<arrow::TableBatchReader>(*combined);
  std::shared_ptr<arrow::RecordBatch> batch;
  auto st = reader->ReadNext(&batch);
  if (!st.ok()) {
    throw std::runtime_error(std::format("Failed to read combined batch: {}", st.ToString()));
  }
  return {batch, write_stats};
}

void node_manager::refresh_stats_tables_if_stale(
  std::unordered_set<std::string> const& referenced_tables)
{
  bool touches_stats = false;
  for (auto const& name : referenced_tables) {
    if (is_stats_table(name)) {
      touches_stats = true;
      break;
    }
  }
  if (!touches_stats) { return; }

  // The catalog statistics version only advances on a statistics change (write) or a table being
  // registered/unregistered. If nothing changed since the last materialization, the shared tables
  // are already current.
  auto const current_version = _catalog->stats_version();
  if (_materialized_stats_version == current_version) { return; }

  GQE_LOG_TRACE("Refreshing statistics tables (version {} -> {})",
                _materialized_stats_version.value_or(0),
                current_version);

  // Project the current statistics into Arrow batches and build the refresh request once; it is
  // identical for every rank (rank 0 writes the shared copy, others update local references).
  proto::RefreshStatsRequest request;
  *request.mutable_query_id() = rpc::serialize_uuid(utility::uuid::generate());
  request.set_epoch(current_version);

  auto const table_stats_name  = std::string{table_stats_table_name};
  auto const column_stats_name = std::string{column_stats_table_name};

  auto* table_write = request.add_writes();
  table_write->set_table_name(table_stats_name);
  table_write->set_arrow_ipc(serialize_record_batch_to_ipc(*project_table_stats(*_catalog)));

  auto* column_write = request.add_writes();
  column_write->set_table_name(column_stats_name);
  column_write->set_arrow_ipc(serialize_record_batch_to_ipc(*project_column_stats(*_catalog)));

  *request.add_table_catalog() =
    rpc::serialize_storage_descriptor(_catalog->storage_descriptor(table_stats_name));
  *request.add_table_catalog() =
    rpc::serialize_storage_descriptor(_catalog->storage_descriptor(column_stats_name));

  auto const group     = acquire_ready_group();
  auto const& channels = group->channels();
  if (channels.empty()) {
    _materialized_stats_version = current_version;
    return;
  }

  // Only rank 0 publishes the shared statistics row groups; every other rank reattaches to the
  // shared contents on its next read of the table (that read's descriptor carries the bumped
  // storage_revision — see remote_table_cache::update). So the data-carrying refresh only needs to
  // reach rank 0. Channel index matches task-manager rank throughout the node manager (see
  // execute_on_task_managers), so channels[0] is rank 0; the task-manager side additionally guards
  // the write on nvshmem_rank_zero().
  struct in_flight_rpc {
    grpc::ClientContext ctx;
    std::future<rank_result<proto::RefreshStatsResponse>> future;
  };
  in_flight_rpc rpc;
  rpc.ctx.set_deadline(std::chrono::system_clock::now() + _config.query_timeout);
  register_in_flight_rpc(rpc.ctx, 0);
  rpc.future = std::async(std::launch::async, [&]() {
    task_manager_client client(channels[0], 0);
    return client.refresh_stats(rpc.ctx, request);
  });

  // Barrier on rank 0's write before the caller's dependent read.
  auto result = rpc.future.get();
  {
    std::scoped_lock guard(_in_flight_mutex);
    std::erase(_in_flight_rpcs, &rpc.ctx);
  }
  switch (result.outcome) {
    case rank_outcome::success: break;
    case rank_outcome::node_fatal:
      _teardown.begin_and_throw(teardown_reason::shared_memory_unrecoverable,
                                std::move(result.message));
    case rank_outcome::rank_fatal: restart_after_rank_fatal(*group, std::move(result.message));
    // Rank 0 may still be writing the statistics tables after a transport failure.
    case rank_outcome::transport_failure: fail_group_unconfirmed(*group, result.message);
    case rank_outcome::error: throw std::runtime_error(result.message);
  }

  _materialized_stats_version = current_version;
}

// ============================================================================
// Substrait execution — delegates to task managers
// ============================================================================

namespace {

gqe::optimizer::optimization_configuration make_default_optimizer_rules()
{
  return gqe::optimizer::optimization_configuration(
    {gqe::optimizer::logical_optimization_rule_type::constant_folding,
     gqe::optimizer::logical_optimization_rule_type::projection_pushdown,
     gqe::optimizer::logical_optimization_rule_type::normalize_decimal_widths,
     gqe::optimizer::logical_optimization_rule_type::identity_project_elimination,
     gqe::optimizer::logical_optimization_rule_type::complex_expression_extraction_into_project,
     gqe::optimizer::logical_optimization_rule_type::join_predicate_extraction_into_project,
     gqe::optimizer::logical_optimization_rule_type::sort_fetch_to_top_k,
     gqe::optimizer::logical_optimization_rule_type::mean_decomposition,
     gqe::optimizer::logical_optimization_rule_type::string_to_int_literal,
     gqe::optimizer::logical_optimization_rule_type::uniqueness_propagation,
     gqe::optimizer::logical_optimization_rule_type::join_unique_keys,
     gqe::optimizer::logical_optimization_rule_type::aggregate_perfect_hash,
     gqe::optimizer::logical_optimization_rule_type::join_perfect_hash,
     gqe::optimizer::logical_optimization_rule_type::fix_partial_filter_column_references},
    {});
}

}  // namespace

std::shared_ptr<arrow::RecordBatch> node_manager::execute_substrait_query(void const* data,
                                                                          std::size_t size)
{
  std::scoped_lock guard(_execute_mutex);
  _teardown.throw_if_begun();

  gqe::substrait_parser parser(_catalog.get(), &_config.opt_params);
  auto queries = parser.from_binary(data, size);

  if (queries.empty()) { return nullptr; }

  GQE_EXPECTS(queries.size() == 1,
              std::format("Expected exactly one parsed query relation, got {}", queries.size()));
  auto const& query = queries.front();

  if (query.relation->type() == gqe::logical::relation::relation_type::write) {
    throw std::logic_error("execute_substrait_query called with a write plan");
  }

  auto rule_config = make_default_optimizer_rules();
  gqe::optimizer::logical_optimizer optimizer(&rule_config, _catalog.get());

  GQE_LOG_TRACE("Catalog statistics before optimization:\n{}", _catalog->to_string());

  auto optimized = optimizer.optimize(query.relation);
  GQE_LOG_TRACE("Optimized logical plan:\n{}", optimized->to_string());

  gqe::physical_plan_builder plan_builder(_catalog.get(), &_config.opt_params);
  auto physical_plan = plan_builder.build(optimized.get());
  GQE_LOG_TRACE("Physical plan:\n{}", physical_plan->to_string());

  // If this query reads a statistics table, lazily re-materialize it from the catalog before the
  // read is dispatched (no-op if already current).
  table_name_collector collector;
  physical_plan->accept(collector);
  refresh_stats_tables_if_stale(collector.table_names);

  auto batch = execute_on_task_managers(physical_plan).batch;
  return relabel_batch_columns(std::move(batch), query.output_column_names);
}

std::optional<gqe::table_statistics> node_manager::execute_substrait_statement(void const* data,
                                                                               std::size_t size)
{
  std::scoped_lock guard(_execute_mutex);
  _teardown.throw_if_begun();

  gqe::substrait_parser parser(_catalog.get(), &_config.opt_params);
  auto queries = parser.from_binary(data, size);

  if (queries.empty()) { return {}; }

  GQE_EXPECTS(queries.size() == 1,
              std::format("Expected exactly one parsed query relation, got {}", queries.size()));
  auto const& query = queries.front();

  if (query.relation->type() != gqe::logical::relation::relation_type::write) {
    throw std::logic_error("execute_substrait_statement called with a non-write plan");
  }

  auto rule_config = make_default_optimizer_rules();
  gqe::optimizer::logical_optimizer optimizer(&rule_config, _catalog.get());
  auto optimized = optimizer.optimize(query.relation);

  gqe::physical_plan_builder plan_builder(_catalog.get(), &_config.opt_params);
  auto physical_plan = plan_builder.build(optimized.get());

  // The plan builder emits the write relation as the root of a write plan.
  auto const table_name =
    static_cast<gqe::physical::write_relation*>(physical_plan.get())->table_name();
  auto const revision_before_write = published_revision(*_catalog, table_name);

  // The catalog's shared-memory facade intentionally remains stale after a write. Publication
  // increments the storage revision, causing task managers to replace stale facades before the
  // next query.
  // TODO: Separate shared-storage ownership from process-local snapshots.

  execution_result result;
  bool is_dispatched = false;
  try {
    result = execute_on_task_managers(physical_plan, &is_dispatched);
    // Commit point. Appending the statistics has a strong exception guarantee, so the write either
    // commits here or leaves the catalog statistics unchanged.
    if (result.write_stats.has_value()) {
      _catalog->append_table_statistics(table_name, result.write_stats.value());
    }
  } catch (unconfirmed_completion_error const&) {
    // A rank may still publish row groups, so an unchanged revision proves nothing.
    mark_unusable_after_failed_write(*_catalog, table_name);
    throw;
  } catch (...) {
    // A plan that reached no task manager published nothing, whatever the storage kind. Otherwise,
    // an unchanged revision proves that the write published nothing. Either way the table stays
    // usable.
    bool is_unpublished = !is_dispatched;
    if (!is_unpublished && revision_before_write.has_value()) {
      try {
        is_unpublished = published_revision(*_catalog, table_name) == revision_before_write;
      } catch (...) {
        GQE_LOG_ERROR("Cannot read the storage revision of table '{}'", table_name);
      }
    }
    if (!is_unpublished) { mark_unusable_after_failed_write(*_catalog, table_name); }
    throw;
  }

  // Serialization runs after the commit and is not part of the statement's outcome. A failure
  // may leave the previous snapshot loadable, which can hold fewer rows than the table, see
  // catalog::serialize_table.
  // File-backed tables already own their files on disk and have no snapshot representation.
  if (_config.opt_params.serialize_tables_to_disk &&
      !gqe::storage_kind::is_file_backed(_catalog->storage_kind(table_name))) {
    try {
      _catalog->serialize_table(table_name,
                                gqe::utility::serialized_data_root(_config.opt_params, table_name));
      GQE_LOG_INFO("Serialized table '{}' to snapshot after write", table_name);
    } catch (std::exception const& error) {
      // An unrecoverable shared-memory error sets the poisoned flag, and the check below tears the
      // node down.
      gqe::record_if_shared_memory_unrecoverable(std::current_exception(),
                                                 "post-commit snapshot serialization");
      GQE_LOG_ERROR(
        "The write to table '{}' committed, but refreshing its snapshot failed: {}. A previous "
        "snapshot may remain loadable.",
        table_name,
        error.what());
    }
  }

  // Shared-memory poisoning takes precedence over the committed write, including poisoning during
  // serialization. The error says that the write had committed before the node began shutting down.
  _teardown.throw_if_poisoned(
    std::format("node manager, after the write to table '{}' committed", table_name));
  return result.write_stats;
}

std::shared_ptr<arrow::RecordBatch> node_manager::execute_physical_plan_query(void const* data,
                                                                              std::size_t size)
{
  std::lock_guard guard(_execute_mutex);
  _teardown.throw_if_begun();

  proto::PhysicalRelation pr;
  if (!pr.ParseFromArray(data, static_cast<int>(size))) {
    throw std::runtime_error("Failed to parse PhysicalRelation from binary data");
  }

  auto physical_plan = gqe::rpc::deserialize_physical_plan(pr).root();
  if (physical_plan->type() == gqe::physical::relation::relation_type::write) {
    throw std::logic_error("execute_physical_plan_query called with a write plan");
  }

  // Re-materialize statistics tables on demand if this plan reads one (no-op otherwise).
  table_name_collector collector;
  physical_plan->accept(collector);
  refresh_stats_tables_if_stale(collector.table_names);

  return execute_on_task_managers(physical_plan).batch;
}

}  // namespace gqe::node_manager
