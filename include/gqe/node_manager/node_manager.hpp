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

#include <gqe/catalog.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/node_manager/context.hpp>
#include <gqe/node_manager/exit_code.hpp>
#include <gqe/node_manager/fault_boundary.hpp>
#include <gqe/node_manager/spawn.hpp>
#include <gqe/node_manager/task_manager_group.hpp>
#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/types.hpp>

#include <grpcpp/channel.h>

#include <arrow/record_batch.h>
#include <arrow/status.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace grpc {
class ClientContext;
class Server;
}  // namespace grpc

namespace gqe::physical {
class relation;
}  // namespace gqe::physical

namespace gqe::node_manager {

/// Default host the Flight SQL endpoint binds to. Loopback by default so the
/// server is not network-exposed unless an operator opts in via --address.
/// Literal "127.0.0.1", not "localhost", to stay IPv4-only and avoid IPv6
/// (::1) resolution.
inline constexpr char default_listen_address[] = "127.0.0.1";

/// Common format for node-manager failure diagnostics.
inline constexpr char failure_log_format[] = "Node manager failed: {}";

/** @brief Configuration for the node manager Flight SQL server. */
struct configuration {
  std::string listen_address = default_listen_address;  ///< Network address to listen on.
  int port                   = 50051;                   ///< Flight SQL server port.
  int num_gpus               = 1;   ///< Number of GPUs / task managers to spawn.
  std::string task_manager_binary;  ///< Path to the gqe_task_manager executable.
  int task_manager_base_port = 0;   ///< Base gRPC port for task managers (0 = port + 1000).
  gqe::storage_kind::type storage =
    gqe::storage_kind::shared_segment_memory{};  ///< Storage backend.
  gqe::segment_layout_policy_type shmem_policy =
    gqe::default_shmem_policy;              ///< Arrangement of the shared memory segment.
  std::chrono::seconds query_timeout{600};  ///< gRPC deadline for ExecutePlan.
  gqe::optimization_parameters opt_params =
    gqe::make_optimization_parameters();  ///< Optimizer and execution parameters.
};

class service;
class log_drain;
class log_stream_service;

/**
 * @brief Orchestrates query execution across task managers.
 *
 * The node manager spawns task manager subprocesses (one per GPU), bootstraps
 * NVSHMEM via gRPC, and distributes query plans to task managers for execution.
 * Flight SQL client interaction is handled by @ref service.
 *
 * Once the shared memory is unrecoverable, the node manager tears the node down and exits with
 * @ref exit_code::shared_memory_unrecoverable. If the task managers cannot be started, also at
 * startup, after three attempts, it tears the node down too and exits with
 * @ref exit_code::task_managers_unavailable. The fault boundary (see
 * gqe/node_manager/fault_boundary.hpp) detects the failure and holds the teardown state, and this
 * class stops the server and the task managers.
 *
 * Queries and statements run one at a time. One that waits for a restart of the task managers
 * therefore also holds back the ones behind it, including `SET` and the reading of session options,
 * until the replacement is ready or the node tears down.
 */
class node_manager {
 public:
  explicit node_manager(configuration config, std::shared_ptr<log_drain> drain);
  ~node_manager();

  /**
   * @brief Start the Flight SQL server and launch task manager subprocesses.
   *
   * Standard exceptions and Arrow status errors from startup and serving are logged and
   * classified here instead of being propagated to the caller.
   *
   * @return The process exit status. Recorded shared-memory teardown takes precedence over
   * an ordinary failure. When both occur, both diagnostics are logged.
   */
  [[nodiscard]] exit_code::value serve();

  /// Shutdown: stop the Flight SQL server (see @ref stop_flight_server), terminate task managers.
  void shutdown();

 private:
  /// Run server startup and serving, preserving Arrow status returns.
  [[nodiscard]] arrow::Status serve_impl();

  /// Result of executing a plan on task managers.
  struct execution_result {
    std::shared_ptr<arrow::RecordBatch> batch;         ///< Query result (null for writes).
    std::optional<gqe::table_statistics> write_stats;  ///< Write statistics (absent for queries).
  };

  void cancel_in_flight_rpcs();

  /**
   * @brief Register @p ctx so that @ref cancel_in_flight_rpcs can cancel its RPC, unless the Flight
   * SQL server is already stopping or node teardown has begun.
   *
   * @ref stop_flight_server sets `_is_server_stopping` and cancels the registered RPCs under
   * `_in_flight_mutex`, and this function checks the flag under the same mutex. An RPC is therefore
   * either registered before the cancellation, and cancelled by it, or registered after it, and
   * cancelled here before it is dispatched. A cancelled context makes its RPC fail as soon as it
   * starts, so a handler that reaches this point after the server began stopping cannot hold
   * `Shutdown()` for the RPC deadline.
   *
   * Teardown can begin before the monitor thread stops the server, for example while it brings up
   * a group, so this checks for it too: an RPC is not dispatched once teardown has begun.
   *
   * @param ctx The context of an RPC that is about to be dispatched. It must stay alive until the
   * caller has unregistered it.
   * @param rank The rank the RPC is for. Only the test-only pause hook uses it.
   */
  void register_in_flight_rpc(grpc::ClientContext& ctx, int rank);

  /**
   * @brief Cancel in-flight task-manager RPCs, then shut down the Flight SQL server if it has been
   * initialized.
   *
   * `Shutdown()` waits for every running Flight SQL handler to return. A handler can be waiting for
   * a task manager until the RPC deadline, so cancelling the RPCs first lets it return promptly.
   * Both @ref shutdown and node teardown stop the server this way.
   *
   * Holds @c _server_mutex, so it never runs while @ref serve is initializing the server. Arrow's
   * `Init()` publishes the transport before initializing it, so a concurrent `Shutdown()` could
   * use a transport that is not ready.
   *
   * @pre @c _service is not null.
   * @return The status of `Shutdown()`, or OK if the server is not initialized yet.
   */
  [[nodiscard]] arrow::Status stop_flight_server();

  /**
   * @brief Spawn a new group of task managers and bootstrap NVSHMEM on it.
   *
   * Forks the task manager processes, waits for their gRPC servers to become reachable, then calls
   * @ref gqe::node_manager::bootstrap_nvshmem to initialise the NVSHMEM collective. The group is
   * `ready` when this returns, and is not installed. If any step fails, the processes of the new
   * group are terminated before the exception propagates.
   *
   * Holds no node-manager lock, because a bootstrap can take minutes and dispatch must not wait
   * for it.
   *
   * @return The new, `ready` group.
   */
  [[nodiscard]] std::shared_ptr<task_manager_group> start_group();

  /**
   * @brief Start a group, retrying a failed attempt with backoff.
   *
   * Makes up to three attempts, one second apart and then two. In-process retries cover transient
   * failures cheaply, and the loaded tables survive them, because the node manager owns the
   * shared memory. If every attempt fails, node teardown begins with
   * @ref teardown_reason::task_managers_unavailable, because a node manager without task managers
   * serves nothing and staying up would hide the failure from a harness that restarts a dead
   * server.
   *
   * Does not start a group if teardown has begun, including for poisoned shared memory, or if
   * @p token is stopped.
   *
   * @param token Interrupts the wait between attempts.
   * @return The new group, or nullptr if none was started.
   */
  [[nodiscard]] std::shared_ptr<task_manager_group> bring_up_group(std::stop_token const& token);

  /**
   * @brief Make @p group the one that serves operations, and wake the operations waiting for it.
   */
  void install_group(std::shared_ptr<task_manager_group> group);

  /**
   * @brief Fail @p group after a rank-fatal error on it, then report the error.
   *
   * A rank-fatal error can hide poisoning, so the poisoned flag in the segment is checked first.
   * If it is set, node teardown begins instead, and the group is not failed.
   *
   * Otherwise, this marks @p group `failed`, and the monitor thread replaces it. It is the group
   * that the operation ran on, which is not necessarily the installed one: the monitor thread may
   * have replaced it meanwhile, and the replacement must not be failed for an error on its
   * predecessor. The caller holds @c _execute_mutex, so replacing the group here would make other
   * operations wait for the whole restart. They wait for the replacement in
   * @ref acquire_ready_group instead, without holding the mutex that this thread needs.
   *
   * @param group The group the operation ran on.
   * @param message The rank-fatal error.
   * @throws shared_memory_unrecoverable_error If the shared memory is poisoned.
   * @throws std::runtime_error With @p message otherwise.
   */
  [[noreturn]] void restart_after_rank_fatal(task_manager_group& group, std::string message);

  /**
   * @brief Terminate @p group because an operation on it did not finish on every rank, then throw.
   *
   * An operation has finished on a rank only when that rank has sent a response. A failed RPC on
   * the node-manager side is not evidence that the rank stopped: after a deadline expires, the
   * rank keeps executing. This covers an expired deadline, a transport failure, a stall, and
   * the node manager's own cancellation.
   *
   * The group is killed before the caller releases `_execute_mutex`, so that the next operation
   * never meets a rank that still runs this one, and the caller can recover the operation's table
   * without a rank appending to it. The ranks get no grace period, because a rank that is stalled
   * or still running the operation would use it to finish the operation. The monitor thread then
   * starts the replacement, and operations wait for it.
   *
   * @param group The group the operation ran on.
   * @param error The reason the operation did not finish.
   * @throws task_manager_group_failed_error Always. Flight SQL clients get it as `Unavailable`.
   */
  [[noreturn]] void fail_group_unconfirmed(task_manager_group& group, std::string const& error);

  /**
   * @brief The monitor thread's loop: keep a ready task manager group, and tear the node down when
   * that is impossible.
   *
   * Runs until @p token requests a stop or node teardown begins. Each pass first checks for
   * poisoning that any process recorded in the shared-memory segment since the last pass, and for
   * teardown that began elsewhere, including because no group could be started. If teardown has
   * begun, it calls @ref stop_for_teardown and returns. Otherwise it calls
   * @ref supervise_task_managers, which replaces a failed group. An exception from that call is
   * logged and the pass is retried, because an exception that escaped this thread would call
   * `std::terminate`. Between passes it sleeps for one second, in a wait that teardown and a stop
   * request both interrupt.
   *
   * @param token Requests the loop to stop.
   */
  void monitor_task_managers(std::stop_token token);

  /**
   * @brief One monitor pass: replace the group if it has failed or a rank has exited.
   *
   * @param token Interrupts the wait between start attempts.
   */
  void supervise_task_managers(std::stop_token const& token);

  /**
   * @brief Stop the Flight SQL server for node teardown, on the monitor thread.
   *
   * Runs on the monitor thread rather than on the handler that began teardown, because
   * `Shutdown()` waits for every in-progress handler to return. If stopping fails, `serve()` would
   * never return, so this terminates the task managers, removes the shared memory, and exits the
   * process with the teardown's exit status.
   */
  void stop_for_teardown();

  /**
   * @brief Return the group that serves operations, waiting for one if there is none.
   *
   * Waits while the group is being replaced, so that an operation that arrives during a restart
   * runs once the new group is `ready` instead of failing. The wait is bounded by the start
   * attempts: if they are exhausted, node teardown begins and this throws its error. Does not wait
   * while holding any lock that the monitor thread needs.
   *
   * The caller keeps the returned group for the whole operation. It stays valid after it fails,
   * and a replacement never changes it.
   *
   * @throws task_managers_unavailable_error or shared_memory_unrecoverable_error If node teardown
   * has begun.
   * @throws std::runtime_error If the Flight SQL server is stopping.
   */
  [[nodiscard]] std::shared_ptr<task_manager_group> acquire_ready_group();

  /**
   * @brief Execute a Substrait SELECT plan.
   *
   * Parses, optimises, and executes the plan on task managers.
   *
   * @param data Raw bytes of the serialised Substrait plan.
   * @param size Size of @p data in bytes.
   * @return The query result as an Arrow RecordBatch, or nullptr if the plan
   *         produced no output.
   * @throws std::logic_error if the plan is a write operation.
   */
  [[nodiscard]] std::shared_ptr<arrow::RecordBatch> execute_substrait_query(void const* data,
                                                                            std::size_t size);

  /**
   * @brief Execute a Substrait write/DDL plan.
   *
   * Parses, optimises, and executes the plan on task managers. Updates the
   * catalog statistics if write statistics are produced.
   *
   * # Commit Point
   * A write commits when execution has succeeded on every rank and its statistics
   * have been appended to the catalog, which has a strong exception guarantee. A
   * write that fails before its commit leaves the catalog statistics unchanged.
   * It flags the table as unusable (see @ref gqe::catalog::mark_unusable) unless
   * the plan was never dispatched to a task manager, or the table's storage
   * revision shows that nothing was published and every rank confirmed that it
   * finished.
   *
   * If @c serialize_tables_to_disk is set, the table's snapshot is serialized
   * after the commit. A serialization failure is logged and does not fail the
   * statement, since the write has committed. A previous snapshot may remain
   * loadable, without the rows of this write (see
   * @ref gqe::catalog::serialize_table).
   *
   * @param data Raw bytes of the serialised Substrait plan.
   * @param size Size of @p data in bytes.
   * @return Write statistics if the plan produced them, or std::nullopt for
   *         DDL operations that don't report row counts.
   * @throws std::logic_error if the plan is not a write operation.
   */
  [[nodiscard]] std::optional<gqe::table_statistics> execute_substrait_statement(void const* data,
                                                                                 std::size_t size);

  /**
   * @brief Execute a serialized `proto::PhysicalRelation` SELECT plan.
   *
   * Deserialises the physical relation tree and dispatches it directly to
   * task managers, bypassing Substrait parsing and logical optimisation.
   *
   * @param data Raw bytes of the serialised `proto::PhysicalRelation`.
   * @param size Size of @p data in bytes.
   * @return The query result as an Arrow RecordBatch, or nullptr if the plan
   *         produced no output.
   * @throws std::runtime_error if the bytes do not parse as a
   *         `proto::PhysicalRelation`.
   * @throws std::logic_error if the plan is a write operation.
   */
  [[nodiscard]] std::shared_ptr<arrow::RecordBatch> execute_physical_plan_query(void const* data,
                                                                                std::size_t size);

  /**
   * @brief Dispatch a physical plan to all task managers and collect results.
   *
   * @par Task manager error contract
   * Each task manager returns a success, a non-fatal error, a rank-fatal error,
   * or a node-fatal error.
   * - Success or non-fatal error: the response is collected and reading continues
   *   with the next task manager. Results are returned only if every task manager
   *   succeeded, since the plan runs across all of them; otherwise the first
   *   non-fatal error is reported.
   * - Rank-fatal error (see @ref rank_outcome::rank_fatal): the remaining task
   *   managers are cancelled, all task managers are restarted, and the first
   *   rank-fatal error is reported.
   * - Node-fatal error (see @ref rank_outcome::node_fatal): the remaining task
   *   managers are cancelled and node teardown begins (see
   *   @ref teardown_state::begin). It takes precedence over rank-fatal errors,
   *   so the remaining responses are still read to find one, and no restart
   *   happens. A cancelled rank cannot report it, so before any restart the
   *   poisoned flag in the segment is checked as well.
   * - A task manager that hangs is bounded by the RPC deadline.
   * - A rank confirms that it finished by returning a response. If an error is
   *   reported while some rank failed in transport (for example, an expired
   *   deadline) and no restart terminated the ranks, completion is unconfirmed,
   *   and the rank may still be executing the plan.
   *
   * The plan is refused before dispatch if it references a table that a failed
   * write left unusable.
   *
   * @param physical_plan The physical plan to execute.
   * @param[out] is_dispatched If not null, set to true once the plan may have
   *        reached a task manager. It stays false if the call fails before the
   *        first dispatch, in which case no task manager executed the plan.
   * @return Combined query results and/or write statistics from all ranks.
   */
  [[nodiscard]] execution_result execute_on_task_managers(
    std::shared_ptr<gqe::physical::relation> physical_plan, bool* is_dispatched = nullptr);

  /**
   * @brief Re-materialize the statistics tables on the task managers if they are stale.
   *
   * If @p referenced_tables includes a statistics table and the catalog's @ref
   * gqe::catalog::stats_version has advanced since the last materialization, projects the current
   * statistics and dispatches a RefreshStatsTables RPC to all task managers (rank 0 writes the
   * shared-memory copy). The node waits for all ranks to finish, which barriers the write before
   * the caller dispatches the dependent read. A no-op when the tables are already up to date.
   *
   * Errors are handled as in @ref execute_on_task_managers: a rank-fatal error restarts every task
   * manager, and a node-fatal error begins node teardown.
   *
   * @pre Caller must hold @c _execute_mutex.
   */
  void refresh_stats_tables_if_stale(std::unordered_set<std::string> const& referenced_tables);

  configuration _config;

  /// Whether node teardown has begun. Must outlive @c _service, whose middleware refers to it.
  teardown_state _teardown;

  std::unique_ptr<context> _ctx;           // Must outlive _catalog.
  std::unique_ptr<gqe::catalog> _catalog;  // Must outlive _service.
  std::unique_ptr<service> _service;

  /// Serializes initializing the Flight SQL server with stopping it, and guards
  /// @c _server_initialized.
  std::mutex _server_mutex;

  /// Whether @c _service->Init() has succeeded. Guarded by @c _server_mutex.
  bool _server_initialized = false;

  std::shared_ptr<log_drain> _log_drain;
  std::unique_ptr<log_stream_service> _log_stream_service;
  std::unique_ptr<grpc::Server> _log_server;
  std::string _log_address;
  std::atomic<bool> _shutdown_started{false};

  /**
   * @brief Protects @c _group.
   *
   * Held only to read or replace the pointer, never across a start attempt or a termination. A
   * start attempt can take minutes and a termination waits for processes, so holding it would make
   * dispatch wait for them.
   *
   * Acquired by @ref install_group, @ref supervise_task_managers, @ref acquire_ready_group,
   * @ref restart_after_rank_fatal and @ref shutdown.
   */
  std::mutex _task_manager_mutex;

  /**
   * @brief The group that serves operations, or the one that failed and is being replaced.
   *
   * Null until the first group is installed. A failed group stays here until its replacement is
   * installed, so that dispatch waits for the replacement. Guarded by @c _task_manager_mutex.
   */
  std::shared_ptr<task_manager_group> _group;

  /// Signalled when @c _group changes. Waiters also poll, because teardown and server stop do not
  /// signal it.
  std::condition_variable _group_cv;

  /// Generation of the next group. Used by one thread at a time: the thread that serves at
  /// startup, then the monitor thread, which starts after it.
  std::uint64_t _next_generation = 0;

  /// Restarts crashed task managers, and tears the node down once teardown begins. Sleeps through
  /// @ref teardown_state::wait, which teardown and a stop request both interrupt.
  std::jthread _monitor_thread;

  /**
   * @brief Protects @c _in_flight_rpcs.
   *
   * Flight SQL handler threads register their @c grpc::ClientContext pointers
   * before dispatching RPCs and unregister them afterwards. On task manager
   * crash or shutdown, @c cancel_in_flight_rpcs() iterates the vector to
   * cancel outstanding RPCs. Without this mutex, a crash during registration
   * could corrupt the vector or miss an in-flight RPC.
   *
   * Acquired by @ref register_in_flight_rpc and by the code that unregisters RPCs,
   * and by @c cancel_in_flight_rpcs() on the monitor thread and during
   * shutdown.
   */
  std::mutex _in_flight_mutex;
  std::vector<grpc::ClientContext*> _in_flight_rpcs;  ///< Guarded by @c _in_flight_mutex.
  /// Set once by @ref stop_flight_server, under @c _in_flight_mutex so that it is ordered with the
  /// registration of RPCs. Atomic so that @ref acquire_ready_group can also read it.
  std::atomic<bool> _is_server_stopping{false};

  /**
   * @brief Serialises plan execution. @c gqe::catalog is unsynchronised; any
   * concurrent path that reads or mutates it would race on @c _table_entries.
   *
   * The optimizer and plan builder use shared catalog state (@c _table_entries)
   * that is not thread-safe, and @c _config.opt_params must not be modified
   * while a query is being planned or dispatched. This mutex ensures only one
   * query, statement, or SET/SHOW handler runs at a time.
   *
   * Acquired by @c execute_substrait_query(), @c execute_substrait_statement(),
   * @c execute_physical_plan_query(), and the set/get session option callbacks
   * in @c serve().
   */
  std::mutex _execute_mutex;

  /**
   * @brief The @ref gqe::catalog::stats_version captured at the last statistics-table
   * materialization, or nullopt if never materialized. Guarded by @c _execute_mutex. A read of a
   * statistics table re-materializes only when this differs from the current version.
   */
  std::optional<std::uint64_t> _materialized_stats_version;
};

}  // namespace gqe::node_manager
