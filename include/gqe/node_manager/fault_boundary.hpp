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

/**
 * @file
 * @brief The fault boundary: how operations report and react to failures that outlive them.
 *
 * # Design
 *
 * Some failures leave state that later operations must not use. A rank whose CUDA context has a
 * sticky error cannot serve further queries (*rank-fatal*). A shared-memory segment with an
 * unrecoverable lock cannot be trusted by any process on the node (*node-fatal*, see
 * @ref gqe::shared_memory_unrecoverable_error). Every operation that crosses a process boundary
 * must detect these failures, report them to the other side, and stop new operations once the node
 * is being torn down.
 *
 * This module is the only place that does so, so that a new operation gets the behavior by using
 * the module rather than by repeating checks. It has one component per boundary.
 *
 * - @ref respond_to_operation runs a task-manager operation and builds its response. It reports
 *   node-fatal and rank-fatal errors in the response's `node_fatal` and `rank_fatal` fields.
 * - @ref task_manager_client sends operations from the node manager to a task manager. It is the
 *   only code that reads those fields, and classifies each response as a @ref rank_outcome.
 * - @ref teardown_state holds whether node teardown has begun. Its @ref teardown_state::guard runs
 *   the node manager's Flight SQL operations, and @ref teardown_gate rejects every Flight SQL call
 *   once teardown has begun.
 *
 * Code outside this module never reads the poisoned flag or the response fields directly. A
 * pre-commit hook enforces this for the node manager and the task manager.
 *
 * Poisoning can also happen outside any operation, for example in a destructor that frees shared
 * memory. The node manager's monitor thread therefore polls the flag through
 * @ref teardown_state::begin_if_poisoned. The components above detect poisoning promptly, and the
 * monitor is the backstop.
 *
 * # Node teardown
 *
 * Once the shared memory is unrecoverable, the node does not try to keep serving the surviving
 * tables. The node also tears down when its task managers cannot be brought back, because a node
 * manager without task managers serves nothing, and staying up would hide the failure from a
 * harness that restarts a dead server. Both causes follow the same steps, and differ in the exit
 * status and in the exception that handlers get (see @ref teardown_reason). Teardown proceeds as
 * follows.
 *
 * 1. @ref teardown_state::begin records the first cause and wakes the monitor thread. From then on,
 *    @ref teardown_gate rejects every new Flight SQL call with the cause.
 * 2. The monitor thread stops the Flight SQL server, so the node manager's `serve()` returns. If
 *    teardown begins before the server is initialized, `serve()` finds it after initializing and
 *    does not serve. If stopping fails, the monitor thread terminates the task managers, removes
 *    the segment, and exits itself.
 * 3. The node manager terminates the task managers. The segment is unlinked when the context is
 *    destroyed, without traversing any storage that a poisoned lock protects.
 * 4. The process exits with the status of the first cause, see @ref teardown_state::exit_status.
 *
 * Recreating the segment in-process instead would reach the same state with more code to get
 * right, and the data must be reloaded either way.
 */

#pragma once

#include <gqe/node_manager/exit_code.hpp>
#include <gqe/storage/shared_memory_poison.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <arrow/flight/server_middleware.h>
#include <arrow/status.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace gqe::node_manager {

// ============================================================================
// Task manager side
// ============================================================================

namespace detail {

/**
 * @brief Return the node-fatal error message if the shared memory is poisoned, and log it.
 *
 * @param operation_error The operation's own error, kept in the message, or empty if it succeeded.
 * @param rank The task manager's rank, used in the log message.
 */
[[nodiscard]] std::optional<std::string> node_fatal_message(std::string_view operation_error,
                                                            int rank);

/**
 * @brief Record a failed operation's exception, and return the rank-fatal error message if the
 * CUDA context has a sticky error.
 *
 * Records the current exception as shared-memory poisoning if it is an unrecoverable lock error
 * (see @ref gqe::record_if_shared_memory_unrecoverable).
 *
 * @pre Called from a `catch` block.
 * @param operation_name Name of the operation, used in the log messages.
 * @param error The exception's message.
 * @param rank The task manager's rank, used in the log messages.
 */
[[nodiscard]] std::optional<std::string> record_operation_error(std::string_view operation_name,
                                                                std::string_view error,
                                                                int rank);

/**
 * @brief Turn @p response into a node-fatal error if the shared memory is poisoned.
 *
 * A node-fatal error takes precedence over a rank-fatal error and over the operation's own error,
 * which is kept in the message. A result computed from the segment is not trusted, so it is
 * cleared.
 *
 * @return Whether the response was turned into a node-fatal error.
 */
template <typename Response>
bool report_if_node_fatal(Response& response, int rank)
{
  auto message =
    node_fatal_message(response.success() ? std::string_view{} : response.error_message(), rank);
  if (!message) { return false; }
  response.set_success(false);
  response.set_node_fatal(true);
  response.set_error_message(std::move(*message));
  response.set_rank_fatal(false);
  if constexpr (requires { response.clear_arrow_ipc_result(); }) {
    response.clear_arrow_ipc_result();
  }
  if constexpr (requires { response.clear_write_statistics(); }) {
    response.clear_write_statistics();
  }
  return true;
}

}  // namespace detail

/**
 * @brief Run a task-manager operation and build its response.
 *
 * Every task-manager RPC that reads or writes the shared memory builds its response through this
 * function, which handles the response's `success`, `error_message`, `rank`, `rank_fatal`, and
 * `node_fatal` fields.
 *
 * - If the shared memory is already poisoned, the operation does not run, and the response is
 *   node-fatal.
 * - Otherwise @p operation runs, including deserializing its request, and fills the response's
 *   result fields. If it throws, the response carries the error, and is rank-fatal if the CUDA
 *   context has a sticky error. An unrecoverable lock error poisons the shared memory.
 * - Afterwards the shared memory is checked again, whether the operation succeeded or failed.
 *   Cleanup can poison it without throwing, so a success is not evidence that it is still usable.
 *
 * The task manager's gRPC handler returns OK after this, so the response always reaches the node
 * manager. A request rejected with a gRPC error status, for example because the task manager is
 * busy, carries no response, so the node manager checks the poisoned flag itself after such a call.
 *
 * @tparam Response `proto::ExecutePlanResponse` or `proto::RefreshStatsResponse`.
 * @tparam Operation Callable with signature `void()`.
 * @param response The response to build.
 * @param rank This task manager's rank.
 * @param operation_name Name of the operation, used in log messages.
 * @param operation The operation to run.
 */
template <typename Response, typename Operation>
void respond_to_operation(Response& response,
                          int rank,
                          std::string_view operation_name,
                          Operation&& operation)
{
  response.set_rank(rank);
  // Do not run an operation on a segment already known to be unrecoverable.
  if (detail::report_if_node_fatal(response, rank)) { return; }
  try {
    std::forward<Operation>(operation)();
    response.set_success(true);
  } catch (std::exception const& error) {
    auto rank_fatal_message = detail::record_operation_error(operation_name, error.what(), rank);
    response.set_success(false);
    response.set_error_message(error.what());
    if (rank_fatal_message) {
      response.set_error_message(std::move(*rank_fatal_message));
      response.set_rank_fatal(true);
    }
  }
  detail::report_if_node_fatal(response, rank);
}

// ============================================================================
// Node manager side: calls to task managers
// ============================================================================

/** @brief Classification of one task manager's answer to an operation. */
enum class rank_outcome {
  /// The operation succeeded.
  success,
  /// The operation failed, and the rank can serve further operations.
  error,
  /// The call failed in transport, for example because its deadline expired or it was cancelled.
  /// The rank did not confirm that it finished, so it may still be running the operation.
  transport_failure,
  /// The rank cannot serve further operations, so every task manager must be restarted.
  rank_fatal,
  /// The shared memory is unrecoverable, so the node must be torn down.
  node_fatal,
};

/** @brief One task manager's answer to an operation. */
template <typename Response>
struct rank_result {
  rank_outcome outcome;

  /// Description of the failure, naming the rank, or empty on success.
  std::string message;

  /// The response. Its result fields are meaningful only on success.
  Response response;

  /// Whether the rank confirmed that it finished, by returning a response.
  [[nodiscard]] bool is_confirmed() const noexcept
  {
    return outcome != rank_outcome::transport_failure;
  }
};

/**
 * @brief Sends operations to one task manager and classifies its answers.
 *
 * The only node-manager code that reads the `rank_fatal` and `node_fatal` fields of a response.
 * Callers act on the @ref rank_outcome, for example by restarting the task managers or by
 * beginning node teardown.
 *
 * ## Thread Safety
 * One instance serves one call at a time.
 */
class task_manager_client {
 public:
  /**
   * @param channel Channel to the task manager.
   * @param rank The task manager's rank, used in messages.
   */
  task_manager_client(std::shared_ptr<grpc::Channel> channel, int rank);

  /** @brief Execute a plan. */
  [[nodiscard]] rank_result<proto::ExecutePlanResponse> execute_plan(
    grpc::ClientContext& context, proto::ExecutePlanRequest const& request);

  /** @brief Refresh the statistics tables. */
  [[nodiscard]] rank_result<proto::RefreshStatsResponse> refresh_stats(
    grpc::ClientContext& context, proto::RefreshStatsRequest const& request);

 private:
  std::unique_ptr<proto::TaskManagerService::Stub> _stub;
  int _rank;
};

// ============================================================================
// Node manager side: node teardown
// ============================================================================

/**
 * @brief Why the node is tearing down.
 *
 * The reason decides the exit status (see @ref teardown_state::exit_status) and the type of the
 * exception that handlers inside the node manager get (see @ref teardown_state::throw_error). The
 * type does not cross the Flight SQL boundary: a client gets the error as `UnknownError` for every
 * reason, and tells the reasons apart only by the message.
 */
enum class teardown_reason {
  /// The shared memory can no longer be trusted. Reported as
  /// @ref gqe::shared_memory_unrecoverable_error.
  shared_memory_unrecoverable,
  /// The task managers failed and could not be restarted. Reported as
  /// @ref task_managers_unavailable_error.
  task_managers_unavailable,
};

/**
 * @brief Describe why the node is tearing down, as a clause that follows "because".
 *
 * For example "the shared memory is unrecoverable". Used in the node manager's log. The function
 * switches over every enumerator without a default, so that the compiler reports a reason that is
 * added without a description.
 *
 * @param reason The reason.
 * @return A static string.
 */
[[nodiscard]] std::string_view teardown_reason_description(teardown_reason reason) noexcept;

/**
 * @brief The error that clients get when the node is shutting down because its task managers could
 * not be brought back.
 */
class task_managers_unavailable_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Whether node teardown has begun, and why.
 *
 * Teardown begins at most once, and never ends. The first reason and cause win. See the file's
 * "Node teardown" section.
 *
 * ## Thread Safety
 * All member functions may be called concurrently.
 */
class teardown_state {
 public:
  /**
   * @brief Begin node teardown, unless it has begun already, and wake @ref wait.
   *
   * @param reason Why the node is tearing down. Ignored if teardown has begun already.
   * @param cause Description of the error, naming the process and operation that observed it.
   * Ignored if teardown has begun already.
   */
  void begin(teardown_reason reason, std::string cause);

  /**
   * @brief Begin node teardown as @ref begin does, then throw the error for the first cause.
   *
   * @throw gqe::shared_memory_unrecoverable_error or @ref task_managers_unavailable_error,
   * according to the first reason.
   */
  [[noreturn]] void begin_and_throw(teardown_reason reason, std::string cause);

  /**
   * @brief Begin node teardown if this process's poisoned flag is set.
   *
   * @param observer Who found the flag set, prefixed to the poisoning's cause.
   * @return Whether teardown has begun, for this or an earlier reason.
   */
  bool begin_if_poisoned(std::string_view observer);

  /** @brief Return whether node teardown has begun. */
  [[nodiscard]] bool has_begun() const noexcept;

  /** @brief Return the first cause of node teardown, or nullopt if it has not begun. */
  [[nodiscard]] std::optional<std::string> cause() const;

  /** @brief Return the first reason for node teardown, or nullopt if it has not begun. */
  [[nodiscard]] std::optional<teardown_reason> reason() const;

  /**
   * @brief Return the exit status of the node manager for the first reason.
   *
   * @pre Teardown has begun. Not checked: the result is then @ref exit_code::failure.
   */
  [[nodiscard]] exit_code::value exit_status() const;

  /**
   * @brief Return the message of the error to report to clients, or nullopt if teardown has not
   * begun.
   */
  [[nodiscard]] std::optional<std::runtime_error> error() const;

  /**
   * @brief Throw the error to report to clients for the first reason.
   *
   * @pre Teardown has begun. Not checked: otherwise this throws a `std::logic_error`.
   * @throw gqe::shared_memory_unrecoverable_error or @ref task_managers_unavailable_error,
   * according to the first reason.
   */
  [[noreturn]] void throw_error() const;

  /**
   * @brief Throw the teardown error if teardown has begun or this process's poisoned flag is set.
   *
   * Begins teardown first if the flag is set but teardown has not begun.
   *
   * @throw gqe::shared_memory_unrecoverable_error If the node can no longer serve operations
   * because of the shared memory, or @ref task_managers_unavailable_error if it can no longer
   * serve because of its task managers.
   */
  void throw_if_begun();

  /**
   * @brief Begin teardown and throw its error if this process's poisoned flag is set.
   *
   * For a point where poisoning must take precedence over an otherwise successful result.
   *
   * @param observer Who found the flag set, prefixed to the poisoning's cause.
   * @throw gqe::shared_memory_unrecoverable_error If the flag is set, unless teardown began earlier
   * for another reason, in which case that reason's error.
   */
  void throw_if_poisoned(std::string_view observer);

  /**
   * @brief Run one node-manager operation and check the shared memory afterwards.
   *
   * Fails immediately if teardown has begun. Otherwise runs @p operation. If this process's
   * poisoned flag is set afterwards, whether the operation succeeded or failed, teardown begins and
   * the operation fails with the teardown error. This covers poisoning in the node manager's own
   * process, such as during CREATE or DROP TABLE.
   *
   * @tparam Operation Callable with signature `R()`, where `R` is neither `void` nor a reference.
   * @param operation The operation to run.
   * @return The result of @p operation.
   */
  template <typename Operation>
  auto guard(Operation&& operation) -> decltype(operation());

  /**
   * @brief Sleep until teardown begins, @p token is stopped, or @p timeout elapses.
   */
  void wait(std::stop_token const& token, std::chrono::milliseconds timeout);

 private:
  [[nodiscard]] static std::string make_message(teardown_reason reason, std::string const& cause);

  mutable std::mutex _mutex;
  std::condition_variable_any _cv;

  /// Set once under @c _mutex together with @c _cause, and never cleared. Read without the mutex
  /// to fail new operations quickly.
  std::atomic<bool> _has_begun{false};
  teardown_reason _reason =
    teardown_reason::shared_memory_unrecoverable;  ///< Guarded by @c _mutex.
  std::string _cause;                              ///< Guarded by @c _mutex.
};

template <typename Operation>
auto teardown_state::guard(Operation&& operation) -> decltype(operation())
{
  static_assert(!std::is_void_v<decltype(operation())>, "The operation must return a value");
  // The result is held in a local variable, so a returned reference would dangle.
  static_assert(!std::is_reference_v<decltype(operation())>,
                "The operation must return a value, not a reference");
  constexpr std::string_view observer = "node manager";
  throw_if_begun();
  try {
    auto result = std::forward<Operation>(operation)();
    throw_if_poisoned(observer);
    return result;
  } catch (shared_memory_unrecoverable_error const&) {
    // Thrown just above, by begin() in a callee, or raised in this process by a shared-memory lock.
    // In the last case the poisoned flag is already set.
    if (!has_begun()) { throw_if_poisoned(observer); }
    throw;
  } catch (...) {
    record_if_shared_memory_unrecoverable(std::current_exception(), "node manager operation");
    throw_if_poisoned(observer);
    throw;
  }
}

/**
 * @brief Flight SQL middleware that rejects every call once node teardown has begun.
 *
 * Covers every Flight SQL endpoint, including ones that do not touch the shared memory, such as
 * getting and setting session options. Also begins teardown if this process's poisoned flag is
 * set. A call already running when teardown begins is not affected, and
 * @ref teardown_state::guard fails it if it touches the shared memory.
 */
class teardown_gate : public arrow::flight::ServerMiddlewareFactory {
 public:
  /** @param teardown The node's teardown state, which must outlive the server. */
  explicit teardown_gate(teardown_state& teardown);

  arrow::Status StartCall(arrow::flight::CallInfo const& info,
                          arrow::flight::ServerCallContext const& context,
                          std::shared_ptr<arrow::flight::ServerMiddleware>* middleware) override;

 private:
  teardown_state& _teardown;
};

}  // namespace gqe::node_manager
