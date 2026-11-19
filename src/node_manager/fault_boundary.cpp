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

#include <gqe/node_manager/fault_boundary.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda_runtime_api.h>

#include <arrow/flight/types.h>

#include <format>
#include <tuple>

namespace gqe::node_manager {

// ============================================================================
// Task manager side
// ============================================================================

namespace detail {

std::optional<std::string> node_fatal_message(std::string_view operation_error, int rank)
{
  if (!is_shared_memory_poisoned()) { return std::nullopt; }
  auto message = std::format("Shared memory is unrecoverable: {}", shared_memory_poison_cause());
  if (!operation_error.empty()) {
    message += std::format(" (operation error: {})", operation_error);
  }
  GQE_LOG_CRITICAL("Task manager rank {}: reporting node-fatal error: {}", rank, message);
  return message;
}

std::optional<std::string> record_operation_error(std::string_view operation_name,
                                                  std::string_view error,
                                                  int rank)
{
  record_if_shared_memory_unrecoverable(std::current_exception(), operation_name);
  GQE_LOG_ERROR("Task manager rank {}: {} failed: {}", rank, operation_name, error);
  auto const sticky = utility::cuda_context_is_unrecoverable();
  if (!sticky) { return std::nullopt; }
  auto message =
    std::format("{}: {} {}", error, cudaGetErrorName(*sticky), cudaGetErrorString(*sticky));
  GQE_LOG_CRITICAL(
    "Task manager rank {}: CUDA context is unrecoverable (sticky error): {}", rank, message);
  return message;
}

}  // namespace detail

// ============================================================================
// Node manager side: calls to task managers
// ============================================================================

namespace {

/**
 * @brief Classify a task manager's answer to an operation.
 *
 * @param status Status of the call.
 * @param response The response, meaningful only if @p status is OK.
 * @param prefix Start of the failure message, naming the rank and the operation.
 */
template <typename Response>
rank_result<Response> classify(grpc::Status const& status,
                               Response response,
                               std::string_view prefix)
{
  if (!status.ok()) {
    return {rank_outcome::transport_failure,
            std::format("{}: {}", prefix, status.error_message()),
            std::move(response)};
  }
  if (response.success()) { return {rank_outcome::success, {}, std::move(response)}; }

  auto outcome = rank_outcome::error;
  if (response.node_fatal()) {
    outcome = rank_outcome::node_fatal;
  } else if (response.rank_fatal()) {
    outcome = rank_outcome::rank_fatal;
  }
  return {outcome, std::format("{}: {}", prefix, response.error_message()), std::move(response)};
}

}  // namespace

task_manager_client::task_manager_client(std::shared_ptr<grpc::Channel> channel, int rank)
  : _stub(proto::TaskManagerService::NewStub(std::move(channel))), _rank(rank)
{
}

rank_result<proto::ExecutePlanResponse> task_manager_client::execute_plan(
  grpc::ClientContext& context, proto::ExecutePlanRequest const& request)
{
  proto::ExecutePlanResponse response;
  auto const status = _stub->ExecutePlan(&context, request, &response);
  return classify(status, std::move(response), std::format("Task manager rank {} failed", _rank));
}

rank_result<proto::RefreshStatsResponse> task_manager_client::refresh_stats(
  grpc::ClientContext& context, proto::RefreshStatsRequest const& request)
{
  proto::RefreshStatsResponse response;
  auto const status = _stub->RefreshStatsTables(&context, request, &response);
  return classify(
    status, std::move(response), std::format("Task manager rank {} stats refresh failed", _rank));
}

// ============================================================================
// Node manager side: node teardown
// ============================================================================

std::string_view teardown_reason_description(teardown_reason reason) noexcept
{
  switch (reason) {
    case teardown_reason::shared_memory_unrecoverable: return "the shared memory is unrecoverable";
    case teardown_reason::task_managers_unavailable:
      return "the task managers could not be restarted";
  }
  return "of an unknown reason";
}

std::string teardown_state::make_message(teardown_reason reason, std::string const& cause)
{
  return std::format(
    "The node manager is shutting down because {}. Restart it and reload every table. Cause: {}",
    teardown_reason_description(reason),
    cause);
}

void teardown_state::begin(teardown_reason reason, std::string cause)
{
  {
    std::scoped_lock guard(_mutex);
    if (!_has_begun.load(std::memory_order_relaxed)) {
      _reason = reason;
      _cause  = std::move(cause);
      _has_begun.store(true, std::memory_order_release);
      GQE_LOG_CRITICAL("Node manager: beginning node teardown: {}", _cause);
    }
  }
  _cv.notify_all();
}

void teardown_state::begin_and_throw(teardown_reason reason, std::string cause)
{
  begin(reason, std::move(cause));
  throw_error();
}

bool teardown_state::begin_if_poisoned(std::string_view observer)
{
  if (!has_begun() && is_shared_memory_poisoned()) {
    begin(teardown_reason::shared_memory_unrecoverable,
          std::format("{}: {}", observer, shared_memory_poison_cause()));
  }
  return has_begun();
}

bool teardown_state::has_begun() const noexcept
{
  return _has_begun.load(std::memory_order_acquire);
}

std::optional<std::string> teardown_state::cause() const
{
  if (!has_begun()) { return std::nullopt; }
  std::scoped_lock guard(_mutex);
  return _cause;
}

std::optional<teardown_reason> teardown_state::reason() const
{
  if (!has_begun()) { return std::nullopt; }
  std::scoped_lock guard(_mutex);
  return _reason;
}

exit_code::value teardown_state::exit_status() const
{
  auto const first_reason = reason();
  if (!first_reason) { return exit_code::failure; }
  switch (*first_reason) {
    case teardown_reason::shared_memory_unrecoverable:
      return exit_code::shared_memory_unrecoverable;
    case teardown_reason::task_managers_unavailable: return exit_code::task_managers_unavailable;
  }
  return exit_code::failure;
}

std::optional<std::runtime_error> teardown_state::error() const
{
  if (!has_begun()) { return std::nullopt; }
  std::scoped_lock guard(_mutex);
  return std::runtime_error(make_message(_reason, _cause));
}

void teardown_state::throw_error() const
{
  if (!has_begun()) { throw std::logic_error("Node teardown has not begun"); }
  teardown_reason reason{};
  std::string message;
  {
    std::scoped_lock guard(_mutex);
    reason  = _reason;
    message = make_message(_reason, _cause);
  }
  switch (reason) {
    case teardown_reason::shared_memory_unrecoverable:
      throw shared_memory_unrecoverable_error(message);
    case teardown_reason::task_managers_unavailable: throw task_managers_unavailable_error(message);
  }
  throw std::logic_error("Unknown teardown reason");
}

void teardown_state::throw_if_begun()
{
  if (!begin_if_poisoned("node manager")) { return; }
  throw_error();
}

void teardown_state::throw_if_poisoned(std::string_view observer)
{
  if (!is_shared_memory_poisoned()) { return; }
  begin_and_throw(teardown_reason::shared_memory_unrecoverable,
                  std::format("{}: {}", observer, shared_memory_poison_cause()));
}

void teardown_state::wait(std::stop_token const& token, std::chrono::milliseconds timeout)
{
  std::unique_lock lock(_mutex);
  _cv.wait_for(lock, token, timeout, [this] { return has_begun(); });
}

teardown_gate::teardown_gate(teardown_state& teardown) : _teardown(teardown) {}

arrow::Status teardown_gate::StartCall(arrow::flight::CallInfo const& info,
                                       arrow::flight::ServerCallContext const& context,
                                       std::shared_ptr<arrow::flight::ServerMiddleware>* middleware)
{
  if (!_teardown.begin_if_poisoned("node manager")) { return arrow::Status::OK(); }
  // Whatever the reason, the node is exiting and does not come back by itself, so the status is not
  // `Unavailable`, which would invite the client to retry against it.
  return arrow::Status::UnknownError(_teardown.error()->what());
}

}  // namespace gqe::node_manager
