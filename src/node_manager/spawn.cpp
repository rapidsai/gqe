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

#include <gqe/node_manager/spawn.hpp>

#include <gqe/utility/logger.hpp>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>
#include <thread>

extern char** environ;

namespace gqe::node_manager {

// ============================================================================
// process_group
// ============================================================================

process_group::~process_group()
{
  if (!_pids.empty()) { terminate_all(); }
}

void process_group::reserve(std::size_t n) { _pids.reserve(n); }

void process_group::add(pid_t pid)
{
  if (pid <= 0) { throw std::invalid_argument(std::format("Invalid child PID: {}", pid)); }
  _pids.push_back(pid);
}

std::size_t process_group::size() const { return _pids.size(); }

pid_t process_group::pid(std::size_t rank) const { return _pids[rank]; }

std::optional<exit_status> process_group::try_wait_any()
{
  for (std::size_t i = 0; i < _pids.size(); ++i) {
    if (_pids[i] <= 0) { continue; }
    int status;
    pid_t result = waitpid(_pids[i], &status, WNOHANG);
    if (result > 0) {
      if (WIFEXITED(status)) {
        auto pid = _pids[i];
        _pids[i] = -1;
        return exit_status{
          .rank = i, .pid = pid, .kind = exit_kind::exited, .code = WEXITSTATUS(status)};
      } else if (WIFSIGNALED(status)) {
        auto pid = _pids[i];
        _pids[i] = -1;
        return exit_status{
          .rank = i, .pid = pid, .kind = exit_kind::signaled, .code = WTERMSIG(status)};
      }
      // Stopped or continued — not a terminal state, skip.
    }
  }
  return std::nullopt;
}

void process_group::signal_all(int sig)
{
  for (auto pid : _pids) {
    if (pid > 0) { kill(pid, sig); }
  }
}

void process_group::wait_all()
{
  for (auto pid : _pids) {
    if (pid > 0) {
      int status;
      waitpid(pid, &status, 0);
    }
  }
  _pids.clear();
}

void process_group::terminate_all(std::chrono::seconds timeout)
{
  signal_all(SIGTERM);

  auto deadline = std::chrono::steady_clock::now() + timeout;
  for (auto pid : _pids) {
    if (pid <= 0) { continue; }
    int status;
    bool exited = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (waitpid(pid, &status, WNOHANG) > 0) {
        exited = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!exited) {
      GQE_LOG_WARN("Child process pid {} did not exit in time, sending SIGKILL", pid);
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
    }
  }
  _pids.clear();
}

// ============================================================================
// spawn_task_managers
// ============================================================================

spawn_result spawn_task_managers(std::string_view binary, int base_port, int num_ranks)
{
  spawn_result result;
  result.processes.reserve(num_ranks);
  result.channels.reserve(num_ranks);

  // Fork each task manager process.
  for (int rank = 0; rank < num_ranks; ++rank) {
    int grpc_port = base_port + rank;
    auto port_str = std::to_string(grpc_port);

    std::string binary_str{binary};
    char* argv[] = {binary_str.data(), const_cast<char*>("--grpc-port"), port_str.data(), nullptr};

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);

    pid_t pid;
    int err = posix_spawn(&pid, binary_str.c_str(), nullptr, &attr, argv, environ);
    posix_spawnattr_destroy(&attr);

    if (err != 0) {
      throw std::runtime_error(
        std::format("posix_spawn failed for rank {}: {}", rank, strerror(err)));
    }

    GQE_LOG_INFO("Launched task manager rank {} (pid {}) on port {}", rank, pid, grpc_port);
    result.processes.add(pid);
    result.channels.push_back(grpc::CreateChannel(std::format("localhost:{}", grpc_port),
                                                  grpc::InsecureChannelCredentials()));
  }

  // Wait for each task manager's gRPC server to become reachable.
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto stub    = proto::TaskManagerService::NewStub(result.channels[rank]);
    bool healthy = false;
    for (int attempt = 0; attempt < 300 && !healthy; ++attempt) {
      grpc::ClientContext ctx;
      ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
      proto::HealthCheckRequest req;
      proto::HealthCheckResponse resp;
      auto status = stub->HealthCheck(&ctx, req, &resp);
      if (status.ok()) {
        healthy = true;
        GQE_LOG_INFO("Task manager rank {} is reachable", rank);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    if (!healthy) {
      throw std::runtime_error(std::format("Task manager rank {} failed to become healthy", rank));
    }
  }

  return result;
}

}  // namespace gqe::node_manager
