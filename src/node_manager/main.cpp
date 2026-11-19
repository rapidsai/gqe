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

#include <gqe/node_manager/exit_code.hpp>
#include <gqe/node_manager/log_drain.hpp>
#include <gqe/node_manager/node_manager.hpp>
#include <gqe/storage/segment_layout.hpp>
#include <gqe/utility/logger.hpp>

#include <signal.h>
#include <spdlog/logger.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

namespace exit_code = gqe::node_manager::exit_code;

namespace {

// Pre-Arrow signal handler: clean up shared memory and the task_manager subtree
// via process-group broadcast (children sigwait on the same set), with a
// bounded reap window before a SIGKILL fallback. Path must match
// node_manager::shared_memory_name in context.hpp.
void early_signal_handler(int signum)
{
  unlink("/dev/shm/gqe_shared_memory");

  if (getpid() != getpgrp()) {
    // we don't lead our pgid; broadcasting would hit unrelated processes
    _exit(exit_code::failure);
  }

  struct sigaction ign{};
  ign.sa_handler = SIG_IGN;
  sigaction(signum, &ign, nullptr);  // don't kill ourselves with the broadcast
  kill(-getpgrp(), signum);

  // Bounded reap loop (1s total) — caps the worst case if a child is wedged
  // in uninterruptible kernel state.
  for (int i = 0; i < 100; ++i) {
    int status;
    pid_t r = waitpid(-1, &status, WNOHANG);
    if (r > 0) { continue; }
    if (r == -1) { break; }             // waitpid error (typically ECHILD: no children left)
    struct timespec ts{0, 10'000'000};  // 10ms
    nanosleep(&ts, nullptr);
  }

  kill(-getpgrp(), SIGKILL);
  _exit(exit_code::failure);
}

void print_usage(char const* program)
{
  std::cerr << "Usage: " << program << " [options]\n"
            << "  --address/-a <addr>            Listen address (default: "
            << gqe::node_manager::default_listen_address << ")\n"
            << "  --port/-p <port>               Listen port (default: 50051)\n"
            << "  --num-gpus <n>                 Number of GPUs / task managers (default: 1)\n"
            << "  --task-manager-binary <path>   Path to gqe_task_manager binary\n"
            << "  --task-manager-base-port <port> Base gRPC port for task managers (default: "
               "port+1000)\n"
            << "  --query-timeout <seconds>      gRPC deadline for ExecutePlan (default: 600)\n"
            << "  --shmem-policy <name>          Shared memory segment layout policy (default: "
            << gqe::to_string(gqe::default_shmem_policy) << ")\n"
            << "                                 Values: single_arena_unplaced, "
               "per_node_arenas_bound_overflow\n";
}

std::shared_ptr<gqe::node_manager::log_drain> initialize_logging()
{
  auto drain  = std::make_shared<gqe::node_manager::log_drain>();
  auto logger = std::make_shared<spdlog::logger>(
    "gqe-node-manager", std::make_shared<gqe::node_manager::drain_sink>(drain, "server"));
  gqe::utility::initialize_logger(std::move(logger));
  return drain;
}

}  // namespace

int main(int argc, char* argv[])
{
  std::shared_ptr<gqe::node_manager::log_drain> log_drain;
  try {
    log_drain = initialize_logging();
  } catch (std::exception const& e) {
    std::cerr << "Failed to initialize node-manager logging: " << e.what() << '\n';
    return exit_code::failure;
  }

  // Try to become our own pgid leader so spawned task_managers inherit our
  // pgid; enables early signal-driven cleanup until Arrow takes over inside
  // Serve().
  if (getpid() != getpgrp() && setpgid(0, 0) != 0) {
    int saved_errno = errno;
    GQE_LOG_WARN(
      "setpgid(0, 0) failed: {}; could not become process group leader. If {} is killed before "
      "bootstrap completes, child processes may zombify until container or system reboot.",
      std::strerror(saved_errno),
      program_invocation_short_name);
  }

  // Ensure CUDA enumerates devices by PCI bus ID, matching nvidia-smi ordering.
  // Without this, CUDA defaults to FASTEST_FIRST which can cause
  // CUDA_VISIBLE_DEVICES indices (set from nvidia-smi queries) to select the
  // wrong GPU. Set in both binaries: the node manager propagates to spawned task
  // managers via environ, but the task manager also sets it for standalone use.
  // Reference:
  // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);  // 0 = don't overwrite if already set

  // Handle SIGHUP, SIGINT, and SIGTERM before Arrow's Serve() installs its own handlers. Without
  // this, SIGHUP during NVSHMEM bootstrap kills the process without cleaning up shared memory or
  // task managers. Arrow takes over once Serve() is entered.
  //
  // SIGHUP respects nohup (SIG_IGN disposition) — don't install if already ignored.
  // If the node manager becomes a daemonic process (fork + setsid), this is unnecessary.
  {
    struct sigaction new_action{};
    new_action.sa_handler = early_signal_handler;
    sigemptyset(&new_action.sa_mask);

    sigaction(SIGINT, &new_action, nullptr);
    sigaction(SIGTERM, &new_action, nullptr);

    struct sigaction old_action{};
    sigaction(SIGHUP, nullptr, &old_action);
    if (old_action.sa_handler != SIG_IGN) { sigaction(SIGHUP, &new_action, nullptr); }
  }

  gqe::node_manager::configuration config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--address" || arg == "-a") && i + 1 < argc) {
      config.listen_address = argv[++i];
    } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
      config.port = std::stoi(argv[++i]);
    } else if (arg == "--num-gpus" && i + 1 < argc) {
      config.num_gpus = std::stoi(argv[++i]);
    } else if (arg == "--task-manager-binary" && i + 1 < argc) {
      config.task_manager_binary = argv[++i];
    } else if (arg == "--task-manager-base-port" && i + 1 < argc) {
      config.task_manager_base_port = std::stoi(argv[++i]);
    } else if (arg == "--query-timeout" && i + 1 < argc) {
      config.query_timeout = std::chrono::seconds(std::stoi(argv[++i]));
    } else if (arg == "--shmem-policy" && i + 1 < argc) {
      config.shmem_policy = gqe::from_string<gqe::segment_layout_policy_type>(argv[++i]);
    } else {
      print_usage(argv[0]);
      return exit_code::failure;
    }
  }

  if (config.task_manager_binary.empty()) {
    std::cerr << "Error: --task-manager-binary is required\n";
    print_usage(argv[0]);
    return exit_code::failure;
  }

  try {
    gqe::node_manager::node_manager nm(std::move(config), std::move(log_drain));
    return nm.serve();
  } catch (std::exception const& e) {
    GQE_LOG_ERROR(gqe::node_manager::failure_log_format, e.what());
    return exit_code::failure;
  }
}
