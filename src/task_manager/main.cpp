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
#include <gqe/utility/async_logging.hpp>
#include <gqe/utility/linux.hpp>
#include <gqe/utility/logger.hpp>

#include <signal.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

void print_usage(char const* program)
{
  std::cerr << "Usage: " << program
            << " --grpc-port <port> [--rank <rank> --log-address <address>]\n";
}

}  // namespace

int main(int argc, char* argv[])
{
  // Block shutdown signals before logging or worker threads start. task_manager::run()
  // consumes them with sigwait().
  sigset_t shutdown_mask;
  sigemptyset(&shutdown_mask);
  sigaddset(&shutdown_mask, SIGTERM);
  sigaddset(&shutdown_mask, SIGINT);
  pthread_sigmask(SIG_BLOCK, &shutdown_mask, nullptr);

  // Capture the process CPU mask before any thread is pinned to its GPU's CPUs. Otherwise, the
  // first capture can happen on a pinned thread, and nvCOMP CPU compression never gets more CPUs.
  gqe::utility::capture_process_affinity();

  // Ensure CUDA enumerates devices by PCI bus ID, matching nvidia-smi ordering.
  // Without this, CUDA defaults to FASTEST_FIRST which can cause
  // CUDA_VISIBLE_DEVICES indices (set from nvidia-smi queries) to select the
  // wrong GPU. Set in both binaries: the node manager propagates to spawned task
  // managers via environ, but the task manager also sets it for standalone use.
  // Reference:
  // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
  setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);  // 0 = don't overwrite if already set

  gqe::task_manager::configuration config{};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--grpc-port" && i + 1 < argc) {
      config.grpc_port = std::stoi(argv[++i]);
    } else if (arg == "--rank" && i + 1 < argc) {
      config.rank = std::stoi(argv[++i]);
    } else if (arg == "--log-address" && i + 1 < argc) {
      config.log_address = argv[++i];
    } else {
      print_usage(argv[0]);
      return 1;
    }
  }

  if (config.grpc_port == 0) {
    std::cerr << "Error: --grpc-port is required\n";
    print_usage(argv[0]);
    return 1;
  }
  if (config.log_address.empty() != (config.rank < 0)) {
    std::cerr << "Error: --rank and --log-address must be provided together\n";
    print_usage(argv[0]);
    return 1;
  }

  std::unique_ptr<gqe::utility::async_logging_runtime> logging;
  try {
    if (!config.log_address.empty()) {
      logging = std::make_unique<gqe::utility::async_logging_runtime>(
        gqe::utility::async_logging_options{.address        = config.log_address,
                                            .source_tag     = "gpu" + std::to_string(config.rank),
                                            .queue_capacity = 8192});
    }

    gqe::task_manager::task_manager tm(config);
    tm.run();
  } catch (std::exception const& e) {
    GQE_LOG_ERROR("Task manager fatal error: {}", e.what());
    if (logging) { logging->shutdown(); }
    return 1;
  }

  if (logging) { logging->shutdown(); }

  return 0;
}
