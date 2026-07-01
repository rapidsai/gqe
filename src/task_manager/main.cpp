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
#include <gqe/utility/logger.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(char const* program)
{
  std::cerr << "Usage: " << program << " --grpc-port <port>\n";
}

}  // namespace

int main(int argc, char* argv[])
{
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

  try {
    gqe::task_manager::task_manager tm(config);
    tm.run();
  } catch (std::exception const& e) {
    GQE_LOG_ERROR("Task manager fatal error: {}", e.what());
    return 1;
  }

  return 0;
}
