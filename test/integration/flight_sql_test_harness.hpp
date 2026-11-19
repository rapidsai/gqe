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

/**
 * @file Test harness for Flight SQL integration and TPC-H tests.
 *
 * Provides the server instance, global configuration, and main() that manages
 * the server lifecycle.  Test files include this header to access the globals.
 */

#pragma once

#include <gqe/executor/optimization_parameters.hpp>

#include <sys/types.h>

#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace gqe_test {

/**
 * @brief Node manager settings for one harness run.
 */
struct ServerConfiguration {
  std::string_view name;
  int num_gpus = 1;
  std::optional<gqe::optimization_parameters> optimization_parameters;
};

inline std::ostream& operator<<(std::ostream& os, ServerConfiguration const& config)
{
  return os << config.name;
}

inline const ServerConfiguration kLocalContiguousTableSingleGPU{
  .name = "LocalContiguousTableSingleGPU", .num_gpus = 1};

inline const ServerConfiguration kLocalContiguousTableMultiGPU{
  .name = "LocalContiguousTableMultiGPU", .num_gpus = 2};

inline const ServerConfiguration kSharedContiguousTableSingleGPU{
  .name                    = "SharedContiguousTableSingleGPU",
  .num_gpus                = 1,
  .optimization_parameters = std::optional<gqe::optimization_parameters>{[] {
    auto params                         = gqe::make_optimization_parameters();
    params.use_in_memory_table_multigpu = true;
    params.compression_format           = gqe::compression_format::none;
    params.secondary_compression_format = gqe::compression_format::none;
    return params;
  }()}};

inline const ServerConfiguration kSharedCompressedTableSingleGPU{
  .name                    = "SharedCompressedTableSingleGPU",
  .num_gpus                = 1,
  .optimization_parameters = std::optional<gqe::optimization_parameters>{[] {
    auto params                         = gqe::make_optimization_parameters();
    params.use_in_memory_table_multigpu = true;
    params.zone_map_partition_size      = 100000;
    params.compression_format           = gqe::compression_format::ans;
    params.secondary_compression_format = gqe::compression_format::none;
    return params;
  }()}};

/**
 * @brief Manages a GQE node manager process for integration testing.
 *
 * Forks a node manager, waits for it to start listening, and loads TPC-H data.
 */
class ServerInstance {
 public:
  /**
   * @brief Start the requested server config and load TPC-H data, reusing the running server when
   * it already matches.
   * @return true if the server is ready for the requested configuration.
   */
  bool start(ServerConfiguration const& config,
             const char* node_manager_bin,
             const char* task_manager_bin,
             const char* load_script,
             const char* data_path);

  /** @brief Stop the node manager process. */
  void stop();

  ~ServerInstance() { stop(); }

  bool ready() const { return ready_; }
  int port() const { return port_; }
  int num_gpus() const { return num_gpus_; }
  std::string const& load_error() const { return load_error_; }

 private:
  int num_gpus_     = 0;
  int port_         = 0;
  pid_t server_pid_ = -1;
  bool ready_       = false;
  std::string load_error_;
  std::string config_name_;
};

/**
 * @brief Global test configuration populated by main().
 */
struct TestConfig {
  const char* node_manager_bin   = nullptr;
  const char* task_manager_bin   = nullptr;
  const char* client_bin         = nullptr;
  const char* load_script        = nullptr;
  const char* run_script         = nullptr;
  const char* data_path          = nullptr;
  const char* queries_dir        = nullptr;
  const char* physical_plans_dir = nullptr;
  const char* ref_results        = nullptr;
  int host_gpu_count             = 0;
};

extern TestConfig g_config;
extern ServerInstance g_server;

}  // namespace gqe_test
