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

#include "flight_sql_test_harness.hpp"

#include "flight_sql_test_utils.hpp"

#include <gqe/rpc/serialization/optimization_parameters.hpp>

#include <gtest/gtest.h>

#include <arrow/flight/client.h>
#include <sys/wait.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>

namespace gqe_test {

TestConfig g_config;
ServerInstance g_server;

namespace {

arrow::Status set_session_options(int port, gqe::optimization_parameters const& parameters)
{
  arrow::flight::SetSessionOptionsRequest request;
  for (auto const& [name, value] :
       gqe::rpc::optimization_parameters_to_session_options(parameters)) {
    request.session_options.emplace(name, value);
  }

  auto location = arrow::flight::Location::ForGrpcTcp("127.0.0.1", port);
  if (!location.ok()) { return location.status(); }

  auto client = arrow::flight::FlightClient::Connect(*location);
  if (!client.ok()) { return client.status(); }

  auto result = (*client)->SetSessionOptions({}, request);
  if (!result.ok()) { return result.status(); }
  if (!result->errors.empty()) {
    return arrow::Status::Invalid("Failed to set session option: ", result->errors.begin()->first);
  }
  return arrow::Status::OK();
}

class HarnessEnvironment : public ::testing::Environment {
 public:
  void SetUp() override
  {
    auto& cfg = g_config;
    cfg.node_manager_bin =
      env_or("GQE_NODE_MANAGER", GQE_PROJECT_BINARY_DIR "/src/node_manager/gqe_node_manager");
    cfg.task_manager_bin =
      env_or("GQE_TASK_MANAGER", GQE_PROJECT_BINARY_DIR "/src/task_manager/gqe_task_manager");
    cfg.client_bin  = env_or("GQE_CLI", GQE_PROJECT_BINARY_DIR "/cargo-target/release/gqe-cli");
    cfg.load_script = env_or("GQE_LOAD_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/load_tpch.py");
    cfg.run_script  = env_or("GQE_RUN_SCRIPT", GQE_PROJECT_SOURCE_DIR "/scripts/run_tpch.py");

    // Export the resolved CLI path so child Python processes inherit it via
    // os.environ["GQE_CLI"].
    setenv("GQE_CLI", cfg.client_bin, /*overwrite=*/0);
    cfg.data_path          = std::getenv("TPCH_DATA_PATH");
    cfg.queries_dir        = std::getenv("TPCH_QUERIES");
    cfg.physical_plans_dir = std::getenv("TPCH_PHYSICAL_PLANS");
    cfg.ref_results        = std::getenv("TPCH_REF_RESULTS");
    cfg.host_gpu_count     = detect_gpu_count();
  }

  void TearDown() override { g_server.stop(); }
};

}  // namespace

bool ServerInstance::start(ServerConfiguration const& config,
                           const char* node_manager_bin,
                           const char* task_manager_bin,
                           const char* load_script,
                           const char* data_path)
{
  if (ready_ && config_name_ == config.name) { return true; }
  stop();

  num_gpus_ = config.num_gpus;
  port_     = find_free_port();

  // Remove any stale shared memory segment from a previous test run.
  std::remove("/dev/shm/gqe_shared_memory");

  server_pid_ = fork();
  if (server_pid_ == -1) { return false; }

  if (server_pid_ == 0) {
    setenv("GQE_MAX_QUERY_MEMORY", "1073741824", /*overwrite=*/0);
    setenv("GQE_MAX_TASK_MANAGER_MEMORY", "1073741824", /*overwrite=*/0);
    auto port_str     = std::to_string(port_);
    auto num_gpus_str = std::to_string(config.num_gpus);
    execl(node_manager_bin,
          node_manager_bin,
          "--address",
          "127.0.0.1",
          "--port",
          port_str.c_str(),
          "--num-gpus",
          num_gpus_str.c_str(),
          "--task-manager-binary",
          task_manager_bin,
          nullptr);
    _exit(127);
  }

  if (!wait_for_port(port_)) {
    stop();
    return false;
  }

  if (config.optimization_parameters) {
    auto status = set_session_options(port_, *config.optimization_parameters);
    if (!status.ok()) {
      load_error_ = status.ToString();
      stop();
      return false;
    }
  }

  // Forward TPCH_SCHEMA to load_tpch.py when set so CI can override the script's
  // default `schema.sql` with the handcoded `ci_schema.sql`.
  std::string schema_arg;
  if (auto const* schema = std::getenv("TPCH_SCHEMA"); schema && schema[0] != '\0') {
    schema_arg = std::format(" --schema {}", schema);
  }
  auto cmd = std::format(
    "{} --server-url http://127.0.0.1:{}{} {}", load_script, port_, schema_arg, data_path);
  auto [exit_code, output] = run_command_with_output(cmd);
  if (exit_code != 0) {
    load_error_ = output;
    stop();
    return false;
  }

  ready_       = true;
  config_name_ = config.name;
  return true;
}

void ServerInstance::stop()
{
  if (server_pid_ > 0) {
    kill(server_pid_, SIGTERM);
    int status;
    waitpid(server_pid_, &status, 0);
    server_pid_ = -1;
  }
  // Remove any shared memory segment left by the node manager.
  if (std::remove("/dev/shm/gqe_shared_memory") == 0) {
    std::fprintf(stderr,
                 "Warning: removed stale shared memory segment /dev/shm/gqe_shared_memory\n");
  }
  ready_ = false;
  config_name_.clear();
}

}  // namespace gqe_test

// RUN_ALL_TESTS() runs every test in the link unit, allowing a
// test executable to split its suites across source files without defining multiple main()
// functions. See https://google.github.io/googletest/primer.html#invoking-the-tests.
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new gqe_test::HarnessEnvironment);
  return RUN_ALL_TESTS();
}
