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

/** @file Shared utilities for Flight SQL integration tests. */

#pragma once

#include <sys/types.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gqe_test {

/** RAII wrapper for a socket file descriptor. */
struct scoped_socket {
  int fd;
  explicit scoped_socket(int fd) : fd(fd) {}
  ~scoped_socket()
  {
    if (fd >= 0) close(fd);
  }
  scoped_socket(scoped_socket const&)            = delete;
  scoped_socket& operator=(scoped_socket const&) = delete;
};

/** Find a free TCP port by binding to port 0 and reading back the assignment. */
int find_free_port();

/**
 * Poll until a TCP port accepts connections, or timeout.
 * @param port          TCP port to probe.
 * @param timeout_seconds  Maximum wait time in seconds.
 * @return true if a connection was established before the deadline.
 */
bool wait_for_port(int port, int timeout_seconds = 60);

/**
 * Run a shell command, capture stdout+stderr, and return {exit_code, output}.
 * @param cmd  The command string passed to @c popen.
 */
std::pair<int, std::string> run_command_with_output(std::string const& cmd);

/**
 * Find child process PIDs of the given parent by reading /proc.
 * @param parent  PID of the parent process.
 * @return PIDs of all immediate children.
 */
std::vector<pid_t> find_child_pids(pid_t parent);

/**
 * Run a SQL statement via the gqe-cli Flight SQL client.
 * @param client_bin  Path to the gqe-cli binary.
 * @param port        Flight SQL server port.
 * @param sql         SQL statement text (piped via stdin).
 * @return {exit_code, combined stdout+stderr}.
 */
std::pair<int, std::string> run_sql(std::string_view client_bin, int port, std::string_view sql);

/** Mode argument for `run_tpch_query`, mapping to `run_tpch.py --mode`. */
enum class InputFormat { Sql, Physical };

/** Return the `--mode` flag value for @p mode (`"sql"` or `"physical"`). */
[[nodiscard]] std::string_view to_string(InputFormat mode);

/**
 * Run a TPC-H query by number via the given run script.
 * @param run_script    Path to the run script (run_tpch.py).
 * @param mode          Run mode, passed through as `--mode <to_string(mode)>`.
 * @param port          Flight SQL server port.
 * @param queries_dir   Directory containing per-query files (`q<N>.sql` for sql mode,
 *                      `q<N>.pb` for physical mode).
 * @param query_num     TPC-H query number (1–22).
 * @param ref_dir       Optional path to reference results directory for validation.
 *                      When set, passes `--validate <ref_dir>` to the script.
 * @return {exit_code, combined stdout+stderr}.
 */
std::pair<int, std::string> run_tpch_query(std::string_view run_script,
                                           InputFormat mode,
                                           int port,
                                           std::string_view queries_dir,
                                           int query_num,
                                           std::optional<std::string_view> ref_dir = std::nullopt);

/**
 * Detect the number of NVIDIA GPUs available on the host via nvidia-smi.
 * @return Number of GPUs detected, or 0 on failure.
 */
int detect_gpu_count();

/**
 * Read an environment variable, returning @p fallback if it is not set.
 */
inline const char* env_or(const char* name, const char* fallback)
{
  auto* val = std::getenv(name);
  return val ? val : fallback;
}

}  // namespace gqe_test
