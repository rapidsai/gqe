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

#include "flight_sql_test_utils.hpp"

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdio>
#include <cstdlib>
#include <format>

namespace gqe_test {

int find_free_port()
{
  scoped_socket sock{socket(AF_INET, SOCK_STREAM, 0)};
  if (sock.fd < 0) { return -1; }
  struct sockaddr_in addr = {};
  addr.sin_family         = AF_INET;
  addr.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
  addr.sin_port           = 0;
  bind(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  getsockname(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), &len);
  return ntohs(addr.sin_port);
}

bool wait_for_port(int port, int timeout_seconds)
{
  for (int attempt = 0; attempt < timeout_seconds * 2; ++attempt) {
    usleep(500'000);
    scoped_socket sock{socket(AF_INET, SOCK_STREAM, 0)};
    struct sockaddr_in addr = {};
    addr.sin_family         = AF_INET;
    addr.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
    addr.sin_port           = htons(port);
    if (connect(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
      return true;
    }
  }
  return false;
}

std::pair<int, std::string> run_command_with_output(std::string const& cmd)
{
  std::string output;
  FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
  if (!pipe) { return {-1, ""}; }
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    output += buffer;
  }
  int status    = pclose(pipe);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return {exit_code, output};
}

std::vector<pid_t> find_child_pids(pid_t parent)
{
  std::vector<pid_t> children;
  DIR* dir = opendir("/proc");
  if (!dir) return children;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type != DT_DIR) continue;
    char* end;
    long pid = strtol(entry->d_name, &end, 10);
    if (*end != '\0' || pid <= 0) continue;
    auto stat_path = std::format("/proc/{}/stat", pid);
    FILE* f        = fopen(stat_path.c_str(), "r");
    if (!f) continue;
    int child_pid, ppid;
    char comm[256], state;
    if (fscanf(f, "%d %255s %c %d", &child_pid, comm, &state, &ppid) == 4) {
      if (ppid == parent) { children.push_back(static_cast<pid_t>(child_pid)); }
    }
    fclose(f);
  }
  closedir(dir);
  return children;
}

std::pair<int, std::string> run_sql(std::string_view client_bin, int port, std::string_view sql)
{
  return run_command_with_output(std::format(
    "echo \"{}\" | {} --server-url http://127.0.0.1:{} --sql-file -", sql, client_bin, port));
}

std::string_view to_string(InputFormat mode)
{
  switch (mode) {
    case InputFormat::Sql: return "sql";
    case InputFormat::Physical: return "physical";
  }
  return "";  // unreachable; switch is exhaustive
}

std::pair<int, std::string> run_tpch_query(std::string_view run_script,
                                           InputFormat mode,
                                           int port,
                                           std::string_view queries_dir,
                                           int query_num,
                                           std::optional<std::string_view> ref_dir)
{
  auto cmd =
    std::format("{} --mode {} --server-url http://127.0.0.1:{}", run_script, to_string(mode), port);
  if (ref_dir.has_value()) { cmd += std::format(" --validate {}", *ref_dir); }
  cmd += std::format(" {} {}", queries_dir, query_num);
  return run_command_with_output(cmd);
}

int detect_gpu_count()
{
  auto const* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  auto cmd =
    (visible && visible[0] != '\0')
      ? std::format("nvidia-smi --query-gpu=name --format=csv,noheader --id={} | wc -l", visible)
      : std::string("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l");
  auto [rc, output] = run_command_with_output(cmd);
  return (rc == 0) ? std::atoi(output.c_str()) : 0;
}

}  // namespace gqe_test
