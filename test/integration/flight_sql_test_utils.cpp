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

#include <gtest/gtest.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect_pipe.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/readable_pipe.hpp>
#include <boost/asio/writable_pipe.hpp>
#include <boost/asio/write.hpp>
#include <boost/process.hpp>

#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
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

std::pair<int, std::string> run_command_with_output(std::string_view cmd)
{
  namespace bp   = boost::process;
  namespace asio = boost::asio;

  asio::io_context ctx;
  asio::readable_pipe out{ctx};

  // Run through the shell so command strings with pipes/redirects still work, and merge
  // stderr into stdout as the previous popen(cmd + " 2>&1", "r") did.
  auto const args = std::to_array<std::string>({"-c", std::format("{} 2>&1", cmd)});
  bp::process proc(ctx, "/bin/sh", args, bp::process_stdio{.out = out});

  boost::system::error_code ec;
  std::string output;
  asio::read(out, asio::dynamic_buffer(output), ec);  // reads until EOF
  return {proc.wait(), output};
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
  namespace bp   = boost::process;
  namespace asio = boost::asio;

  // Pipe the SQL to the client's stdin directly, without a shell. Interpolating it into
  // `echo "..."` would let the shell strip the double quotes around quoted identifiers
  // (e.g. CREATE TABLE "MixedCase"), silently turning the test into an unquoted-identifier
  // test. Piping preserves the statement verbatim, special characters and all.
  asio::io_context ctx;
  asio::writable_pipe in{ctx};
  asio::readable_pipe out{ctx};
  asio::writable_pipe sink{ctx};

  boost::system::error_code ec;
  // Give the child's stdout and stderr the write end of a single pipe (equivalent to 2>&1)
  // so the returned string is the combined stream and CLI errors reach the assertions.
  asio::connect_pipe(out, sink, ec);

  auto const args = std::to_array<std::string>(
    {"--server-url", std::format("http://127.0.0.1:{}", port), "--sql-file", "-"});
  bp::process proc(
    ctx, std::string(client_bin), args, bp::process_stdio{.in = in, .out = sink, .err = sink});

  // Close the parent's copy of the write end so the read below sees EOF once the child exits.
  sink.close();

  // The client reads all of stdin before producing output, so writing the whole statement
  // and then reading the response cannot deadlock.
  asio::write(in, asio::buffer(sql.data(), sql.size()), ec);
  in.close();

  std::string output;
  asio::read(out, asio::dynamic_buffer(output), ec);  // reads until EOF

  int exit_code = proc.wait();
  return {exit_code, output};
}

std::string_view to_string(InputFormat mode)
{
  switch (mode) {
    case InputFormat::Sql: return "sql";
    case InputFormat::Physical: return "physical";
  }
  return "";  // unreachable; switch is exhaustive
}

namespace {

[[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value)
{
  constexpr std::string_view whitespace = " \t\r";
  auto const first                      = value.find_first_not_of(whitespace);
  if (first == std::string_view::npos) { return {}; }
  auto const last = value.find_last_not_of(whitespace);
  return value.substr(first, last - first + 1);
}

/** Quote @p value as a SQL string literal. */
[[nodiscard]] std::string quote_sql_string(std::string_view value)
{
  std::string quoted = "'";
  for (char const c : value) {
    if (c == '\'') { quoted += '\''; }
    quoted += c;
  }
  quoted += '\'';
  return quoted;
}

}  // namespace

std::optional<std::string> show_session_option(std::string_view client_bin,
                                               int port,
                                               std::string_view name)
{
  auto const [exit_code, output] = run_sql(client_bin, port, std::format("SHOW {}", name));
  if (exit_code != 0) { return std::nullopt; }

  // SHOW prints a table row of the form `| <name> | <value> |`.
  std::string_view const text = output;
  for (std::size_t line_start = 0; line_start < text.size();) {
    auto const line_end = text.find('\n', line_start);
    auto const line     = text.substr(line_start, line_end - line_start);
    auto const first    = line.find('|');
    auto const second   = first == std::string_view::npos ? first : line.find('|', first + 1);
    auto const third    = second == std::string_view::npos ? second : line.find('|', second + 1);
    if (third != std::string_view::npos &&
        trim_ascii_whitespace(line.substr(first + 1, second - first - 1)) == name) {
      return std::string(trim_ascii_whitespace(line.substr(second + 1, third - second - 1)));
    }
    if (line_end == std::string_view::npos) { break; }
    line_start = line_end + 1;
  }
  return std::nullopt;
}

scoped_session_option::scoped_session_option(std::string client_bin,
                                             int port,
                                             std::string name,
                                             std::string_view value)
  : _client_bin(std::move(client_bin)), _port(port), _name(std::move(name))
{
  auto const previous = show_session_option(_client_bin, _port, _name);
  if (!previous.has_value()) {
    _error_output = std::format("SHOW {} did not print the option", _name);
    return;
  }
  bool const is_string = value.starts_with('\'');
  _restore_literal     = is_string ? quote_sql_string(*previous) : *previous;

  auto [exit_code, output] = run_sql(_client_bin, _port, std::format("SET {} TO {}", _name, value));
  if (exit_code != 0) {
    _error_output = std::move(output);
    return;
  }
  _is_set = true;
}

scoped_session_option::~scoped_session_option()
{
  // Restore even after a failed SET, since the server may have applied it before the failure.
  if (!_restore_literal.has_value()) { return; }
  // The destructor is noexcept, so an escaping exception would abort the whole test binary.
  try {
    auto const [exit_code, output] =
      run_sql(_client_bin, _port, std::format("SET {} TO {}", _name, *_restore_literal));
    if (exit_code != 0) {
      ADD_FAILURE() << "Cannot restore session option " << _name << ". Output:\n" << output;
    }
  } catch (std::exception const& error) {
    ADD_FAILURE() << "Cannot restore session option " << _name << ": " << error.what();
  } catch (...) {
    ADD_FAILURE() << "Cannot restore session option " << _name << ": unknown exception";
  }
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
