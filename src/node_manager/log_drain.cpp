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

#include <gqe/node_manager/log_drain.hpp>

#include <fcntl.h>
#include <spdlog/details/log_msg.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <format>
#include <stdexcept>
#include <system_error>

namespace gqe::node_manager {
namespace {

std::string_view to_string(log_level level)
{
  switch (level) {
    case log_level::trace: return "trace";
    case log_level::debug: return "debug";
    case log_level::info: return "info";
    case log_level::warn: return "warn";
    case log_level::error: return "error";
    case log_level::critical: return "critical";
    default: return "unknown";
  }
}

log_level from_spdlog(spdlog::level::level_enum level)
{
  switch (level) {
    case spdlog::level::trace: return log_level::trace;
    case spdlog::level::debug: return log_level::debug;
    case spdlog::level::info: return log_level::info;
    case spdlog::level::warn: return log_level::warn;
    case spdlog::level::err: return log_level::error;
    case spdlog::level::critical: return log_level::critical;
    case spdlog::level::off:
    case spdlog::level::n_levels: return log_level::info;
    default: return log_level::info;
  }
}

void write_all(int fd, std::string_view text)
{
  while (!text.empty()) {
    auto const written = ::write(fd, text.data(), text.size());
    if (written > 0) {
      text.remove_prefix(static_cast<std::size_t>(written));
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      return;
    }
  }
}

}  // namespace

log_drain::log_drain()
{
  if (auto const* path = std::getenv("GQE_LOG_FILE")) {
    fd_ = ::open(path, O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fd_ < 0) {
      throw std::system_error(errno, std::generic_category(), "Failed to open GQE_LOG_FILE");
    }
    owns_fd_ = true;
  } else {
    fd_ = STDERR_FILENO;
  }
}

log_drain::~log_drain()
{
  if (owns_fd_) { ::close(fd_); }
}

void log_drain::write(std::string_view source,
                      std::uint64_t timestamp_ns,
                      log_level level,
                      std::uint64_t thread_id,
                      std::string_view body)
{
  auto const timestamp = std::chrono::sys_time<std::chrono::nanoseconds>{
    std::chrono::nanoseconds{static_cast<std::chrono::nanoseconds::rep>(timestamp_ns)}};
  auto const whole_seconds = std::chrono::floor<std::chrono::seconds>(timestamp);
  auto const micros =
    std::chrono::duration_cast<std::chrono::microseconds>(timestamp - whole_seconds).count();
  auto const time = std::chrono::system_clock::to_time_t(whole_seconds);
  std::tm local_time{};
  localtime_r(&time, &local_time);

  std::array<char, 32> time_buffer{};
  std::strftime(time_buffer.data(), time_buffer.size(), "%Y-%m-%d %H:%M:%S", &local_time);

  auto const line = std::format("[{}.{:06}] [{}] [{}] [thread {}] {}\n",
                                time_buffer.data(),
                                micros,
                                source,
                                to_string(level),
                                thread_id,
                                body);
  std::scoped_lock lock(mutex_);
  write_all(fd_, line);
}

drain_sink::drain_sink(std::shared_ptr<log_drain> drain, std::string source)
  : drain_(std::move(drain)), source_(std::move(source))
{
}

void drain_sink::log(spdlog::details::log_msg const& msg)
{
  auto const timestamp_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(msg.time.time_since_epoch()).count();
  drain_->write(source_,
                static_cast<std::uint64_t>(timestamp_ns),
                from_spdlog(msg.level),
                msg.thread_id,
                std::string_view{msg.payload.data(), msg.payload.size()});
}

void drain_sink::flush() {}

void drain_sink::set_pattern(std::string const&) {}

void drain_sink::set_formatter(std::unique_ptr<spdlog::formatter>) {}

}  // namespace gqe::node_manager
