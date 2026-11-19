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

#pragma once

#include <spdlog/sinks/sink.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace gqe::node_manager {

enum class log_level { trace, debug, info, warn, error, critical };

/**
 * @brief Serialize node and task-manager records into one process-owned log destination.
 *
 * The drain writes to the file selected by `GQE_LOG_FILE`, truncating it when the drain is
 * constructed, or to standard error when the variable is unset. Concurrent calls to @ref write
 * are serialized so each formatted record is emitted as one complete line.
 */
class log_drain {
 public:
  /**
   * @brief Open the configured log destination.
   *
   * @throws std::system_error if `GQE_LOG_FILE` is set but cannot be opened.
   */
  log_drain();
  ~log_drain();

  log_drain(log_drain const&)            = delete;
  log_drain& operator=(log_drain const&) = delete;

  /**
   * @brief Format and write one source-tagged log record.
   *
   * @param source Identifier for the process that produced the record, such as `server` or `gpu0`.
   * @param timestamp_ns Nanoseconds since the Unix epoch captured by the producing logger.
   * @param level Record severity.
   * @param thread_id Identifier of the producing thread.
   * @param body Already-formatted message body; the drain adds the common prefix and newline.
   */
  void write(std::string_view source,
             std::uint64_t timestamp_ns,
             log_level level,
             std::uint64_t thread_id,
             std::string_view body);

 private:
  std::mutex mutex_;
  int fd_       = -1;
  bool owns_fd_ = false;
};

/**
 * @brief Adapt a @ref log_drain for the node manager's process-wide spdlog logger.
 *
 * The adapter forwards spdlog message payloads to the shared drain with a fixed source tag.
 * Pattern and formatter configuration are ignored because the drain applies the common
 * multi-process log format, and flushing is a no-op because records are written directly to the
 * drain's file descriptor.
 */
class drain_sink final : public spdlog::sinks::sink {
 public:
  /**
   * @param drain Shared destination for node and task-manager records.
   * @param source Source tag attached to every record from this sink.
   */
  drain_sink(std::shared_ptr<log_drain> drain, std::string source);

  void log(spdlog::details::log_msg const& msg) override;
  void flush() override;
  void set_pattern(std::string const& pattern) override;
  void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) override;

 private:
  std::shared_ptr<log_drain> drain_;
  std::string source_;
};

}  // namespace gqe::node_manager
