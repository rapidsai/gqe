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

#include <gqe/node_manager/log_stream_service.hpp>

#include <gqe/node_manager/log_drain.hpp>

#include <chrono>
#include <cstdint>
#include <format>
#include <string>

namespace gqe::node_manager {
namespace {

constexpr std::size_t max_log_body_bytes   = 1024 * 1024;
constexpr std::size_t max_source_tag_bytes = 128;

std::uint64_t now_ns()
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count());
}

log_level from_proto(proto::LogLevel level)
{
  switch (level) {
    case proto::LOG_LEVEL_TRACE: return log_level::trace;
    case proto::LOG_LEVEL_DEBUG: return log_level::debug;
    case proto::LOG_LEVEL_WARN: return log_level::warn;
    case proto::LOG_LEVEL_ERROR: return log_level::error;
    case proto::LOG_LEVEL_CRITICAL: return log_level::critical;
    case proto::LOG_LEVEL_INFO:
    default: return log_level::info;
  }
}

}  // namespace

log_stream_service::log_stream_service(std::shared_ptr<log_drain> drain) : drain_(std::move(drain))
{
}

grpc::Status log_stream_service::StreamLogs(grpc::ServerContext*,
                                            grpc::ServerReader<proto::LogStreamMessage>* reader,
                                            proto::LogStreamAck* response)
{
  proto::LogStreamMessage message;
  if (!reader->Read(&message)) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "Log stream ended before the required header"};
  }
  if (!message.has_header()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "First log stream message must be a header"};
  }

  auto const source = message.header().source_tag();
  if (source.empty() || source.size() > max_source_tag_bytes) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Log source tag must contain 1 to {} bytes", max_source_tag_bytes)};
  }

  drain_->write(source,
                now_ns(),
                log_level::info,
                0,
                std::format("=== log stream opened (pid {})", message.header().pid()));

  std::uint64_t records_received = 0;
  while (reader->Read(&message)) {
    if (!message.has_record()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "Log header may only appear once"};
    }
    auto const& record = message.record();
    if (record.body().size() > max_log_body_bytes) {
      return {grpc::StatusCode::RESOURCE_EXHAUSTED, "Log record body is too large"};
    }
    if (record.dropped_before() != 0) {
      drain_->write(source,
                    record.timestamp_ns(),
                    log_level::warn,
                    0,
                    std::format("!!! {} log records dropped at source", record.dropped_before()));
    }
    drain_->write(
      source, record.timestamp_ns(), from_proto(record.level()), record.thread_id(), record.body());
    ++records_received;
  }

  response->set_records_received(records_received);
  drain_->write(source, now_ns(), log_level::info, 0, "=== log stream closed");
  return grpc::Status::OK;
}

}  // namespace gqe::node_manager
