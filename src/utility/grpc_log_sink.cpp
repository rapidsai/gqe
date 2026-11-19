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

#include "grpc_log_sink.hpp"

#include <gqe/rpc/task_migration.hpp>

#include <grpc/impl/channel_arg_names.h>

#include <spdlog/details/log_msg.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <tuple>
#include <utility>

namespace gqe::utility {
namespace {

/**
 * Publish whether the sink's sole logging worker is inside a potentially blocking operation.
 *
 * This guard tracks activity for shutdown polling; it does not serialize calls. The owning async
 * logging runtime configures exactly one worker.
 */
class activity_guard {
 public:
  explicit activity_guard(std::atomic<bool>& active) : active_(active)
  {
    active_.store(true, std::memory_order_relaxed);
  }
  ~activity_guard() { active_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool>& active_;
};

proto::LogLevel to_proto(spdlog::level::level_enum level)
{
  switch (level) {
    case spdlog::level::trace: return proto::LOG_LEVEL_TRACE;
    case spdlog::level::debug: return proto::LOG_LEVEL_DEBUG;
    case spdlog::level::info: return proto::LOG_LEVEL_INFO;
    case spdlog::level::warn: return proto::LOG_LEVEL_WARN;
    case spdlog::level::err: return proto::LOG_LEVEL_ERROR;
    case spdlog::level::critical: return proto::LOG_LEVEL_CRITICAL;
    case spdlog::level::off:
    case spdlog::level::n_levels: return proto::LOG_LEVEL_INFO;
  }
  return proto::LOG_LEVEL_INFO;
}

}  // namespace

grpc_log_sink::grpc_log_sink(std::string address,
                             std::string source_tag,
                             std::weak_ptr<spdlog::details::thread_pool> pool)
  : source_tag_(std::move(source_tag)), pool_(std::move(pool))
{
  grpc::ChannelArguments args;
  // Keepalive detects a node manager that died without closing the connection.
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, grpc_keepalive_time_ms);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 3'000);
  // Pings on an idle connection; `apply_default_server_args` must permit pings without calls too,
  // or the node manager sends a "too_many_pings" GOAWAY during quiet periods.
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
  stub_        = proto::LogStreamService::NewStub(std::move(channel));
}

grpc_log_sink::~grpc_log_sink() { request_cancel(); }

void grpc_log_sink::log(spdlog::details::log_msg const& msg)
{
  activity_guard guard(active_);
  if (closing_.load(std::memory_order_acquire)) {
    sink_drops_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  if (!writer_ && !open_stream()) {
    sink_drops_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  proto::LogStreamMessage message;
  auto* record = message.mutable_record();
  auto const timestamp_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(msg.time.time_since_epoch()).count();
  record->set_timestamp_ns(static_cast<std::uint64_t>(timestamp_ns));
  record->set_level(to_proto(msg.level));
  record->set_thread_id(msg.thread_id);
  record->set_body(msg.payload.data(), msg.payload.size());

  std::uint64_t queue_drops = 0;
  if (auto pool = pool_.lock()) {
    auto const current       = pool->overrun_counter();
    queue_drops              = current - previous_queue_overruns_;
    previous_queue_overruns_ = current;
  }
  record->set_dropped_before(sink_drops_.exchange(0, std::memory_order_relaxed) + queue_drops);

  if (!writer_->Write(message)) {
    sink_drops_.fetch_add(record->dropped_before() + 1, std::memory_order_relaxed);
    finish_failed_stream();
    schedule_reconnect();
  }
}

void grpc_log_sink::flush() {}

void grpc_log_sink::set_pattern(std::string const&) {}

void grpc_log_sink::set_formatter(std::unique_ptr<spdlog::formatter>) {}

void grpc_log_sink::request_cancel()
{
  closing_.store(true, std::memory_order_release);
  backoff_cv_.notify_all();
  std::shared_ptr<grpc::ClientContext> context;
  {
    std::scoped_lock lock(context_mutex_);
    context = context_;
  }
  // Retain the context locally so TryCancel() can run without holding the sink's mutex.
  if (context) { context->TryCancel(); }
}

void grpc_log_sink::finish()
{
  request_cancel();
  finish_failed_stream();
}

bool grpc_log_sink::idle() const { return !active_.load(std::memory_order_acquire); }

bool grpc_log_sink::open_stream()
{
  if (closing_.load(std::memory_order_acquire)) { return false; }
  if (std::chrono::steady_clock::now() < reconnect_after_) {
    std::unique_lock lock(backoff_mutex_);
    backoff_cv_.wait_until(
      lock, reconnect_after_, [this] { return closing_.load(std::memory_order_acquire); });
    if (closing_.load(std::memory_order_acquire)) { return false; }
  }

  auto context = std::make_shared<grpc::ClientContext>();
  {
    std::scoped_lock lock(context_mutex_);
    if (closing_.load(std::memory_order_acquire)) { return false; }
    context_ = context;
  }

  writer_ = stub_->StreamLogs(context.get(), &ack_);
  if (!writer_) {
    finish_failed_stream();
    schedule_reconnect();
    return false;
  }

  proto::LogStreamMessage message;
  auto* header = message.mutable_header();
  header->set_source_tag(source_tag_);
  header->set_pid(static_cast<std::uint64_t>(::getpid()));
  if (!writer_->Write(message)) {
    finish_failed_stream();
    schedule_reconnect();
    return false;
  }

  backoff_ = std::chrono::milliseconds{250};
  return true;
}

void grpc_log_sink::finish_failed_stream()
{
  if (writer_) {
    std::ignore = writer_->Finish();
    writer_.reset();
  }
  std::scoped_lock lock(context_mutex_);
  context_.reset();
}

void grpc_log_sink::schedule_reconnect()
{
  reconnect_after_ = std::chrono::steady_clock::now() + backoff_;
  backoff_         = std::min(backoff_ * 2, std::chrono::milliseconds{5'000});
}

}  // namespace gqe::utility
