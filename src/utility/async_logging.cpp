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

#include <gqe/utility/async_logging.hpp>

#include "grpc_log_sink.hpp"

#include <gqe/utility/logger.hpp>

#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/null_sink.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace gqe::utility {
namespace {

/**
 * Own the complete async backend behind one shared control block.
 *
 * The process-wide logger stores an aliasing shared_ptr into this object. A logging call that
 * races with logger replacement therefore keeps the logger, its thread pool, and its sink alive
 * until that call returns. Destruction cancels an outstanding gRPC operation before joining the
 * worker and only finishes the stream after the worker has stopped.
 */
struct async_logging_backend {
  explicit async_logging_backend(async_logging_options options)
    : pool{std::make_shared<spdlog::details::thread_pool>(options.queue_capacity, std::size_t{1})},
      sink{std::make_shared<grpc_log_sink>(
        std::move(options.address), std::move(options.source_tag), pool)},
      logger{std::make_shared<spdlog::async_logger>(
        "gqe", sink, pool, spdlog::async_overflow_policy::overrun_oldest)}
  {
  }

  std::shared_ptr<spdlog::details::thread_pool> pool;
  std::shared_ptr<grpc_log_sink> sink;
  std::shared_ptr<spdlog::async_logger> logger;

  ~async_logging_backend()
  {
    sink->request_cancel();
    logger.reset();
    pool.reset();
    sink->finish();
  }
};

}  // namespace

struct async_logging_runtime::implementation {
  std::shared_ptr<async_logging_backend> backend;
  std::mutex shutdown_mutex;
  bool stopped = false;  // Guarded by shutdown_mutex.
};

async_logging_runtime::async_logging_runtime(async_logging_options options)
  : impl_(std::make_unique<implementation>())
{
  if (options.queue_capacity == 0) {
    throw std::invalid_argument(
      "Async logging queue capacity must be greater than zero; omit async_logging_runtime to "
      "disable async forwarding");
  }
  impl_->backend = std::make_shared<async_logging_backend>(std::move(options));
  configure_logger(*impl_->backend->logger);

  // The alias points at the logger while sharing ownership of the complete backend. This is what
  // keeps the async thread pool alive when a log call overlaps shutdown's logger replacement.
  auto logger = std::shared_ptr<spdlog::logger>(impl_->backend, impl_->backend->logger.get());
  gqe::utility::initialize_logger(std::move(logger));
}

async_logging_runtime::~async_logging_runtime() { shutdown(); }

void async_logging_runtime::shutdown(std::chrono::milliseconds grace_period)
{
  if (!impl_) { return; }

  std::scoped_lock lock(impl_->shutdown_mutex);
  if (std::exchange(impl_->stopped, true)) { return; }

  // Stop new calls from acquiring the async backend. Calls that already acquired it retain the
  // complete backend through the aliasing shared_ptr installed by the constructor.
  auto replacement = std::make_shared<spdlog::logger>(
    "gqe-shutdown", std::make_shared<spdlog::sinks::null_sink_mt>());
  gqe::utility::initialize_logger(std::move(replacement));

  auto backend = std::exchange(impl_->backend, {});
  backend->logger->flush();
  auto const deadline = std::chrono::steady_clock::now() + grace_period;
  while (std::chrono::steady_clock::now() < deadline &&
         (backend->pool->queue_size() != 0 || !backend->sink->idle())) {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }

  backend->sink->request_cancel();
}

}  // namespace gqe::utility
