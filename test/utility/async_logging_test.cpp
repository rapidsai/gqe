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

#include <gqe/utility/logger.hpp>

#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>
#include <proto/log_stream.grpc.pb.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr auto wait_timeout                = std::chrono::seconds{10};
constexpr auto shutdown_bound              = std::chrono::seconds{5};
constexpr auto short_shutdown_grace        = std::chrono::milliseconds{10};
constexpr std::size_t test_queue_size      = 4;
constexpr std::size_t reconnect_queue_size = 64;
constexpr std::size_t large_body_size      = 512 * 1024;
constexpr std::size_t burst_size           = 64;

/// Behavior injected into the first stream accepted by the test service.
enum class first_stream_behavior { read, stall, fail_after_header };

/**
 * @brief In-process log receiver with controllable first-stream behavior.
 *
 * The service can read normally, stop before its first read to create transport backpressure, or
 * fail after the first header to exercise reconnection. Received records and stream state are
 * observable through bounded waits.
 */
class test_log_service final : public gqe::proto::LogStreamService::Service {
 public:
  explicit test_log_service(first_stream_behavior behavior) : behavior_(behavior) {}

  grpc::Status StreamLogs(grpc::ServerContext* context,
                          grpc::ServerReader<gqe::proto::LogStreamMessage>* reader,
                          gqe::proto::LogStreamAck* response) override
  {
    std::size_t stream_number;
    {
      std::scoped_lock lock(mutex_);
      stream_number = ++streams_started_;
    }
    state_changed_.notify_all();

    if (behavior_ == first_stream_behavior::stall && stream_number == 1) {
      std::unique_lock lock(mutex_);
      while (!reading_allowed_) {
        if (context->IsCancelled()) {
          ++streams_cancelled_;
          lock.unlock();
          state_changed_.notify_all();
          return {grpc::StatusCode::CANCELLED, "Test stream cancelled"};
        }
        state_changed_.wait_for(lock, std::chrono::milliseconds{10});
      }
    }

    gqe::proto::LogStreamMessage message;
    std::uint64_t records_received = 0;
    while (reader->Read(&message)) {
      if (behavior_ == first_stream_behavior::fail_after_header && stream_number == 1 &&
          message.has_header()) {
        return {grpc::StatusCode::UNAVAILABLE, "Injected stream failure"};
      }
      if (message.has_record()) {
        {
          std::scoped_lock lock(mutex_);
          records_.push_back(message.record());
        }
        ++records_received;
        state_changed_.notify_all();
      }
    }
    response->set_records_received(records_received);
    return grpc::Status::OK;
  }

  void allow_reading()
  {
    {
      std::scoped_lock lock(mutex_);
      reading_allowed_ = true;
    }
    state_changed_.notify_all();
  }

  [[nodiscard]] bool wait_for_streams(std::size_t expected,
                                      std::chrono::milliseconds timeout = wait_timeout)
  {
    std::unique_lock lock(mutex_);
    return state_changed_.wait_for(lock, timeout, [&] { return streams_started_ >= expected; });
  }

  [[nodiscard]] bool wait_for_cancelled_stream(std::chrono::milliseconds timeout = wait_timeout)
  {
    std::unique_lock lock(mutex_);
    return state_changed_.wait_for(lock, timeout, [&] { return streams_cancelled_ != 0; });
  }

  [[nodiscard]] bool wait_for_dropped_record(std::chrono::milliseconds timeout = wait_timeout)
  {
    std::unique_lock lock(mutex_);
    return state_changed_.wait_for(lock, timeout, [&] {
      return std::ranges::any_of(
        records_, [](gqe::proto::LogRecord const& record) { return record.dropped_before() != 0; });
    });
  }

 private:
  first_stream_behavior const behavior_;
  std::mutex mutex_;
  std::condition_variable state_changed_;
  bool reading_allowed_          = false;
  std::size_t streams_started_   = 0;
  std::size_t streams_cancelled_ = 0;
  std::vector<gqe::proto::LogRecord> records_;
};

/**
 * @brief Own a loopback gRPC server and its controllable log service.
 */
class test_log_server {
 public:
  explicit test_log_server(first_stream_behavior behavior) : service_(behavior)
  {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    if (!server_ || port == 0) { throw std::runtime_error("Failed to start test log server"); }
    address_ = "127.0.0.1:" + std::to_string(port);
  }

  ~test_log_server()
  {
    service_.allow_reading();
    server_->Shutdown();
    server_->Wait();
  }

  [[nodiscard]] std::string const& address() const { return address_; }
  [[nodiscard]] test_log_service& service() { return service_; }

 private:
  test_log_service service_;
  std::unique_ptr<grpc::Server> server_;
  std::string address_;
};

/// Override any process environment setting that would suppress the test's info records.
void enable_test_logging() { gqe::utility::logger()->set_level(spdlog::level::info); }

/// Enqueue enough large records to exceed the deliberately small async queue.
void log_large_burst(std::shared_ptr<spdlog::logger> const& logger)
{
  std::string const body(large_body_size, 'x');
  for (std::size_t index = 0; index < burst_size; ++index) {
    logger->info("{} {}", index, body);
  }
}

/**
 * @brief Queue saturation drops records and reports the loss without blocking the producer.
 */
TEST(AsyncLoggingTest, QueuePressureReportsDrops)
{
  test_log_server server(first_stream_behavior::stall);
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "queue-pressure-test", test_queue_size});
  enable_test_logging();

  auto producer = gqe::utility::logger();
  producer->info("open stream");
  ASSERT_TRUE(server.service().wait_for_streams(1));

  auto burst = std::async(std::launch::async, [producer] { log_large_burst(producer); });
  auto const completion_status = burst.wait_for(shutdown_bound);

  // Release the injected backpressure before joining a producer that missed the deadline.
  server.service().allow_reading();
  EXPECT_EQ(completion_status, std::future_status::ready);
  burst.get();

  producer.reset();
  ASSERT_TRUE(server.service().wait_for_dropped_record());
  runtime.shutdown();
}

/**
 * @brief Bounded shutdown cancels a stream whose logging worker is blocked by backpressure.
 */
TEST(AsyncLoggingTest, ShutdownCancelsInFlightLog)
{
  test_log_server server(first_stream_behavior::stall);
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "in-flight-shutdown-test", test_queue_size});
  enable_test_logging();

  auto producer = gqe::utility::logger();
  producer->info("open stream");
  ASSERT_TRUE(server.service().wait_for_streams(1));
  log_large_burst(producer);
  producer.reset();

  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  auto shutdown =
    std::async(std::launch::async, [&runtime] { runtime.shutdown(short_shutdown_grace); });
  auto const completion_status = shutdown.wait_for(shutdown_bound);

  // Transport backpressure should not prevent cleanup after a failed bounded wait.
  if (completion_status != std::future_status::ready) { server.service().allow_reading(); }
  EXPECT_EQ(completion_status, std::future_status::ready);
  shutdown.get();
  EXPECT_TRUE(server.service().wait_for_cancelled_stream());
}

/**
 * @brief Concurrent shutdown callers both complete while the logging worker is in flight.
 */
TEST(AsyncLoggingTest, ConcurrentShutdownCompletes)
{
  test_log_server server(first_stream_behavior::stall);
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "concurrent-shutdown-test", test_queue_size});
  enable_test_logging();

  auto producer                                  = gqe::utility::logger();
  std::weak_ptr<spdlog::logger> backend_lifetime = producer;
  producer->info("open stream");
  ASSERT_TRUE(server.service().wait_for_streams(1));
  log_large_burst(producer);
  producer.reset();
  std::this_thread::sleep_for(std::chrono::milliseconds{100});

  std::barrier start_shutdown{3};
  std::atomic<std::size_t> completed{0};
  std::jthread first([&] {
    start_shutdown.arrive_and_wait();
    runtime.shutdown(short_shutdown_grace);
    completed.fetch_add(1, std::memory_order_relaxed);
  });
  std::jthread second([&] {
    start_shutdown.arrive_and_wait();
    runtime.shutdown(short_shutdown_grace);
    completed.fetch_add(1, std::memory_order_relaxed);
  });

  start_shutdown.arrive_and_wait();
  first.join();
  second.join();
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 2);
  EXPECT_TRUE(server.service().wait_for_cancelled_stream());
  EXPECT_TRUE(backend_lifetime.expired());
}

/**
 * @brief A logger acquired before shutdown retains the complete async backend until released.
 */
TEST(AsyncLoggingTest, RetainedLoggerKeepsBackendAlive)
{
  test_log_server server(first_stream_behavior::read);
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "retained-logger-test", test_queue_size});
  enable_test_logging();

  auto acquired_logger                           = gqe::utility::logger();
  std::weak_ptr<spdlog::logger> backend_lifetime = acquired_logger;
  std::atomic<std::size_t> logging_errors{0};
  // spdlog reports an expired async thread pool through its error handler instead of throwing.
  acquired_logger->set_error_handler(
    [&](std::string const&) { logging_errors.fetch_add(1, std::memory_order_relaxed); });

  runtime.shutdown(short_shutdown_grace);
  EXPECT_FALSE(backend_lifetime.expired());
  acquired_logger->info("log through retained logger");
  EXPECT_EQ(logging_errors.load(std::memory_order_relaxed), 0);

  acquired_logger.reset();
  EXPECT_TRUE(backend_lifetime.expired());
}

/**
 * @brief Queued async messages do not form an ownership cycle that survives shutdown.
 */
TEST(AsyncLoggingTest, QueuedMessagesDoNotCreateOwnershipCycle)
{
  test_log_server server(first_stream_behavior::stall);
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "ownership-cycle-test", test_queue_size});
  enable_test_logging();

  auto producer                                  = gqe::utility::logger();
  std::weak_ptr<spdlog::logger> backend_lifetime = producer;
  producer->info("open stream");
  ASSERT_TRUE(server.service().wait_for_streams(1));
  log_large_burst(producer);
  producer.reset();

  runtime.shutdown(short_shutdown_grace);
  EXPECT_TRUE(backend_lifetime.expired());
}

/**
 * @brief Reconnection after a failed stream reports the records lost by that failure.
 */
TEST(AsyncLoggingTest, ReconnectReportsDrops)
{
  test_log_server server(first_stream_behavior::fail_after_header);
  // This queue holds every attempted record, isolating sink failures from queue overruns.
  gqe::utility::async_logging_runtime runtime(
    {server.address(), "reconnect-test", reconnect_queue_size});
  enable_test_logging();

  auto producer = gqe::utility::logger();
  producer->info("open failing stream");
  ASSERT_TRUE(server.service().wait_for_streams(1));

  for (std::size_t attempt = 0; attempt < 20; ++attempt) {
    producer->info("reconnect attempt {}", attempt);
    if (server.service().wait_for_dropped_record(std::chrono::milliseconds{100})) { break; }
  }

  EXPECT_TRUE(server.service().wait_for_streams(2));
  EXPECT_TRUE(server.service().wait_for_dropped_record());
  producer.reset();
  runtime.shutdown();
}

}  // namespace
