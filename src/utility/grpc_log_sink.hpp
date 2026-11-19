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

#include <grpcpp/grpcpp.h>
#include <proto/log_stream.grpc.pb.h>

#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/sink.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace gqe::utility {

/**
 * @brief Send task-manager log records to the node manager over a client-streaming gRPC call.
 *
 * The owning @ref async_logging_runtime connects this sink to a spdlog thread pool with exactly
 * one worker. Consequently, calls to @ref log are serialized by construction; this class does not
 * provide its own serialization for multiple logging workers.
 *
 * gRPC writes are synchronous and may block the logging worker. Application threads remain
 * isolated from that blocking operation because they only enqueue records in the bounded spdlog
 * queue. The sink drops records when it is closing or cannot open a stream and reports accumulated
 * sink and queue drops in the next record sent successfully.
 *
 * The thread pool is held weakly so the sink cannot extend its lifetime or participate in an
 * ownership cycle through async messages that retain the logger.
 */
class grpc_log_sink final : public spdlog::sinks::sink {
 public:
  /**
   * @param address Node-manager log service address.
   * @param source_tag Source identifier included in the stream header.
   * @param pool Async thread pool used to read queue-overrun statistics.
   */
  grpc_log_sink(std::string address,
                std::string source_tag,
                std::weak_ptr<spdlog::details::thread_pool> pool);
  ~grpc_log_sink() override;

  /**
   * @brief Convert and synchronously write one spdlog record to the active gRPC stream.
   *
   * This method may block and must only be invoked by the owning runtime's single logging worker.
   */
  void log(spdlog::details::log_msg const& msg) override;
  void flush() override;
  void set_pattern(std::string const& pattern) override;
  void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) override;

  /**
   * @brief Stop accepting records and request cancellation of the current gRPC operation.
   *
   * Marks the sink as closing and wakes reconnect backoff before requesting best-effort gRPC
   * cancellation. Its purpose is to unblock a synchronous operation so bounded shutdown can
   * proceed.
   */
  void request_cancel();

  /**
   * @brief Request cancellation and finish the current stream after the logging worker has stopped.
   */
  void finish();

  /**
   * @brief Return whether the sink is outside its @ref log operation.
   *
   * This is a local activity snapshot. It does not prove that the server received or persisted all
   * records, or that the stream completed successfully.
   */
  [[nodiscard]] bool idle() const;

 private:
  bool open_stream();
  void finish_failed_stream();
  void schedule_reconnect();

  std::string source_tag_;
  std::weak_ptr<spdlog::details::thread_pool> pool_;  ///< Observes queue-overrun statistics.
  std::unique_ptr<proto::LogStreamService::Stub> stub_;

  std::mutex context_mutex_;  ///< Protects context access by the worker and shutdown thread.
  std::shared_ptr<grpc::ClientContext> context_;
  // Accessed only by the single logging worker, except after that worker has stopped.
  std::unique_ptr<grpc::ClientWriter<proto::LogStreamMessage>> writer_;
  proto::LogStreamAck ack_;

  std::atomic<bool> closing_{false};  ///< Stops writes and interrupts reconnect backoff.
  std::atomic<bool> active_{false};   ///< Whether the sole worker is executing log().
  std::atomic<std::uint64_t> sink_drops_{0};
  std::mutex backoff_mutex_;
  std::condition_variable backoff_cv_;
  std::uint64_t previous_queue_overruns_ = 0;
  std::chrono::milliseconds backoff_{250};
  std::chrono::steady_clock::time_point reconnect_after_{};
};

}  // namespace gqe::utility
