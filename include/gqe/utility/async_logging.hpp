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

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace spdlog {
class logger;
}  // namespace spdlog

namespace gqe::utility {

struct async_logging_options {
  std::string address;
  std::string source_tag;
  std::size_t queue_capacity = 8192;  ///< Number of queued records; must be greater than zero.
};

/**
 * @brief Owns a private spdlog async queue and the task-manager log stream.
 *
 * To disable async forwarding, do not construct this runtime; the process will retain its
 * previously configured logger.
 */
class async_logging_runtime {
 public:
  explicit async_logging_runtime(async_logging_options options);
  ~async_logging_runtime();

  async_logging_runtime(async_logging_runtime const&)            = delete;
  async_logging_runtime& operator=(async_logging_runtime const&) = delete;

  /**
   * @brief Stop async forwarding after a bounded drain period.
   *
   * Concurrent calls are serialized; callers that arrive during shutdown wait for the active call
   * to finish.
   *
   * @param grace_period Maximum time to wait for the local queue and sink to become idle.
   */
  void shutdown(std::chrono::milliseconds grace_period = std::chrono::seconds{2});

 private:
  struct implementation;
  std::unique_ptr<implementation> impl_;
};

}  // namespace gqe::utility
