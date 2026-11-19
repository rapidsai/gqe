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

#include <gqe/utility/logger.hpp>

#include <gtest/gtest.h>

#include <spdlog/sinks/null_sink.h>

#include <future>
#include <memory>
#include <thread>

namespace {

/**
 * @brief A logger acquired by one thread remains valid while the process-wide logger is replaced.
 */
TEST(LoggerTest, AcquiredLoggerSurvivesProcessWideReplacement)
{
  auto old_logger =
    std::make_shared<spdlog::logger>("old-logger", std::make_shared<spdlog::sinks::null_sink_mt>());
  std::weak_ptr<spdlog::logger> old_logger_lifetime = old_logger;
  gqe::utility::initialize_logger(old_logger);
  old_logger.reset();

  std::promise<void> logger_acquired;
  auto logger_acquired_future = logger_acquired.get_future();
  std::promise<void> continue_logging;
  auto continue_logging_future = continue_logging.get_future();

  std::jthread logging_thread([&] {
    auto acquired_logger = gqe::utility::logger();
    logger_acquired.set_value();
    continue_logging_future.wait();
    acquired_logger->info("log after process-wide logger replacement");
  });

  logger_acquired_future.wait();
  auto replacement = std::make_shared<spdlog::logger>(
    "replacement-logger", std::make_shared<spdlog::sinks::null_sink_mt>());
  gqe::utility::initialize_logger(std::move(replacement));

  EXPECT_FALSE(old_logger_lifetime.expired());
  continue_logging.set_value();
  logging_thread.join();
  EXPECT_TRUE(old_logger_lifetime.expired());
}

}  // namespace
