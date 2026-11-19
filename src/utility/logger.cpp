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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gqe::utility {
namespace {

std::shared_ptr<spdlog::logger> make_default_logger()
{
  auto const* log_file_path = std::getenv("GQE_LOG_FILE");
  std::shared_ptr<spdlog::logger> result;

  if (log_file_path != nullptr) {
    std::ofstream file_test(log_file_path, std::ios::app);
    if (file_test.is_open()) {
      file_test.close();
      result = spdlog::basic_logger_mt("gqe", log_file_path);
      result->info("All logs will be written to the file path {}", log_file_path);
    } else {
      result = spdlog::stdout_color_mt("gqe");
      result->warn("GQE_LOG_FILE path {} is invalid. Falling back to console logger.",
                   log_file_path);
    }
  } else {
    result = spdlog::stdout_color_mt("gqe");
  }

  configure_logger(*result);
  return result;
}

struct logger_state {
  std::mutex initialization_mutex;
  std::atomic<std::shared_ptr<spdlog::logger>> instance;
};

logger_state& state()
{
  static logger_state value;
  return value;
}

}  // namespace

std::shared_ptr<spdlog::logger> logger()
{
  auto& global = state();
  if (auto current = global.instance.load(std::memory_order_acquire)) { return current; }

  std::scoped_lock lock(global.initialization_mutex);
  if (auto current = global.instance.load(std::memory_order_acquire)) { return current; }

  auto current = make_default_logger();
  global.instance.store(current, std::memory_order_release);
  return current;
}

void initialize_logger(std::shared_ptr<spdlog::logger> new_logger)
{
  if (!new_logger) { throw std::invalid_argument("GQE logger cannot be null"); }
  configure_logger(*new_logger);

  auto& global = state();
  std::scoped_lock lock(global.initialization_mutex);
  global.instance.store(std::move(new_logger), std::memory_order_release);
}

void configure_logger(spdlog::logger& target)
{
  if (auto const* log_level = std::getenv("GQE_LOG_LEVEL")) {
    target.set_level(spdlog::level::from_str(log_level));
  }
  target.flush_on(spdlog::level::err);
}

}  // namespace gqe::utility
