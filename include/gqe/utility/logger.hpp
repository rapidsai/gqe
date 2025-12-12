/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <fstream>
#include <memory>

namespace gqe::utility {

/**
 * @brief Return the logger used by the GQE library.
 *
 * @note The easiest way to log messages is to use the `GQE_LOG_*` macros.
 */
inline spdlog::logger* logger()
{
  static std::shared_ptr<spdlog::logger> _logger = []() {
    auto const log_file_path = std::getenv("GQE_LOG_FILE");
    std::shared_ptr<spdlog::logger> gqe_logger;

    if (log_file_path) {
      std::ofstream file_test(log_file_path);
      if (file_test.is_open()) {
        file_test.close();
        gqe_logger = spdlog::basic_logger_mt("gqe", log_file_path);
        gqe_logger->info("All logs will be written to the file path {}", log_file_path);
      } else {
        gqe_logger = spdlog::stdout_color_mt("gqe");
        gqe_logger->warn("GQE_LOG_FILE path {} is invalid. Falling back to console logger.",
                         log_file_path);
      }
    } else {
      gqe_logger = spdlog::stdout_color_mt("gqe");
    }

    auto const log_level = std::getenv("GQE_LOG_LEVEL");
    if (log_level) { gqe_logger->set_level(spdlog::level::from_str(log_level)); }
    gqe_logger->flush_on(spdlog::level::err);

    return gqe_logger;
  }();

  return _logger.get();
}

}  // namespace gqe::utility

#define GQE_LOG_TRACE(...)    gqe::utility::logger()->trace(__VA_ARGS__)
#define GQE_LOG_DEBUG(...)    gqe::utility::logger()->debug(__VA_ARGS__)
#define GQE_LOG_INFO(...)     gqe::utility::logger()->info(__VA_ARGS__)
#define GQE_LOG_WARN(...)     gqe::utility::logger()->warn(__VA_ARGS__)
#define GQE_LOG_ERROR(...)    gqe::utility::logger()->error(__VA_ARGS__)
#define GQE_LOG_CRITICAL(...) gqe::utility::logger()->critical(__VA_ARGS__)
