/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
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
    auto gqe_logger = spdlog::stdout_color_mt("gqe");

    auto const log_level = std::getenv("GQE_LOG_LEVEL");
    if (log_level) gqe_logger->set_level(spdlog::level::from_str(log_level));

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