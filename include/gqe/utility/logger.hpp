/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <spdlog/spdlog.h>

#include <memory>

namespace gqe::utility {

/**
 * @brief Return the process-wide GQE logger as a shared-ownership handle.
 *
 * @note The easiest way to log messages is to use the `GQE_LOG_*` macros.
 * @note Retaining shared ownership for the complete log call makes replacing the process-wide
 * logger safe with respect to calls already in progress.
 * @return A shared pointer that keeps the selected logger and its backend alive.
 */
std::shared_ptr<spdlog::logger> logger();

/**
 * @brief Replace the process-wide GQE logger.
 *
 * Calls that already acquired the previous logger retain it until their log operation completes.
 */
void initialize_logger(std::shared_ptr<spdlog::logger> new_logger);

/** @brief Apply the standard GQE environment level and flush settings. */
void configure_logger(spdlog::logger& target);

}  // namespace gqe::utility

#define GQE_LOG_TRACE(...)    gqe::utility::logger()->trace(__VA_ARGS__)
#define GQE_LOG_DEBUG(...)    gqe::utility::logger()->debug(__VA_ARGS__)
#define GQE_LOG_INFO(...)     gqe::utility::logger()->info(__VA_ARGS__)
#define GQE_LOG_WARN(...)     gqe::utility::logger()->warn(__VA_ARGS__)
#define GQE_LOG_ERROR(...)    gqe::utility::logger()->error(__VA_ARGS__)
#define GQE_LOG_CRITICAL(...) gqe::utility::logger()->critical(__VA_ARGS__)
