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

#include <gqe/executor/optimization_parameters.hpp>

#include <proto/node_task_manager.pb.h>

#include <arrow/flight/types.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <map>
#include <string>
#include <string_view>

namespace gqe::rpc {

/**
 * @brief Return all optimization parameters as a Flight session option map.
 *
 * Parameters with no value (e.g. nullopt) are represented as std::monostate
 * entries, matching the Flight SQL SHOW convention.
 */
[[nodiscard]] std::map<std::string, arrow::flight::SessionOptionValue>
optimization_parameters_to_session_options(gqe::optimization_parameters const& params);

/**
 * @brief Set a single optimization parameter from a Flight session option value.
 *
 * @return An error status if the name is unknown or the value has the wrong type.
 */
arrow::Status apply_session_option(gqe::optimization_parameters& params,
                                   std::string_view name,
                                   arrow::flight::SessionOptionValue const& value);

/** @brief Convert optimization_parameters to a proto OptimizationParameters. */
[[nodiscard]] proto::OptimizationParameters serialize_optimization_parameters(
  gqe::optimization_parameters const& params);

/** @brief Convert a proto OptimizationParameters to optimization_parameters. */
[[nodiscard]] arrow::Result<gqe::optimization_parameters> deserialize_optimization_parameters(
  proto::OptimizationParameters const& proto);

}  // namespace gqe::rpc
