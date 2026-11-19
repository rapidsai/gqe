/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/decode_throughput.hpp>

/**
 * @file
 * @brief The calibration profiles behind `calibration_profiles()`, one generated source per GPU.
 *
 * Points are keyed by the nvCOMP type that `get_optimal_nvcomp_data_type()` picks for a buffer, not
 * by its SQL type:
 *
 * - DECIMAL64 and DECIMAL128: LONGLONG
 * - FLOAT64 and string characters: CHAR
 * - string offsets: INT, or LONGLONG once cuDF widens them
 */
namespace gqe::optimizer::calibration_tables {

[[nodiscard]] calibration_profile const& gb200_profile();
[[nodiscard]] calibration_profile const& gb300_profile();
[[nodiscard]] calibration_profile const& b200_profile();

}  // namespace gqe::optimizer::calibration_tables
