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

#include <gqe/types.hpp>

#include <nvcomp/shared_types.h>

#include <span>
#include <string_view>

/**
 * @file
 * @brief Per-GPU decode-throughput calibration for automatic compression format selection.
 *
 * A profile holds the measurements of one GPU in one system, since the host-to-device bandwidth
 * depends on how the GPU is connected. A session selects a profile by name, never by device name.
 * See `docs/tuning/automatic-compression-calibration.md` for how to measure and add a profile.
 */
namespace gqe::optimizer {

/**
 * @brief Decode throughput of one (nvCOMP data type, format) curve at one measured compression
 * ratio.
 */
struct throughput_point {
  nvcompType_t physical_type;
  gqe::compression_format format;
  double compression_ratio;
  double throughput_gibps;
};

/**
 * @brief The decode-throughput tables and cost-model constants of one GPU.
 */
struct calibration_profile {
  /** @brief Name a session selects the profile by, for example "gb300". */
  std::string_view name;

  /** @brief Device-name prefix of the calibrated GPU, to tell whether a profile fits the GPU. */
  std::string_view calibrated_on;

  /** @brief Decompression engine (DE) table. Empty on a GPU without a DE. */
  std::span<throughput_point const> de_points;

  /** @brief SM decode table. */
  std::span<throughput_point const> sm_points;

  /** @brief Host-to-device copy bandwidth in GiB/s. */
  double memcpy_bw_gibps;

  /**
   * @brief Multiplier by which an SM representation's estimated time must undercut the DE time
   * to replace it.
   */
  double sm_advantage;
};

/**
 * @brief Every calibrated profile.
 */
[[nodiscard]] std::span<calibration_profile const> calibration_profiles();

/**
 * @brief The profile of @p profiles named @p name.
 *
 * @throws std::invalid_argument if no profile has that name. The message lists the known names.
 */
[[nodiscard]] calibration_profile const& profile_by_name(
  std::span<calibration_profile const> profiles, std::string_view name);

/**
 * @brief The profile of `calibration_profiles()` named @p name.
 *
 * @throws std::invalid_argument like the overload above.
 */
[[nodiscard]] calibration_profile const& profile_by_name(std::string_view name);

/**
 * @brief Whether @p gpu_name, as reported by `cudaDeviceProp::name`, starts with `calibrated_on`.
 */
[[nodiscard]] bool is_calibrated_on(calibration_profile const& profile, std::string_view gpu_name);

/**
 * @brief Decompression engine (DE) throughput in GiB/s at the calibrated ratio nearest to
 * @p compression_ratio, or 0 if the (type, format) curve was not calibrated.
 *
 * The lookup does not interpolate. On a sparse curve the estimate therefore steps at the midpoint
 * between two calibrated ratios, which can be a large jump.
 *
 * Only the nvCOMP types CHAR, INT and LONGLONG are calibrated. Any other type, such as SHORT or an
 * unsigned type, has no curve and returns 0, so a column of that type gets no automatic candidate
 * and is stored uncompressed.
 */
[[nodiscard]] double de_throughput_gibps(calibration_profile const& profile,
                                         nvcompType_t data_type,
                                         gqe::compression_format format,
                                         double compression_ratio);

/**
 * @brief SM decode throughput in GiB/s at the calibrated ratio nearest to @p compression_ratio, or
 * 0 if the (type, format) curve was not calibrated.
 *
 * The lookup and the uncalibrated types are as for `de_throughput_gibps()`.
 */
[[nodiscard]] double sm_throughput_gibps(calibration_profile const& profile,
                                         nvcompType_t data_type,
                                         gqe::compression_format format,
                                         double compression_ratio);

}  // namespace gqe::optimizer
