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

#include <gqe/optimizer/decode_throughput.hpp>

#include "calibration/tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>

namespace gqe::optimizer {

namespace {

/**
 * @brief Throughput of the (type, format) point of @p points nearest to @p compression_ratio, or 0
 * if there is none. Ties go to the first point.
 */
double nearest_point_throughput_gibps(std::span<throughput_point const> points,
                                      nvcompType_t data_type,
                                      gqe::compression_format format,
                                      double compression_ratio)
{
  auto curve         = points | std::views::filter([&](throughput_point const& point) {
                 return point.physical_type == data_type && point.format == format;
               });
  auto const nearest = std::ranges::min_element(curve, {}, [&](throughput_point const& point) {
    return std::abs(point.compression_ratio - compression_ratio);
  });
  return nearest == curve.end() ? 0.0 : nearest->throughput_gibps;
}

}  // namespace

std::span<calibration_profile const> calibration_profiles()
{
  static std::array<calibration_profile, 3> const profiles{calibration_tables::gb200_profile(),
                                                           calibration_tables::gb300_profile(),
                                                           calibration_tables::b200_profile()};
  return profiles;
}

calibration_profile const& profile_by_name(std::span<calibration_profile const> profiles,
                                           std::string_view name)
{
  std::string known;
  for (auto const& profile : profiles) {
    if (profile.name == name) { return profile; }
    known += std::format("{}'{}'", known.empty() ? "" : ", ", profile.name);
  }
  throw std::invalid_argument(
    std::format("No calibration profile is named '{}'. The profiles are {}.", name, known));
}

calibration_profile const& profile_by_name(std::string_view name)
{
  return profile_by_name(calibration_profiles(), name);
}

bool is_calibrated_on(calibration_profile const& profile, std::string_view gpu_name)
{
  return gpu_name.starts_with(profile.calibrated_on);
}

double de_throughput_gibps(calibration_profile const& profile,
                           nvcompType_t data_type,
                           gqe::compression_format format,
                           double compression_ratio)
{
  return nearest_point_throughput_gibps(profile.de_points, data_type, format, compression_ratio);
}

double sm_throughput_gibps(calibration_profile const& profile,
                           nvcompType_t data_type,
                           gqe::compression_format format,
                           double compression_ratio)
{
  return nearest_point_throughput_gibps(profile.sm_points, data_type, format, compression_ratio);
}

}  // namespace gqe::optimizer
