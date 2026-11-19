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

#include <gqe/optimizer/decode_throughput.hpp>

#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using gqe::compression_format;
using gqe::optimizer::calibration_profile;
using gqe::optimizer::calibration_profiles;
using gqe::optimizer::de_throughput_gibps;
using gqe::optimizer::is_calibrated_on;
using gqe::optimizer::profile_by_name;
using gqe::optimizer::sm_throughput_gibps;
using gqe::optimizer::throughput_point;

namespace {

/**
 * @brief The (type, format) curve of @p points, in table order.
 */
std::vector<throughput_point> curve(std::span<throughput_point const> points,
                                    nvcompType_t data_type,
                                    compression_format format)
{
  std::vector<throughput_point> result;
  std::ranges::copy_if(points, std::back_inserter(result), [&](throughput_point const& point) {
    return point.physical_type == data_type && point.format == format;
  });
  return result;
}

}  // namespace

/**
 * @brief Profiles are found by name, and two profiles may share a GPU.
 */
TEST(DecodeThroughputTest, ProfileIsFoundByName)
{
  for (auto const& profile : calibration_profiles()) {
    EXPECT_EQ(&profile_by_name(profile.name), &profile);
  }
  EXPECT_EQ(profile_by_name("gb200").calibrated_on, "NVIDIA GB200");
  EXPECT_EQ(profile_by_name("gb300").calibrated_on, "NVIDIA GB300");
  EXPECT_EQ(profile_by_name("b200").calibrated_on, "NVIDIA B200");

  calibration_profile const first{.name = "gb200", .calibrated_on = "NVIDIA GB200"};
  calibration_profile const second{.name = "gb200_c1g1", .calibrated_on = "NVIDIA GB200"};
  std::array<calibration_profile, 2> const profiles{first, second};
  EXPECT_EQ(&profile_by_name(profiles, "gb200_c1g1"), &profiles[1]);
}

/**
 * @brief An unknown or empty profile name is an error whose message names it and lists the
 * profiles.
 */
TEST(DecodeThroughputTest, UnknownProfileNameIsAnErrorListingTheProfiles)
{
  try {
    static_cast<void>(profile_by_name("no_such_profile"));
    FAIL() << "expected an unknown profile name to be rejected";
  } catch (std::invalid_argument const& error) {
    std::string const message{error.what()};
    EXPECT_NE(message.find("'no_such_profile'"), std::string::npos);
    EXPECT_NE(message.find("'gb200', 'gb300', 'b200'"), std::string::npos);
  }
  EXPECT_THROW(static_cast<void>(profile_by_name("")), std::invalid_argument);
}

/**
 * @brief A device name matches a profile when it starts with `calibrated_on`.
 */
TEST(DecodeThroughputTest, ProfileIsCalibratedOnTheGpuWhoseNameStartsWithItsPrefix)
{
  auto const& gb200 = profile_by_name("gb200");
  EXPECT_TRUE(is_calibrated_on(gb200, "NVIDIA GB200"));
  EXPECT_TRUE(is_calibrated_on(gb200, "NVIDIA GB200 NVL72"));
  EXPECT_FALSE(is_calibrated_on(gb200, "NVIDIA GB300"));
  EXPECT_FALSE(is_calibrated_on(gb200, "NVIDIA B200"));
  EXPECT_FALSE(is_calibrated_on(gb200, ""));
}

/**
 * @brief Every profile has a non-empty name that no other profile has.
 */
TEST(DecodeThroughputTest, ProfilesHaveDistinctNames)
{
  std::vector<std::string_view> names;
  for (auto const& profile : calibration_profiles()) {
    EXPECT_FALSE(profile.name.empty());
    names.push_back(profile.name);
  }
  std::ranges::sort(names);
  EXPECT_EQ(std::ranges::adjacent_find(names), names.end());
}

/**
 * @brief Every profile has a bandwidth, an SM advantage, and an SM and a DE table.
 */
TEST(DecodeThroughputTest, EveryProfileCarriesACompleteCalibration)
{
  for (auto const& profile : calibration_profiles()) {
    SCOPED_TRACE(std::string{profile.name});
    EXPECT_GT(profile.memcpy_bw_gibps, 0.0);
    EXPECT_GT(profile.sm_advantage, 0.0);
    EXPECT_FALSE(profile.sm_points.empty());
    EXPECT_FALSE(profile.de_points.empty());
  }
}

/**
 * @brief A lookup takes the calibrated point nearest in compression ratio.
 *
 * A ratio on a point takes it, one between two points takes the closer one, and one beyond the
 * range takes the end point. Expectations come from the table, so recalibrating keeps them valid.
 */
TEST(DecodeThroughputTest, LookupsTakeTheNearestCalibratedPoint)
{
  auto const& gb200 = profile_by_name("gb200");
  auto const points = curve(gb200.de_points, NVCOMP_TYPE_INT, compression_format::lz4);
  ASSERT_GE(points.size(), 2U);

  auto const de_throughput = [&](double ratio) {
    return de_throughput_gibps(gb200, NVCOMP_TYPE_INT, compression_format::lz4, ratio);
  };
  EXPECT_DOUBLE_EQ(de_throughput(points[0].compression_ratio), points[0].throughput_gibps);
  EXPECT_DOUBLE_EQ(de_throughput(points[1].compression_ratio - 1e-3), points[1].throughput_gibps);
  EXPECT_DOUBLE_EQ(de_throughput(1000.0), points.back().throughput_gibps);
}

/**
 * @brief Between two distant calibrated ratios the lookup steps at their midpoint.
 *
 * The lookup does not interpolate, so a ratio just below the midpoint takes the lower point and one
 * just above it the upper point, however far apart their throughputs are.
 */
TEST(DecodeThroughputTest, LookupStepsAtTheMidpointOfAWideGap)
{
  std::array<throughput_point, 2> const points{
    throughput_point{NVCOMP_TYPE_INT, compression_format::lz4, 1.0, 100.0},
    throughput_point{NVCOMP_TYPE_INT, compression_format::lz4, 9.0, 500.0}};
  calibration_profile const profile{.name = "sparse", .de_points = points, .sm_points = points};

  for (auto const lookup : {de_throughput_gibps, sm_throughput_gibps}) {
    EXPECT_DOUBLE_EQ(lookup(profile, NVCOMP_TYPE_INT, compression_format::lz4, 4.9), 100.0);
    EXPECT_DOUBLE_EQ(lookup(profile, NVCOMP_TYPE_INT, compression_format::lz4, 5.1), 500.0);
  }
}

/**
 * @brief Only CHAR, INT and LONGLONG are calibrated; every other nvCOMP type returns 0.
 *
 * `get_optimal_nvcomp_data_type()` maps INT16 and the unsigned cuDF types to these nvCOMP types, so
 * such columns get no automatic candidate.
 */
TEST(DecodeThroughputTest, UncalibratedDataTypesReturnZero)
{
  for (auto const& profile : calibration_profiles()) {
    SCOPED_TRACE(std::string{profile.name});
    for (auto const type : {NVCOMP_TYPE_SHORT,
                            NVCOMP_TYPE_USHORT,
                            NVCOMP_TYPE_UCHAR,
                            NVCOMP_TYPE_UINT,
                            NVCOMP_TYPE_ULONGLONG}) {
      EXPECT_DOUBLE_EQ(de_throughput_gibps(profile, type, compression_format::lz4, 2.0), 0.0);
      for (auto const format : {compression_format::ans,
                                compression_format::bitpacking,
                                compression_format::cascaded}) {
        EXPECT_DOUBLE_EQ(sm_throughput_gibps(profile, type, format, 2.0), 0.0);
      }
    }
  }
}

/**
 * @brief A (type, format) pair without a curve returns 0 instead of another curve's throughput.
 */
TEST(DecodeThroughputTest, UncalibratedCurveIsUnusable)
{
  for (auto const& profile : calibration_profiles()) {
    SCOPED_TRACE(std::string{profile.name});
    for (auto const type : {NVCOMP_TYPE_CHAR, NVCOMP_TYPE_INT, NVCOMP_TYPE_LONGLONG}) {
      auto const points = curve(profile.de_points, type, compression_format::lz4);
      ASSERT_FALSE(points.empty());
      EXPECT_DOUBLE_EQ(
        de_throughput_gibps(profile, type, compression_format::lz4, points[0].compression_ratio),
        points[0].throughput_gibps);
    }
    EXPECT_DOUBLE_EQ(de_throughput_gibps(profile, NVCOMP_TYPE_SHORT, compression_format::lz4, 2.0),
                     0.0);
    EXPECT_DOUBLE_EQ(de_throughput_gibps(profile, NVCOMP_TYPE_INT, compression_format::snappy, 2.0),
                     0.0);
    EXPECT_DOUBLE_EQ(sm_throughput_gibps(profile, NVCOMP_TYPE_INT, compression_format::lz4, 2.0),
                     0.0);
  }
}
