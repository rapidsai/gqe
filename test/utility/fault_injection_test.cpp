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

#include <gqe/utility/fault_injection.hpp>

#include <gtest/gtest.h>

#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

constexpr char const* trigger_variable = "GQE_TEST_FAULT_INJECTION_UNIT_TEST_FILE";

/// Sets the trigger variable to a fresh temporary file for the test and removes both afterwards.
class FaultInjectionTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    _path = std::filesystem::temp_directory_path() /
            ("gqe_fault_injection_test_" + std::to_string(getpid()) + "_" +
             ::testing::UnitTest::GetInstance()->current_test_info()->name());
    setenv(trigger_variable, _path.c_str(), /*overwrite=*/1);
  }

  void TearDown() override
  {
    unsetenv(trigger_variable);
    std::filesystem::remove(_path);
  }

  void write_trigger_file(std::string_view contents) const { std::ofstream(_path) << contents; }

  std::filesystem::path _path;
};

}  // namespace

TEST_F(FaultInjectionTest, UnsetVariableDoesNotTrigger)
{
  unsetenv(trigger_variable);
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}

TEST_F(FaultInjectionTest, EmptyVariableDoesNotTrigger)
{
  setenv(trigger_variable, "", /*overwrite=*/1);
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}

TEST_F(FaultInjectionTest, MissingFileDoesNotTrigger)
{
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}

TEST_F(FaultInjectionTest, TriggersOnlyForListedRank)
{
  write_trigger_file("1\n");
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
  EXPECT_TRUE(gqe::utility::fault_injection::find_trigger(trigger_variable, 1));
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 2));
}

TEST_F(FaultInjectionTest, DelayDefaultsToZero)
{
  write_trigger_file("0\n");
  auto const trigger = gqe::utility::fault_injection::find_trigger(trigger_variable, 0);
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->delay, std::chrono::seconds(0));
}

TEST_F(FaultInjectionTest, ReadsDelay)
{
  write_trigger_file("0 15\n");
  auto const trigger = gqe::utility::fault_injection::find_trigger(trigger_variable, 0);
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->delay, std::chrono::seconds(15));
}

TEST_F(FaultInjectionTest, MatchesAmongSeveralLines)
{
  write_trigger_file("0 3\n\n1 7\n2\n");
  EXPECT_EQ(gqe::utility::fault_injection::find_trigger(trigger_variable, 0)->delay,
            std::chrono::seconds(3));
  EXPECT_EQ(gqe::utility::fault_injection::find_trigger(trigger_variable, 1)->delay,
            std::chrono::seconds(7));
  EXPECT_EQ(gqe::utility::fault_injection::find_trigger(trigger_variable, 2)->delay,
            std::chrono::seconds(0));
}

TEST_F(FaultInjectionTest, FirstLineForARankApplies)
{
  write_trigger_file("0 3\n0 9\n");
  EXPECT_EQ(gqe::utility::fault_injection::find_trigger(trigger_variable, 0)->delay,
            std::chrono::seconds(3));
}

TEST_F(FaultInjectionTest, IgnoresMalformedLines)
{
  // A malformed line for the rank is skipped, and a later valid line for the same rank applies.
  write_trigger_file("not-a-rank\n0 soon\n0 -4\n0 6\n");
  auto const trigger = gqe::utility::fault_injection::find_trigger(trigger_variable, 0);
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->delay, std::chrono::seconds(6));
}

TEST_F(FaultInjectionTest, MalformedLinesAloneDoNotTrigger)
{
  write_trigger_file("0 soon\n0 -4\n");
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}

TEST_F(FaultInjectionTest, LineWithTrailingFieldsIsIgnored)
{
  // A line with anything after the delay does not parse, so a later valid line applies.
  write_trigger_file("0 8 unexpected\n0 8x\n0 3\n");
  auto const trigger = gqe::utility::fault_injection::find_trigger(trigger_variable, 0);
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->delay, std::chrono::seconds(3));
}

TEST_F(FaultInjectionTest, LineWithTrailingFieldsAloneDoesNotTrigger)
{
  write_trigger_file("0 8 unexpected\n");
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}

TEST_F(FaultInjectionTest, TrailingWhitespaceIsAllowed)
{
  write_trigger_file("0 8 \t \n");
  auto const trigger = gqe::utility::fault_injection::find_trigger(trigger_variable, 0);
  ASSERT_TRUE(trigger);
  EXPECT_EQ(trigger->delay, std::chrono::seconds(8));
}

TEST_F(FaultInjectionTest, FileCreatedAfterwardsArmsTheTrigger)
{
  // Tests arm a hook by creating the file after the process started, and disarm it by removing it.
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
  write_trigger_file("0\n");
  EXPECT_TRUE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
  std::filesystem::remove(_path);
  EXPECT_FALSE(gqe::utility::fault_injection::find_trigger(trigger_variable, 0));
}
