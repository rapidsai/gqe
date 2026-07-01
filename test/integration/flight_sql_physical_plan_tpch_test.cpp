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

#include "flight_sql_test_harness.hpp"
#include "flight_sql_test_utils.hpp"

#include <gtest/gtest.h>

#include <proto/physical_plan.pb.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <tuple>

namespace gqe_test {

class TpchPhysicalPlanQueryTest : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path || !g_config.physical_plans_dir || !g_config.ref_results) {
      GTEST_SKIP() << "TPCH_DATA_PATH, TPCH_PHYSICAL_PLANS, and TPCH_REF_RESULTS must be set";
    }
    auto [num_gpus, query_num] = GetParam();
    if (num_gpus > g_config.host_gpu_count) {
      GTEST_SKIP() << "Test requires " << num_gpus << " GPUs but host has "
                   << g_config.host_gpu_count;
    }
    ASSERT_TRUE(g_server.ready()) << "Server with " << num_gpus << " GPU(s) is not ready";
  }
};

TEST_P(TpchPhysicalPlanQueryTest, ExecuteQuery)
{
  auto [num_gpus, query_num] = GetParam();
  auto [exit_code, output]   = run_tpch_query(g_config.run_script,
                                            InputFormat::Physical,
                                            g_server.port(),
                                            g_config.physical_plans_dir,
                                            query_num,
                                            g_config.ref_results);
  ASSERT_EQ(exit_code, 0) << "TPC-H Q" << query_num << " failed. Output:\n" << output;
}

// Disabled queries:
//   Q8, Q14 — no plan-builder fixture available.
//   Q15    — flaky, matches flight_sql_tpch_test exclusion.

// Q11 is included here (the .pb fixture has the SF-substituted threshold
// baked in); flight_sql_tpch_test excludes it because the SQL text doesn't.

// clang-format off
auto const kTpchPhysicalPlanQueries = ::testing::Values(
  1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22);
// clang-format on

INSTANTIATE_TEST_SUITE_P(SingleGpu,
                         TpchPhysicalPlanQueryTest,
                         ::testing::Combine(::testing::Values(1), kTpchPhysicalPlanQueries),
                         [](::testing::TestParamInfo<std::tuple<int, int>> const& info) {
                           return std::format("Q{}", std::get<1>(info.param));
                         });

INSTANTIATE_TEST_SUITE_P(MultiGpu,
                         TpchPhysicalPlanQueryTest,
                         ::testing::Combine(::testing::Values(2), kTpchPhysicalPlanQueries),
                         [](::testing::TestParamInfo<std::tuple<int, int>> const& info) {
                           return std::format("Q{}", std::get<1>(info.param));
                         });

// ---------------------------------------------------------------------------
// Negative-path tests for the physical-plan dispatch.
// ---------------------------------------------------------------------------

class TpchPhysicalPlanDispatchErrorTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path) {
      GTEST_SKIP() << "TPCH_DATA_PATH must be set (server does not start without it)";
    }
    int num_gpus = GetParam();
    if (num_gpus > g_config.host_gpu_count) {
      GTEST_SKIP() << "Test requires " << num_gpus << " GPUs but host has "
                   << g_config.host_gpu_count;
    }
    ASSERT_TRUE(g_server.ready()) << "Server with " << num_gpus << " GPU(s) is not ready";
  }

  /** Send raw bytes as a serialized PhysicalRelation and return {exit_code, output}. */
  std::pair<int, std::string> send_physical_plan(std::string_view bytes) const
  {
    // Bytes go via a temp file because gqe-cli reads with `std::fs::read`, and
    // piping binary through `echo` mangles null bytes.
    auto tmp = std::filesystem::temp_directory_path() / std::format("gqe_pp_neg_{}.pb", getpid());
    {
      std::ofstream out(tmp, std::ios::binary);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    auto cmd    = std::format("{} --server-url http://127.0.0.1:{} --physical-plan {}",
                           g_config.client_bin,
                           g_server.port(),
                           tmp.string());
    auto result = run_command_with_output(cmd);
    std::filesystem::remove(tmp);
    return result;
  }

  /** Send bytes and assert the cli rejected them (non-zero exit). */
  void assert_rejected(std::string_view bytes) const
  {
    auto [exit_code, output] = send_physical_plan(bytes);
    ASSERT_NE(exit_code, 0) << "Plan should have been rejected. Output:\n" << output;
  }

  /** Send bytes, assert rejection, and assert the output contains @p expected. */
  void assert_rejected_with(std::string_view bytes, std::string_view expected) const
  {
    auto [exit_code, output] = send_physical_plan(bytes);
    ASSERT_NE(exit_code, 0) << "Plan should have been rejected. Output:\n" << output;
    EXPECT_NE(output.find(expected), std::string::npos)
      << "Output should contain \"" << expected << "\". Output:\n"
      << output;
  }
};

TEST_P(TpchPhysicalPlanDispatchErrorTest, RejectsCorruptProtobuf)
{
  // Random non-protobuf bytes; ParseFromArray returns false.
  assert_rejected_with(std::string_view{"\x42\x91\xab\xcd\x00\xff\x42\x91", 8},
                       "Failed to parse PhysicalRelation");
}

TEST_P(TpchPhysicalPlanDispatchErrorTest, RejectsEmptyRelation)
{
  // PhysicalRelation with no oneof field set; the deserializer rejects.
  gqe::proto::PhysicalRelation pb;
  std::string bytes;
  pb.SerializeToString(&bytes);
  assert_rejected(bytes);
}

TEST_P(TpchPhysicalPlanDispatchErrorTest, RejectsWriteRelation)
{
  // A write-relation submitted via the query path is rejected. The deserializer
  // requires a valid child; a minimal read of 'supplier' clears it.
  gqe::proto::PhysicalRelation pb;
  auto* write = pb.mutable_write();
  write->set_table_name("tmp_write_target");
  auto* read = write->mutable_child()->mutable_read();
  read->set_table_name("supplier");
  std::string bytes;
  pb.SerializeToString(&bytes);
  assert_rejected_with(bytes, "called with a write plan");
}

INSTANTIATE_TEST_SUITE_P(SingleGpu,
                         TpchPhysicalPlanDispatchErrorTest,
                         ::testing::Values(1),
                         [](::testing::TestParamInfo<int> const& info) {
                           return std::format("{}GPU", info.param);
                         });

INSTANTIATE_TEST_SUITE_P(MultiGpu,
                         TpchPhysicalPlanDispatchErrorTest,
                         ::testing::Values(2),
                         [](::testing::TestParamInfo<int> const& info) {
                           return std::format("{}GPU", info.param);
                         });

}  // namespace gqe_test
