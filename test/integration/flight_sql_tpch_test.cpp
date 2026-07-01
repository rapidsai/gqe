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

#include <format>
#include <tuple>

namespace gqe_test {

// ---------------------------------------------------------------------------
// TPC-H query test — parameterized by (num_gpus, query_number).
// ---------------------------------------------------------------------------

class TpchQueryTest : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path || !g_config.queries_dir || !g_config.ref_results) {
      GTEST_SKIP() << "TPCH_DATA_PATH, TPCH_QUERIES, and TPCH_REF_RESULTS must be set";
    }
    auto [num_gpus, query_num] = GetParam();
    if (num_gpus > g_config.host_gpu_count) {
      GTEST_SKIP() << "Test requires " << num_gpus << " GPUs but host has "
                   << g_config.host_gpu_count;
    }
    ASSERT_TRUE(g_server.ready()) << "Server with " << num_gpus << " GPU(s) is not ready";
  }
};

TEST_P(TpchQueryTest, ExecuteQuery)
{
  auto [num_gpus, query_num] = GetParam();
  auto [exit_code, output]   = run_tpch_query(g_config.run_script,
                                            InputFormat::Sql,
                                            g_server.port(),
                                            g_config.queries_dir,
                                            query_num,
                                            g_config.ref_results);
  ASSERT_EQ(exit_code, 0) << "TPC-H Q" << query_num << " failed. Output:\n" << output;
}

// clang-format off
auto const kTpchQueries = ::testing::Values(
  // Q15 is disabled here and in test/end_to_end/ due to flaky results.
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, /* 15, */ 16, 17, 18, 19, 20, 21, 22);
// clang-format on

INSTANTIATE_TEST_SUITE_P(SingleGpu,
                         TpchQueryTest,
                         ::testing::Combine(::testing::Values(1), kTpchQueries),
                         [](::testing::TestParamInfo<std::tuple<int, int>> const& info) {
                           return std::format("Q{}", std::get<1>(info.param));
                         });

INSTANTIATE_TEST_SUITE_P(MultiGpu,
                         TpchQueryTest,
                         ::testing::Combine(::testing::Values(2), kTpchQueries),
                         [](::testing::TestParamInfo<std::tuple<int, int>> const& info) {
                           return std::format("Q{}", std::get<1>(info.param));
                         });

}  // namespace gqe_test
