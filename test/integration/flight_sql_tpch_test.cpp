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
#include "tpch_query_sets.hpp"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

namespace gqe_test {

namespace {

/** @brief Remove leading and trailing ASCII whitespace from a borrowed string. */
[[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value)
{
  constexpr std::string_view whitespace = " \t\r\n\v\f";
  auto const first                      = value.find_first_not_of(whitespace);
  if (first == std::string_view::npos) { return {}; }
  auto const last = value.find_last_not_of(whitespace);
  return value.substr(first, last - first + 1);
}

/** @brief Extract the value row printed by `gqe-cli SHOW execution_backend`. */
[[nodiscard]] std::optional<std::string> parse_execution_backend(std::string_view output)
{
  for (std::size_t line_start = 0; line_start < output.size();) {
    auto const line_end = output.find('\n', line_start);
    auto const line     = output.substr(line_start, line_end - line_start);
    auto const first    = line.find('|');
    auto const second   = first == std::string_view::npos ? first : line.find('|', first + 1);
    auto const third    = second == std::string_view::npos ? second : line.find('|', second + 1);
    if (third != std::string_view::npos) {
      auto const name = trim_ascii_whitespace(line.substr(first + 1, second - first - 1));
      if (name == "execution_backend") {
        auto const value = trim_ascii_whitespace(line.substr(second + 1, third - second - 1));
        if (!value.empty()) { return std::string(value); }
      }
    }
    if (line_end == std::string_view::npos) { break; }
    line_start = line_end + 1;
  }
  return std::nullopt;
}

/** @brief Set the execution backend through the `gqe-cli` Flight SQL interface. */
[[nodiscard]] std::pair<int, std::string> set_execution_backend(std::string_view backend)
{
  return run_sql(
    g_config.client_bin, g_server.port(), std::format("SET execution_backend TO '{}'", backend));
}

}  // namespace

/** @brief Base fixture that starts the configured Flight SQL server for TPC-H tests. */
class FlightSqlTpchTest : public ::testing::Test {
 protected:
  void SetUp() override final
  {
    if (!g_config.data_path || !g_config.queries_dir || !g_config.ref_results) {
      GTEST_SKIP() << "TPCH_DATA_PATH, TPCH_QUERIES, and TPCH_REF_RESULTS must be set";
    }
    auto const& server_config = server_configuration();
    if (server_config.num_gpus > g_config.host_gpu_count) {
      GTEST_SKIP() << "Test requires " << server_config.num_gpus << " GPUs but host has "
                   << g_config.host_gpu_count;
    }
    ASSERT_TRUE(g_server.start(server_config,
                               g_config.node_manager_bin,
                               g_config.task_manager_bin,
                               g_config.load_script,
                               g_config.data_path))
      << "Failed to start server config " << server_config.name << ": " << g_server.load_error();
    ASSERT_TRUE(g_server.ready()) << "Server is not ready";
    server_started();
  }

  /** @brief Return the server configuration required by this fixture. */
  [[nodiscard]] virtual ServerConfiguration const& server_configuration() const = 0;

  /** @brief Perform fixture-specific setup after the server is ready. */
  virtual void server_started() {}
};

/** @brief TPC-H query test parameterized by server configuration and query number. */
class TpchQueryTest : public FlightSqlTpchTest,
                      public ::testing::WithParamInterface<std::tuple<ServerConfiguration, int>> {
 protected:
  [[nodiscard]] ServerConfiguration const& server_configuration() const override
  {
    return std::get<0>(GetParam());
  }
};

TEST_P(TpchQueryTest, ExecuteQuery)
{
  auto const query_num     = std::get<1>(GetParam());
  auto [exit_code, output] = run_tpch_query(g_config.run_script,
                                            InputFormat::Sql,
                                            g_server.port(),
                                            g_config.queries_dir,
                                            query_num,
                                            g_config.ref_results);
  ASSERT_EQ(exit_code, 0) << "TPC-H Q" << query_num << " failed. Output:\n" << output;
}

/** @brief Exercises QEP backend selection and restores the server's prior backend after the test.
 */
class QepFlightSqlTpchTest : public FlightSqlTpchTest {
 protected:
  [[nodiscard]] ServerConfiguration const& server_configuration() const override
  {
    return kLocalContiguousTableSingleGPU;
  }

  void server_started() override
  {
    auto [exit_code, output] =
      run_sql(g_config.client_bin, g_server.port(), "SHOW execution_backend");
    ASSERT_EQ(exit_code, 0) << "SHOW execution_backend failed. Output:\n" << output;
    initial_backend_ = parse_execution_backend(output);
    ASSERT_TRUE(initial_backend_.has_value())
      << "SHOW execution_backend returned an unknown value. Output:\n"
      << output;
  }

  void TearDown() override
  {
    if (!initial_backend_.has_value() || !g_server.ready()) { return; }

    auto [restore_exit_code, restore_output] = set_execution_backend(*initial_backend_);
    EXPECT_EQ(restore_exit_code, 0) << "Failed to restore execution_backend. Output:\n"
                                    << restore_output;
    if (restore_exit_code != 0) { return; }

    auto [show_exit_code, show_output] =
      run_sql(g_config.client_bin, g_server.port(), "SHOW execution_backend");
    EXPECT_EQ(show_exit_code, 0) << "SHOW execution_backend after restoration failed. Output:\n"
                                 << show_output;
    if (show_exit_code == 0) {
      EXPECT_EQ(parse_execution_backend(show_output), initial_backend_)
        << "execution_backend was not restored. Output:\n"
        << show_output;
    }
  }

 private:
  std::optional<std::string> initial_backend_;
};

TEST_F(QepFlightSqlTpchTest, SetsExecutionBackendWithGqeCli)
{
  auto const query_path     = std::filesystem::path(g_config.queries_dir) / "q06.sql";
  auto const reference_path = std::filesystem::path(g_config.ref_results) / "q6.parquet";
  ASSERT_TRUE(std::filesystem::is_regular_file(query_path))
    << "TPC-H Q6 input is missing: " << query_path;
  ASSERT_TRUE(std::filesystem::is_regular_file(reference_path))
    << "TPC-H Q6 reference result is missing: " << reference_path;

  auto const qep_backend           = gqe::to_string(gqe::execution_backend_type::qep_task_graph);
  auto [set_exit_code, set_output] = set_execution_backend(qep_backend);
  ASSERT_EQ(set_exit_code, 0) << "SET execution_backend failed. Output:\n" << set_output;

  auto [show_exit_code, show_output] =
    run_sql(g_config.client_bin, g_server.port(), "SHOW execution_backend");
  ASSERT_EQ(show_exit_code, 0) << "SHOW execution_backend failed. Output:\n" << show_output;
  ASSERT_EQ(parse_execution_backend(show_output), qep_backend)
    << "execution_backend was not set to qep_task_graph. Output:\n"
    << show_output;

  auto [exit_code, output] = run_tpch_query(g_config.run_script,
                                            InputFormat::Sql,
                                            g_server.port(),
                                            g_config.queries_dir,
                                            6,
                                            g_config.ref_results);
  ASSERT_EQ(exit_code, 0) << "TPC-H Q6 with qep_task_graph failed. Output:\n" << output;
}

/**
 * @brief End-to-end SQL fetch (`LIMIT`/`OFFSET`) through the QEP backend.
 *
 * A `LIMIT` without `ORDER BY` returns arbitrary rows (an ordered limit is instead rewritten to
 * top-K), so the assertions check row counts only. The TPC-H region table has exactly 5 rows at
 * every scale factor, and lineitem has far more than three, so the lineitem query exercises the
 * fetch early exit over a larger scan. The expected patterns include the parentheses of the
 * gqe-cli `Results (N rows):` banner so they cannot match inside a larger count.
 */
TEST_F(QepFlightSqlTpchTest, ExecutesSqlFetchQueries)
{
  auto const qep_backend           = gqe::to_string(gqe::execution_backend_type::qep_task_graph);
  auto [set_exit_code, set_output] = set_execution_backend(qep_backend);
  ASSERT_EQ(set_exit_code, 0) << "SET execution_backend failed. Output:\n" << set_output;

  // Verify the backend actually changed, so the fetch assertions below cannot silently pass on a
  // different backend.
  auto [show_exit_code, show_output] =
    run_sql(g_config.client_bin, g_server.port(), "SHOW execution_backend");
  ASSERT_EQ(show_exit_code, 0) << "SHOW execution_backend failed. Output:\n" << show_output;
  ASSERT_EQ(parse_execution_backend(show_output), qep_backend)
    << "execution_backend was not set to qep_task_graph. Output:\n"
    << show_output;

  struct fetch_case {
    std::string_view sql;
    std::string_view expected;
  };
  constexpr std::array<fetch_case, 3> cases{{
    // Early-exits the scan once three rows are emitted.
    {"SELECT l_orderkey FROM lineitem LIMIT 3", "(3 rows)"},
    // The window [3, 6) clamps to the two remaining rows.
    {"SELECT r_regionkey FROM region LIMIT 3 OFFSET 3", "(2 rows)"},
    // A count larger than the table returns every row.
    {"SELECT r_regionkey FROM region LIMIT 10", "(5 rows)"},
  }};
  for (auto const& fetch : cases) {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), fetch.sql);
    ASSERT_EQ(exit_code, 0) << "Query failed: " << fetch.sql << "\nOutput:\n" << output;
    EXPECT_NE(output.find(fetch.expected), std::string::npos)
      << "Expected " << fetch.expected << " for: " << fetch.sql << "\nOutput:\n"
      << output;
  }
}

INSTANTIATE_TEST_SUITE_P(
  LocalContiguousTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kLocalContiguousTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  LocalContiguousTableMultiGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kLocalContiguousTableMultiGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  SharedContiguousTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kSharedContiguousTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

INSTANTIATE_TEST_SUITE_P(
  SharedCompressedTableSingleGPU,
  TpchQueryTest,
  ::testing::Combine(::testing::Values(kSharedCompressedTableSingleGPU),
                     ::testing::ValuesIn(kTpchSqlQueries)),
  [](::testing::TestParamInfo<std::tuple<ServerConfiguration, int>> const& info) {
    return std::format("Q{}", std::get<1>(info.param));
  });

}  // namespace gqe_test
