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

// End-to-end tests for the statistics system tables (_gqe_table_stats / _gqe_column_stats).
// Precise projection values are covered by the deterministic component test
// (test/node_manager/stats_tables_test.cpp); these tests exercise the full round-trip:
// catalog -> projection -> RefreshStatsTables -> shared-memory write -> SELECT.

#include "flight_sql_test_harness.hpp"
#include "flight_sql_test_utils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <format>
#include <string>
#include <string_view>

namespace gqe_test {

class FlightSqlStatsTest : public ::testing::TestWithParam<ServerConfiguration> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path) { GTEST_SKIP() << "TPCH_DATA_PATH must be set"; }
    auto const& server_config = GetParam();
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
  }

  void assert_sql_ok(std::string_view sql) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
  }

  void assert_sql_fails(std::string_view statement) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), statement);
    ASSERT_NE(exit_code, 0) << statement << " unexpectedly succeeded. Output:\n" << output;
  }

  void assert_sql_fails_with(std::string_view statement, std::string_view expected) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), statement);
    ASSERT_NE(exit_code, 0) << statement << " unexpectedly succeeded. Output:\n" << output;
    EXPECT_NE(output.find(expected), std::string::npos)
      << statement << " should fail with \"" << expected << "\". Output:\n"
      << output;
  }

  void assert_sql_output_contains(std::string_view sql, std::string_view expected) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
    EXPECT_NE(output.find(expected), std::string::npos)
      << sql << " output should contain \"" << expected << "\". Output:\n"
      << output;
  }

  void assert_sql_output_excludes(std::string_view sql, std::string_view unexpected) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
    EXPECT_EQ(output.find(unexpected), std::string::npos)
      << sql << " output should NOT contain \"" << unexpected << "\". Output:\n"
      << output;
  }

  std::pair<int, std::string> sql(std::string_view statement) const
  {
    return run_sql(g_config.client_bin, g_server.port(), statement);
  }

  /** COPY the TPC-H region table (5 rows) into @p table. */
  void copy_region_into(std::string_view table) const
  {
    assert_sql_ok(
      std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table, g_config.data_path));
  }
};

// The statistics tables are always registered as shared_segment_memory, so every configuration
// exercises the shared-memory refresh/write path. The multi-GPU configuration additionally
// exercises the rank-0-writes / other-ranks-reattach flow across task-manager processes.
INSTANTIATE_TEST_SUITE_P(SharedContiguousTableSingleGPU,
                         FlightSqlStatsTest,
                         ::testing::Values(kSharedContiguousTableSingleGPU));

INSTANTIATE_TEST_SUITE_P(LocalContiguousTableMultiGPU,
                         FlightSqlStatsTest,
                         ::testing::Values(kLocalContiguousTableMultiGPU));

// The stats tables exist and are queryable on a fresh-ish server without crashing, regardless of
// whether any user table has been created (regression for the empty-read SIGFPE).
TEST_P(FlightSqlStatsTest, StatsTablesAreQueryable)
{
  assert_sql_ok("SELECT * FROM _gqe_table_stats");
  assert_sql_ok("SELECT * FROM _gqe_column_stats");
}

TEST_P(FlightSqlStatsTest, StatsTablesRejectUserModification)
{
  struct system_table {
    std::string_view name;
    std::string_view known_column;
  };
  constexpr std::array tables{
    system_table{"_gqe_table_stats", "table_name"},
    system_table{"_gqe_column_stats", "column_name"},
  };

  for (auto const& table : tables) {
    auto const select = std::format("SELECT {} FROM {}", table.known_column, table.name);

    assert_sql_fails(std::format("DROP TABLE {}", table.name));
    assert_sql_ok(select);

    assert_sql_fails(std::format("DROP TABLE IF EXISTS {}", table.name));
    assert_sql_ok(select);

    assert_sql_fails(
      std::format("CREATE OR REPLACE TABLE {}(replacement INTEGER NOT NULL)", table.name));
    assert_sql_ok(select);

    assert_sql_fails(std::format("CREATE TABLE {}(replacement INTEGER NOT NULL)", table.name));
    assert_sql_ok(select);

    assert_sql_fails_with(
      std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table.name, g_config.data_path),
      "not writeable");
    assert_sql_ok(select);
  }
}

// After CREATE + COPY, the loaded table appears in _gqe_table_stats — i.e. the full
// catalog -> project -> refresh -> shared-write -> read round-trip works.
TEST_P(FlightSqlStatsTest, TableStatsReflectLoadedTable)
{
  assert_sql_ok("DROP TABLE IF EXISTS stats_loaded");
  assert_sql_ok(
    "CREATE TABLE stats_loaded(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
    "r_comment VARCHAR NOT NULL)");
  copy_region_into("stats_loaded");

  assert_sql_output_contains("SELECT * FROM _gqe_table_stats", "stats_loaded");
  assert_sql_ok("DROP TABLE IF EXISTS stats_loaded");
}

// Column-level stats include the table's columns, identified by a uniquely-named column.
TEST_P(FlightSqlStatsTest, ColumnStatsReflectColumns)
{
  assert_sql_ok("DROP TABLE IF EXISTS stats_cols");
  assert_sql_ok(
    "CREATE TABLE stats_cols(uniquecol_zzz INTEGER NOT NULL, namecol VARCHAR NOT NULL)");

  assert_sql_output_contains("SELECT * FROM _gqe_column_stats", "uniquecol_zzz");
  assert_sql_output_contains("SELECT * FROM _gqe_column_stats", "namecol");
  assert_sql_ok("DROP TABLE IF EXISTS stats_cols");
}

// A CREATE EXTERNAL TABLE (parquet, file-backed) is excluded from the stats tables, while a normal
// in-memory table is included.
TEST_P(FlightSqlStatsTest, ExternalTableExcluded)
{
  assert_sql_ok("DROP TABLE IF EXISTS stats_inmem");
  assert_sql_ok("DROP TABLE IF EXISTS stats_external_excluded");
  assert_sql_ok("CREATE TABLE stats_inmem(r_regionkey INTEGER NOT NULL)");
  assert_sql_ok(std::format(
    "CREATE EXTERNAL TABLE stats_external_excluded(r_regionkey INTEGER NOT NULL, "
    "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL) STORED AS PARQUET LOCATION '{}/region'",
    g_config.data_path));

  assert_sql_output_contains("SELECT * FROM _gqe_table_stats", "stats_inmem");
  assert_sql_output_excludes("SELECT * FROM _gqe_table_stats", "stats_external_excluded");
  assert_sql_output_excludes("SELECT * FROM _gqe_column_stats", "stats_external_excluded");

  assert_sql_ok("DROP TABLE IF EXISTS stats_inmem");
  assert_sql_ok("DROP TABLE IF EXISTS stats_external_excluded");
}

// Statistics are re-materialized lazily: a table queried after a COPY reflects the load. region has
// exactly 5 rows at every scale factor, so the row count is a stable assertion.
TEST_P(FlightSqlStatsTest, StatsRefreshAfterCopy)
{
  assert_sql_ok("DROP TABLE IF EXISTS stats_refresh");
  assert_sql_ok(
    "CREATE TABLE stats_refresh(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
    "r_comment VARCHAR NOT NULL)");
  copy_region_into("stats_refresh");

  // Project just the row count for this table; region loads 5 rows.
  assert_sql_output_contains(
    "SELECT num_rows FROM _gqe_table_stats WHERE table_name = 'stats_refresh'", "5");
  assert_sql_ok("DROP TABLE IF EXISTS stats_refresh");
}

}  // namespace gqe_test
