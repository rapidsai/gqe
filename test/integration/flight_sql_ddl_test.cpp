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

#include <filesystem>
#include <format>
#include <future>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace gqe_test {

class FlightSqlDdlTest : public ::testing::TestWithParam<ServerConfiguration> {
 protected:
  void SetUp() override
  {
    if (!g_config.data_path || !g_config.queries_dir || !g_config.ref_results) {
      GTEST_SKIP() << "TPCH_DATA_PATH, TPCH_QUERIES, and TPCH_REF_RESULTS must be set";
    }
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

  /** Run a SQL statement and assert that it succeeds. */
  void assert_sql_ok(std::string_view sql) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
  }

  /**
   * Run a SQL statement, assert success, and assert the output contains @p expected.
   *
   * The match is a simple substring search over the full gqe-cli text output, which includes
   * boilerplate such as "Results (N rows):". Avoid asserting short patterns that could
   * accidentally match boilerplate instead of actual data cells. For example, use "| 1" to
   * match a result cell containing 1, not bare "1" which would match "Results (1 rows):".
   */
  void assert_sql_output_contains(std::string_view sql, std::string_view expected) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_EQ(exit_code, 0) << sql << " failed. Output:\n" << output;
    EXPECT_NE(output.find(expected), std::string::npos)
      << sql << " output should contain \"" << expected << "\". Output:\n"
      << output;
  }

  /** Run a SQL statement and assert that it fails. */
  void assert_sql_fails(std::string_view sql) const
  {
    auto [exit_code, output] = run_sql(g_config.client_bin, g_server.port(), sql);
    ASSERT_NE(exit_code, 0) << sql << " should have failed. Output:\n" << output;
  }

  /** Run a SQL statement via the Flight SQL client and return {exit_code, output}. */
  std::pair<int, std::string> sql(std::string_view statement) const
  {
    return run_sql(g_config.client_bin, g_server.port(), statement);
  }

  /** Run a TPC-H query by number via run_tpch.py (sql mode) with result validation. */
  std::pair<int, std::string> query(int query_num) const
  {
    return run_tpch_query(g_config.run_script,
                          InputFormat::Sql,
                          g_server.port(),
                          g_config.queries_dir,
                          query_num,
                          g_config.ref_results);
  }
};

// ---------------------------------------------------------------------------
// FlightSqlDdlTest test cases
// ---------------------------------------------------------------------------

/** Verify tables were created and loaded via DDL+COPY by running an ad-hoc query. */
TEST_P(FlightSqlDdlTest, CreateTableAndCopy)
{
  assert_sql_output_contains("SELECT r_name FROM region ORDER BY r_regionkey", "AFRICA");
}

/**
 * Verify result columns are labelled with the query's real names / aliases (captured from the
 * Substrait RelRoot.names), not generic placeholders like column_0. Every matched token appears
 * only in the header row, never in region's data values, so a match can only come from the header.
 */
TEST_P(FlightSqlDdlTest, ResultColumnNames)
{
  // SELECT * surfaces the table's real column names, and the old generic placeholders
  // (column_0, column_1, ...) must be gone.
  {
    auto [exit_code, output] = sql("SELECT * FROM region");
    ASSERT_EQ(exit_code, 0) << output;
    EXPECT_NE(output.find("r_regionkey"), std::string::npos) << output;
    EXPECT_NE(output.find("r_comment"), std::string::npos) << output;
    EXPECT_EQ(output.find("column_0"), std::string::npos)
      << "result columns should carry real names, not placeholders:\n"
      << output;
  }

  // An explicit alias becomes the column header (DataFusion lowercases unquoted identifiers).
  assert_sql_output_contains("SELECT r_regionkey AS keyalias FROM region", "keyalias");

  // Mixed projection: an aliased column and a bare column are both labelled correctly.
  {
    auto [exit_code, output] = sql("SELECT r_name AS region_name, r_comment FROM region");
    ASSERT_EQ(exit_code, 0) << output;
    EXPECT_NE(output.find("region_name"), std::string::npos) << output;  // aliased column
    EXPECT_NE(output.find("r_comment"), std::string::npos) << output;    // bare column
  }
}

/**
 * Column names for derived, nested, and joined results. Aliases (user-controlled) are asserted
 * rather than DataFusion's auto-generated names, so these stay stable across datafusion upgrades.
 */
TEST_P(FlightSqlDdlTest, ResultColumnNamesDerivedAndNested)
{
  // Aggregate results carry a real function-derived name (e.g. "count(*)").
  assert_sql_output_contains("SELECT COUNT(*) FROM region", "count");

  // An aliased expression uses the alias as its header.
  assert_sql_output_contains("SELECT r_regionkey + 1 AS incremented FROM region", "incremented");

  // A subquery's outer projection determines the output name; the inner name does not leak.
  assert_sql_output_contains("SELECT r_name AS outer_name FROM (SELECT r_name FROM region) AS sub",
                             "outer_name");

  // A join across two distinct tables carries the real column names from both inputs.
  {
    auto [exit_code, output] =
      sql("SELECT n_name, r_name FROM nation JOIN region ON n_regionkey = r_regionkey");
    ASSERT_EQ(exit_code, 0) << output;
    EXPECT_NE(output.find("n_name"), std::string::npos) << output;  // left-input column
    EXPECT_NE(output.find("r_name"), std::string::npos) << output;  // right-input column
  }
}

/** Verify DROP TABLE works by dropping a table, then confirming a query that uses it fails. */
TEST_P(FlightSqlDdlTest, DropTable)
{
  assert_sql_ok("CREATE TABLE tmp_drop(id INTEGER NOT NULL)");
  assert_sql_ok("DROP TABLE tmp_drop");
  assert_sql_fails("SELECT id FROM tmp_drop LIMIT 1");
}

/** Verify DROP TABLE IF EXISTS succeeds when the table exists. */
TEST_P(FlightSqlDdlTest, DropTableIfExistsPresent)
{
  assert_sql_ok("CREATE TABLE tmp_drop_test(id INTEGER NOT NULL)");
  assert_sql_ok("DROP TABLE IF EXISTS tmp_drop_test");
}

/** Verify DROP TABLE IF EXISTS succeeds even when the table doesn't exist. */
TEST_P(FlightSqlDdlTest, DropTableIfExistsAbsent)
{
  assert_sql_ok("DROP TABLE IF EXISTS nonexistent_table");
}

/** Verify CREATE OR REPLACE TABLE replaces an existing table without error. */
TEST_P(FlightSqlDdlTest, CreateOrReplaceTable)
{
  assert_sql_ok("CREATE TABLE tmp_replace(id INTEGER NOT NULL)");
  assert_sql_ok("CREATE OR REPLACE TABLE tmp_replace(id INTEGER NOT NULL, name VARCHAR NOT NULL)");
  sql("DROP TABLE IF EXISTS tmp_replace");
}

/**
 * A table created with a quoted identifier must stay queryable — for a mixed-case name,
 * a SQL reserved word, and a name containing a dot.
 *
 * Regression: quoted DDL names were once registered under a catalog key that no
 * subsequent query could resolve (`"MixedCase"` lowercased, `"a.b"` split into
 * schema.table). The `"a.b"` case also guards run_sql itself: a shell-interpolated
 * `echo "..."` would strip the quotes, leaving `a.b` which parses as schema.table.
 */
TEST_P(FlightSqlDdlTest, QuotedMixedCaseAndReservedWordNames)
{
  auto const region_dir = std::format("{}/region", g_config.data_path);
  for (std::string_view name : {R"("MixedCase")", R"("Order")", R"("a.b")"}) {
    assert_sql_ok(
      std::format("CREATE TABLE {}(r_regionkey INTEGER NOT NULL, "
                  "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL)",
                  name));
    assert_sql_ok(std::format("COPY {} FROM '{}' (FORMAT parquet)", name, region_dir));
    assert_sql_output_contains(std::format("SELECT r_regionkey FROM {} ORDER BY r_regionkey", name),
                               "5 rows");
    assert_sql_ok(std::format("DROP TABLE {}", name));
  }
}

class SharedTableLifecycleTest : public FlightSqlDdlTest {};

/** Verify recreating a shared table refreshes the task manager's cached table wrapper. */
TEST_P(SharedTableLifecycleTest, DropAndRecreateRefreshesTaskManagerCache)
{
  constexpr std::string_view table_name = "tmp_shared_recreate";
  auto const create_sql                 = std::format(
    "CREATE TABLE {}(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
                    "r_comment VARCHAR NOT NULL) WITH (storage_kind = 'shared_segment_memory')",
    table_name);
  auto const copy_sql =
    std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table_name, g_config.data_path);
  auto const query_sql = std::format("SELECT r_regionkey FROM {} ORDER BY r_regionkey", table_name);
  auto const drop_sql  = std::format("DROP TABLE {}", table_name);

  assert_sql_ok(create_sql);
  assert_sql_ok(copy_sql);
  assert_sql_output_contains(query_sql, "5 rows");
  assert_sql_ok(drop_sql);

  assert_sql_ok(create_sql);
  assert_sql_ok(copy_sql);
  assert_sql_ok(copy_sql);
  assert_sql_output_contains(query_sql, "10 rows");
  assert_sql_ok(drop_sql);
}

/** A COPY that fails before publishing anything leaves a shared-segment table usable. */
TEST_P(SharedTableLifecycleTest, FailedCopyBeforePublishingKeepsTableUsable)
{
  constexpr std::string_view table_name = "tmp_boost_failed_copy";
  assert_sql_ok(
    std::format("CREATE TABLE {}(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
                "r_comment VARCHAR NOT NULL) WITH (storage_kind = 'shared_segment_memory')",
                table_name));

  assert_sql_fails(std::format(
    "COPY {} FROM '{}/no_such_table' (FORMAT parquet)", table_name, g_config.data_path));

  assert_sql_ok(
    std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table_name, g_config.data_path));
  assert_sql_output_contains(
    std::format("SELECT r_regionkey FROM {} ORDER BY r_regionkey", table_name), "5 rows");
  assert_sql_ok(std::format("DROP TABLE {}", table_name));
}

/**
 * A COPY that fails after publishing row groups leaves the table unusable until it is dropped. The
 * input directory holds a valid region file followed by a nation file, which lacks region's
 * columns, so the COPY publishes the region rows before it fails.
 */
TEST_P(SharedTableLifecycleTest, FailedCopyAfterPublishingMakesTableUnusable)
{
  constexpr std::string_view table_name = "tmp_boost_partial_copy";
  auto const input_dir                  = std::filesystem::temp_directory_path() /
                         std::format("gqe_flight_sql_partial_copy_{}", std::random_device{}());
  std::filesystem::create_directories(input_dir);
  std::filesystem::create_symlink(std::format("{}/region/data_0.parquet", g_config.data_path),
                                  input_dir / "a_region.parquet");
  std::filesystem::create_symlink(std::format("{}/nation/data_0.parquet", g_config.data_path),
                                  input_dir / "b_nation.parquet");

  auto const create_sql = std::format(
    "CREATE TABLE {}(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
    "r_comment VARCHAR NOT NULL) WITH (storage_kind = 'shared_segment_memory')",
    table_name);
  auto const query_sql = std::format("SELECT r_regionkey FROM {} ORDER BY r_regionkey", table_name);

  assert_sql_ok(create_sql);
  auto [copy_exit_code, copy_output] =
    sql(std::format("COPY {} FROM '{}' (FORMAT parquet)", table_name, input_dir.string()));
  std::error_code error;
  std::filesystem::remove_all(input_dir, error);
  ASSERT_NE(copy_exit_code, 0) << "COPY should fail on the nation file. Output:\n" << copy_output;

  auto [query_exit_code, query_output] = sql(query_sql);
  EXPECT_NE(query_exit_code, 0) << "A query on the partial table should fail. Output:\n"
                                << query_output;
  EXPECT_NE(query_output.find("incomplete after a failed write"), std::string::npos)
    << query_output;
  assert_sql_fails(
    std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table_name, g_config.data_path));

  assert_sql_ok(std::format("DROP TABLE {}", table_name));
  assert_sql_ok(create_sql);
  assert_sql_ok(
    std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table_name, g_config.data_path));
  assert_sql_output_contains(query_sql, "5 rows");
  assert_sql_ok(std::format("DROP TABLE {}", table_name));
}

/**
 * A snapshot serialization failure after the COPY committed does not fail the COPY, and leaves no
 * loadable snapshot. Serialization fails here because it supports only compressed columns, and the
 * SharedContiguousTableSingleGPU configuration sets `compression_format` to `none`.
 */
TEST_P(SharedTableLifecycleTest, SnapshotFailureAfterCommitKeepsCopy)
{
  constexpr std::string_view table_name = "tmp_boost_snapshot_failure";
  auto const snapshot_location =
    std::filesystem::temp_directory_path() /
    std::format("gqe_flight_sql_snapshot_failure_{}", std::random_device{}());

  assert_sql_ok(
    std::format("CREATE TABLE {}(r_regionkey INTEGER NOT NULL, r_name VARCHAR NOT NULL, "
                "r_comment VARCHAR NOT NULL) WITH (storage_kind = 'shared_segment_memory')",
                table_name));
  {
    // Later tests share the server, so both settings are restored when this scope ends.
    scoped_session_option const location(g_config.client_bin,
                                         g_server.port(),
                                         "serialized_dataset_location",
                                         std::format("'{}'", snapshot_location.string()));
    ASSERT_TRUE(location.is_set()) << location.error_output();
    scoped_session_option const serialize(
      g_config.client_bin, g_server.port(), "serialize_tables_to_disk", "true");
    ASSERT_TRUE(serialize.is_set()) << serialize.error_output();

    auto [exit_code, output] =
      sql(std::format("COPY {} FROM '{}/region' (FORMAT parquet)", table_name, g_config.data_path));
    EXPECT_EQ(exit_code, 0) << "COPY should succeed although serialization failed. Output:\n"
                            << output;
  }

  assert_sql_output_contains(
    std::format("SELECT r_regionkey FROM {} ORDER BY r_regionkey", table_name), "5 rows");

  std::error_code error;
  for (auto const& entry :
       std::filesystem::recursive_directory_iterator(snapshot_location, error)) {
    EXPECT_NE(entry.path().filename(), "_COMPLETE")
      << "A failed serialization left a completion marker at " << entry.path();
  }
  std::filesystem::remove_all(snapshot_location, error);
  assert_sql_ok(std::format("DROP TABLE {}", table_name));
}

/**
 * Query an empty table (created but never loaded). Should return empty result, not error.
 */
TEST_P(FlightSqlDdlTest, EmptyTableQuery)
{
  assert_sql_ok("CREATE TABLE tmp_empty(id INTEGER NOT NULL, val BIGINT NOT NULL)");
  auto [exit_code, output] = sql("SELECT id, val FROM tmp_empty");
  EXPECT_EQ(exit_code, 0) << "Query on empty table should succeed. Output:\n" << output;
  sql("DROP TABLE IF EXISTS tmp_empty");
}

/**
 * COPY data into the same table twice. The row count should double and queries
 * should return the combined data from both loads.
 */
TEST_P(FlightSqlDdlTest, MultipleWritesToSameTable)
{
  assert_sql_ok(
    "CREATE TABLE tmp_region2(r_regionkey INTEGER NOT NULL, "
    "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL)");

  auto* data_path = g_config.data_path;
  ASSERT_NE(data_path, nullptr);
  auto copy_sql = std::format("COPY tmp_region2 FROM '{}/region' (FORMAT parquet)", data_path);

  assert_sql_ok(copy_sql);
  assert_sql_ok(copy_sql);

  // The TPC-H region table has exactly 5 rows at every scale factor (one per
  // region: AFRICA, AMERICA, ASIA, EUROPE, MIDDLE EAST).  Two COPYs should
  // therefore produce 2 × 5 = 10 rows.
  // Use a SELECT that scans actual data (COUNT(*) relies on statistics which may be stale).
  assert_sql_output_contains("SELECT r_regionkey FROM tmp_region2 ORDER BY r_regionkey", "10 rows");

  sql("DROP TABLE IF EXISTS tmp_region2");
}

/**
 * Verify that COPY reports the correct number of rows affected.
 * The TPC-H region table has exactly 5 rows at every scale factor.
 */
TEST_P(FlightSqlDdlTest, DISABLED_CopyRowsAffected)
{
  assert_sql_ok(
    "CREATE TABLE tmp_copy_rows(r_regionkey INTEGER NOT NULL, "
    "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL)");

  auto* data_path = g_config.data_path;
  ASSERT_NE(data_path, nullptr);
  auto copy_sql = std::format("COPY tmp_copy_rows FROM '{}/region' (FORMAT parquet)", data_path);

  assert_sql_output_contains(copy_sql, "5 rows affected");

  sql("DROP TABLE IF EXISTS tmp_copy_rows");
}

/** Submit multiple queries concurrently and verify all succeed. */
TEST_P(FlightSqlDdlTest, DISABLED_ConcurrentQueries)
{
  std::vector<std::string_view> statements = {
    "SELECT r_name FROM region ORDER BY r_regionkey",
    "SELECT n_name FROM nation ORDER BY n_nationkey",
    "SELECT COUNT(s_suppkey) FROM supplier",
  };
  std::vector<std::future<std::pair<int, std::string>>> futures;

  for (auto stmt : statements) {
    futures.push_back(std::async(std::launch::async, [this, stmt]() { return sql(stmt); }));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    auto [exit_code, output] = futures[i].get();
    EXPECT_EQ(exit_code, 0) << "Concurrent query failed: " << statements[i] << "\nOutput:\n"
                            << output;
  }
}

/**
 * Verify that querying a table concurrently with dropping it does not crash the server.
 * The in-flight query may succeed (shared_ptr keeps storage alive) or fail, but must
 * not crash. A subsequent query on the dropped table should fail cleanly.
 */
TEST_P(FlightSqlDdlTest, ConcurrentDdlAndQuery)
{
  assert_sql_ok(
    "CREATE TABLE tmp_concurrent(r_regionkey INTEGER NOT NULL, "
    "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL)");
  auto* data_path = g_config.data_path;
  ASSERT_NE(data_path, nullptr);
  assert_sql_ok(std::format("COPY tmp_concurrent FROM '{}/region' (FORMAT parquet)", data_path));

  auto query_future =
    std::async(std::launch::async, [this]() { return sql("SELECT * FROM tmp_concurrent"); });
  auto drop_future =
    std::async(std::launch::async, [this]() { return sql("DROP TABLE IF EXISTS tmp_concurrent"); });

  auto [q_rc, q_out] = query_future.get();
  auto [d_rc, d_out] = drop_future.get();

  EXPECT_EQ(d_rc, 0) << "DROP TABLE should succeed. Output:\n" << d_out;

  auto [rc, output] = sql("SELECT r_name FROM region ORDER BY r_regionkey");
  EXPECT_EQ(rc, 0) << "Server should be responsive after concurrent DDL+query. Output:\n" << output;

  sql("DROP TABLE IF EXISTS tmp_concurrent");
}

/**
 * Verify that COUNT(*) returns the correct row count after COPY.
 * The TPC-H 'region' table has exactly 5 rows at any scale factor.
 */
TEST_P(FlightSqlDdlTest, CountStar)
{
  assert_sql_output_contains("SELECT COUNT(*) FROM region", "5");
}

// ---------------------------------------------------------------------------
// PK / UNIQUE constraint DDL tests
//
// These verify that the full round-trip
//   gqe-cli SQL parser → ddl.rs encoder → node_manager → parse_ddl_command → catalog
// succeeds without error for every supported PRIMARY KEY / UNIQUE syntax.
// No SELECT from the created tables (empty in-memory tables are not queryable yet;
// see DISABLED_EmptyTableQuery). Correctness of the uniqueness metadata fed into the
// optimizer is covered by test/optimizer/uniqueness_propagation_test.cpp.
// ---------------------------------------------------------------------------

/** Inline single-column UNIQUE constraint. */
TEST_P(FlightSqlDdlTest, CreateTableWithInlineUnique)
{
  assert_sql_ok("CREATE TABLE t_inline_unique(id BIGINT NOT NULL UNIQUE, v BIGINT NOT NULL)");
  sql("DROP TABLE IF EXISTS t_inline_unique");
}

/** Inline single-column PRIMARY KEY constraint. */
TEST_P(FlightSqlDdlTest, CreateTableWithInlinePrimaryKey)
{
  assert_sql_ok("CREATE TABLE t_inline_pk(id BIGINT PRIMARY KEY NOT NULL, v BIGINT NOT NULL)");
  sql("DROP TABLE IF EXISTS t_inline_pk");
}

/** Table-level single-column PRIMARY KEY. */
TEST_P(FlightSqlDdlTest, CreateTableWithTableLevelPrimaryKey)
{
  assert_sql_ok("CREATE TABLE t_table_pk(a BIGINT NOT NULL, b BIGINT NOT NULL, PRIMARY KEY (a))");
  sql("DROP TABLE IF EXISTS t_table_pk");
}

/** Table-level composite PRIMARY KEY (a, b). */
TEST_P(FlightSqlDdlTest, CreateTableWithCompositePrimaryKey)
{
  assert_sql_ok(
    "CREATE TABLE t_composite_pk(a BIGINT NOT NULL, b BIGINT NOT NULL, c BIGINT NOT NULL, "
    "PRIMARY KEY (a, b))");
  sql("DROP TABLE IF EXISTS t_composite_pk");
}

/** Table-level composite UNIQUE (a, b). */
TEST_P(FlightSqlDdlTest, CreateTableWithCompositeUnique)
{
  assert_sql_ok(
    "CREATE TABLE t_composite_uk(a BIGINT NOT NULL, b BIGINT NOT NULL, c BIGINT NOT NULL, "
    "UNIQUE (a, b))");
  sql("DROP TABLE IF EXISTS t_composite_uk");
}

/** CREATE OR REPLACE TABLE with a PRIMARY KEY constraint. */
TEST_P(FlightSqlDdlTest, CreateOrReplaceTableWithPrimaryKey)
{
  assert_sql_ok("CREATE TABLE t_or_replace_pk(id BIGINT NOT NULL, v BIGINT NOT NULL)");
  assert_sql_ok(
    "CREATE OR REPLACE TABLE t_or_replace_pk(id BIGINT NOT NULL PRIMARY KEY, v BIGINT NOT NULL)");
  sql("DROP TABLE IF EXISTS t_or_replace_pk");
}

/** Mixed: inline UNIQUE on one column, table-level PRIMARY KEY on another. */
TEST_P(FlightSqlDdlTest, CreateTableWithMixedInlineAndTableLevelConstraints)
{
  assert_sql_ok(
    "CREATE TABLE t_mixed_pk(a BIGINT NOT NULL, b BIGINT NOT NULL UNIQUE, c BIGINT NOT NULL, "
    "PRIMARY KEY (a))");
  sql("DROP TABLE IF EXISTS t_mixed_pk");
}
/**
 * Verify that COUNT(*) with a WHERE filter returns the correct row count.
 * The TPC-H 'region' table has exactly 1 row where r_name = 'ASIA'.
 */
TEST_P(FlightSqlDdlTest, CountStarWhereRNameAsia)
{
  // Check for "| 1" rather than bare "1": the header "Results (1 rows):" always contains "1",
  // so a bare substring check would pass even when COUNT returns 0.
  assert_sql_output_contains("SELECT COUNT(*) FROM region WHERE r_name = 'ASIA'", "| 1");
}

/**
 * Regression test for filtered COUNT(*) returning 0. The TPC-H 'region' table has r_regionkey
 * values 0..4; exactly 3 rows satisfy r_regionkey < 3. A bug in projection_pushdown zeroed the
 * FilterRel's _projection_indices when a literal-only ProjectRel (the COUNT(*) row-count proxy
 * inserted by DataFusion) sat above the FilterRel, causing cudf select([]) to drop all rows.
 */
TEST_P(FlightSqlDdlTest, CountStarWhereRRegionKeyLtThree)
{
  assert_sql_output_contains("SELECT COUNT(*) FROM region WHERE r_regionkey < 3", "| 3");
}

/**
 * Verify that COUNT(*) with GROUP BY produces the correct per-group output.
 * Each TPC-H region name is unique, so the output contains a row for 'ASIA'
 * with a count of 1; presence of 'ASIA' in the output confirms the group key
 * was projected and the query was not silently folded to an ungrouped COUNT.
 */
TEST_P(FlightSqlDdlTest, CountStarGroupByRName)
{
  assert_sql_output_contains("SELECT r_name, COUNT(*) FROM region GROUP BY r_name", "ASIA");
}

/**
 * Verify that a query mixing COUNT(*) with another aggregate over a real
 * column returns correct values. The TPC-H 'region' table has r_regionkey
 * values 0..4, so SUM(r_regionkey) = 10.
 */
TEST_P(FlightSqlDdlTest, CountStarWithSumRegionKey)
{
  assert_sql_output_contains("SELECT COUNT(*), SUM(r_regionkey) FROM region", "10");
}

// ---------------------------------------------------------------------------
// Storage kind WITH clause tests
//
// Same round-trip as the PK/UNIQUE tests above; additionally exercises the
// storage_options encoding in ddl.rs and parse_storage_kind in from_substrait.cpp.
// No SELECT from the created tables (empty in-memory tables are not queryable yet;
// see DISABLED_EmptyTableQuery).
// ---------------------------------------------------------------------------

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindSharedSegmentMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_shared(x BIGINT NOT NULL) WITH (storage_kind = 'shared_segment_memory')");
  sql("DROP TABLE IF EXISTS t_shared");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindSystemMemory)
{
  assert_sql_ok("CREATE TABLE t_system(x BIGINT NOT NULL) WITH (storage_kind = 'system_memory')");
  sql("DROP TABLE IF EXISTS t_system");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindPinnedMemory)
{
  assert_sql_ok("CREATE TABLE t_pinned(x BIGINT NOT NULL) WITH (storage_kind = 'pinned_memory')");
  sql("DROP TABLE IF EXISTS t_pinned");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindManagedMemory)
{
  assert_sql_ok("CREATE TABLE t_managed(x BIGINT NOT NULL) WITH (storage_kind = 'managed_memory')");
  sql("DROP TABLE IF EXISTS t_managed");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindDeviceMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_device(x BIGINT NOT NULL) WITH (storage_kind = 'device_memory', device_id = "
    "0)");
  sql("DROP TABLE IF EXISTS t_device");
}

/** device_memory without an explicit device_id defaults to the current CUDA device. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindDeviceMemoryDefaultDeviceId)
{
  assert_sql_ok(
    "CREATE TABLE t_device_default(x BIGINT NOT NULL) WITH (storage_kind = 'device_memory')");
  sql("DROP TABLE IF EXISTS t_device_default");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaPoolMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_pool(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'numa_pool_memory', numa_node_id = 0)");
  sql("DROP TABLE IF EXISTS t_numa_pool");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindSharedNumaPoolMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_shared_numa(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'shared_numa_pool_memory', numa_node_id = 0)");
  sql("DROP TABLE IF EXISTS t_shared_numa");
}

/** numa_pool_memory without numa_node_id auto-selects based on current device affinity. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaPoolMemoryDefaultNumaNodeId)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_pool_default(x BIGINT NOT NULL) WITH (storage_kind = 'numa_pool_memory')");
  sql("DROP TABLE IF EXISTS t_numa_pool_default");
}

/** shared_numa_pool_memory without numa_node_id auto-selects based on current device affinity. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindSharedNumaPoolMemoryDefaultNumaNodeId)
{
  assert_sql_ok(
    "CREATE TABLE t_shared_numa_default(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'shared_numa_pool_memory')");
  sql("DROP TABLE IF EXISTS t_shared_numa_default");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_numa(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'numa_memory', numa_node_set = ARRAY[0])");
  sql("DROP TABLE IF EXISTS t_numa");
}

TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaPinnedMemory)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_pin(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'numa_pinned_memory', numa_node_set = ARRAY[0])");
  sql("DROP TABLE IF EXISTS t_numa_pin");
}

/** numa_memory without an explicit numa_node_set auto-selects based on current device affinity. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaMemoryDefaultNumaNodeSet)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_default(x BIGINT NOT NULL) WITH (storage_kind = 'numa_memory')");
  sql("DROP TABLE IF EXISTS t_numa_default");
}

/** numa_pinned_memory without numa_node_set but with an explicit page_kind. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaPinnedMemoryDefaultNumaNodeSetWithPageKind)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_pin_default(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'numa_pinned_memory', page_kind = 'huge2mb')");
  sql("DROP TABLE IF EXISTS t_numa_pin_default");
}

/** numa_memory with an explicit page_kind. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindNumaMemoryWithPageKind)
{
  assert_sql_ok(
    "CREATE TABLE t_numa_page(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'numa_memory', numa_node_set = ARRAY[0], page_kind = 'huge2mb')");
  sql("DROP TABLE IF EXISTS t_numa_page");
}

/** storage_kind matching is case-insensitive. */
TEST_P(FlightSqlDdlTest, CreateTableWithStorageKindCaseInsensitive)
{
  assert_sql_ok(
    "CREATE TABLE t_case(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'Device_Memory', device_id = 0)");
  sql("DROP TABLE IF EXISTS t_case");
}

/** Parquet-backed table: register a table that reads directly from existing parquet files. */
TEST_P(FlightSqlDdlTest, CreateExternalTableStoredAsParquet)
{
  auto* data_path = g_config.data_path;
  ASSERT_NE(data_path, nullptr);
  assert_sql_ok(
    std::format("CREATE EXTERNAL TABLE region_parquet(r_regionkey INTEGER NOT NULL, "
                "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL) "
                "STORED AS PARQUET LOCATION '{}/region'",
                data_path));

  assert_sql_output_contains("SELECT r_name FROM region_parquet ORDER BY r_regionkey", "AFRICA");

  sql("DROP TABLE IF EXISTS region_parquet");
}

/** EXTERNAL is optional: STORED AS PARQUET + LOCATION works without the keyword too. */
TEST_P(FlightSqlDdlTest, StoredAsParquetWithoutExternal)
{
  auto* data_path = g_config.data_path;
  ASSERT_NE(data_path, nullptr);
  assert_sql_ok(
    std::format("CREATE TABLE region_noext(r_regionkey INTEGER NOT NULL, "
                "r_name VARCHAR NOT NULL, r_comment VARCHAR NOT NULL) "
                "STORED AS PARQUET LOCATION '{}/region'",
                data_path));

  assert_sql_output_contains("SELECT r_name FROM region_noext ORDER BY r_regionkey", "AFRICA");

  sql("DROP TABLE IF EXISTS region_noext");
}

/** Unknown storage_kind is rejected. */
TEST_P(FlightSqlDdlTest, CreateTableWithUnknownStorageKindFails)
{
  assert_sql_fails(
    "CREATE TABLE t_bad(x BIGINT NOT NULL) WITH (storage_kind = 'nonexistent_memory')");
  sql("DROP TABLE IF EXISTS t_bad");
}

/** An extra key not valid for the chosen storage_kind is rejected. */
TEST_P(FlightSqlDdlTest, CreateTableWithExtraKeyFails)
{
  assert_sql_fails(
    "CREATE TABLE t_bad(x BIGINT NOT NULL) "
    "WITH (storage_kind = 'shared_segment_memory', device_id = 0)");
  sql("DROP TABLE IF EXISTS t_bad");
}

INSTANTIATE_TEST_SUITE_P(LocalContiguousTableSingleGPU,
                         FlightSqlDdlTest,
                         ::testing::Values(kLocalContiguousTableSingleGPU));

INSTANTIATE_TEST_SUITE_P(LocalContiguousTableMultiGPU,
                         FlightSqlDdlTest,
                         ::testing::Values(kLocalContiguousTableMultiGPU));

INSTANTIATE_TEST_SUITE_P(SharedContiguousTableSingleGPU,
                         SharedTableLifecycleTest,
                         ::testing::Values(kSharedContiguousTableSingleGPU));

}  // namespace gqe_test
