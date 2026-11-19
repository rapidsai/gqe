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

#include <gqe/node_manager/stats_tables.hpp>

#include <gqe/catalog.hpp>
#include <gqe/node_manager/context.hpp>
#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/types.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/types.hpp>

#include <arrow/api.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gqe::node_manager {
namespace {

cudf::data_type int32_type() { return cudf::data_type{cudf::type_id::INT32}; }
cudf::data_type string_type() { return cudf::data_type{cudf::type_id::STRING}; }

/// Build a fixed_width_compression_statistics with all seven fields set to distinct, recognizable
/// values derived from `base` so the test can assert exact field placement.
fixed_width_compression_statistics make_fixed_width(std::int64_t base)
{
  return fixed_width_compression_statistics{
    .num_compressed_row_groups           = static_cast<std::size_t>(base + 1),
    .compressed_size                     = base + 2,
    .uncompressed_size                   = base + 3,
    .primary_compressed_size             = base + 4,
    .secondary_compressed_size           = base + 5,
    .num_primary_compressed_row_groups   = static_cast<std::size_t>(base + 6),
    .num_secondary_compressed_row_groups = static_cast<std::size_t>(base + 7),
  };
}

std::shared_ptr<arrow::Int64Array> int64_col(arrow::RecordBatch const& batch, char const* name)
{
  auto col = batch.GetColumnByName(name);
  EXPECT_NE(col, nullptr) << "missing column " << name;
  return std::static_pointer_cast<arrow::Int64Array>(col);
}

std::shared_ptr<arrow::StringArray> string_col(arrow::RecordBatch const& batch, char const* name)
{
  auto col = batch.GetColumnByName(name);
  EXPECT_NE(col, nullptr) << "missing column " << name;
  return std::static_pointer_cast<arrow::StringArray>(col);
}

/// Find the row index for (table, column) in a _gqe_column_stats batch, or -1.
std::int64_t find_column_row(arrow::RecordBatch const& batch,
                             std::string const& table,
                             std::string const& column)
{
  auto names   = string_col(batch, "table_name");
  auto columns = string_col(batch, "column_name");
  for (std::int64_t i = 0; i < batch.num_rows(); ++i) {
    if (names->GetString(i) == table && columns->GetString(i) == column) { return i; }
  }
  return -1;
}

/// Find the row index for a table in a _gqe_table_stats batch, or -1.
std::int64_t find_table_row(arrow::RecordBatch const& batch, std::string const& table)
{
  auto names = string_col(batch, "table_name");
  for (std::int64_t i = 0; i < batch.num_rows(); ++i) {
    if (names->GetString(i) == table) { return i; }
  }
  return -1;
}

class StatsTablesTest : public gqe::test::BaseFixture {};

// ---------------------------------------------------------------------------
// is_stats_table
// ---------------------------------------------------------------------------

TEST_F(StatsTablesTest, IsStatsTable)
{
  EXPECT_TRUE(is_stats_table(table_stats_table_name));
  EXPECT_TRUE(is_stats_table(column_stats_table_name));
  EXPECT_FALSE(is_stats_table("region"));
  EXPECT_FALSE(is_stats_table("_gqe_other"));
}

// ---------------------------------------------------------------------------
// Catalog stats version (dirty flag)
// ---------------------------------------------------------------------------

TEST_F(StatsTablesTest, StatsVersionAdvancesOnMutations)
{
  gqe::catalog catalog{get_task_manager_ctx()};

  auto const v0 = catalog.stats_version();

  catalog.register_table(
    "t", {{"a", int32_type()}}, storage_kind::system_memory{}, partitioning_schema_kind::none{});
  auto const v_after_register = catalog.stats_version();
  EXPECT_NE(v_after_register, v0) << "register_table should advance the stats version";

  gqe::table_statistics stats(/*num_rows=*/5, std::array{int32_type()});
  catalog.append_table_statistics("t", stats);
  auto const v_after_append = catalog.stats_version();
  EXPECT_NE(v_after_append, v_after_register) << "appending statistics should advance the version";

  catalog.unregister_table("t");
  EXPECT_NE(catalog.stats_version(), v_after_append)
    << "unregister_table should advance the stats version";
}

TEST_F(StatsTablesTest, StatsVersionStableWithoutMutations)
{
  gqe::catalog catalog{get_task_manager_ctx()};
  catalog.register_table(
    "t", {{"a", int32_type()}}, storage_kind::system_memory{}, partitioning_schema_kind::none{});
  auto const v = catalog.stats_version();
  // Reads do not mutate; the version must be stable so the lazy refresh is a no-op.
  EXPECT_EQ(catalog.stats_version(), v);
  EXPECT_EQ(catalog.stats_version(), v);
}

TEST_F(StatsTablesTest, SystemManagedTableRejectsUserLifecycleAndWrites)
{
  gqe::catalog catalog{get_task_manager_ctx()};
  catalog.register_system_table("system_table",
                                {{"a", int32_type()}},
                                storage_kind::system_memory{},
                                partitioning_schema_kind::none{});

  auto const version = catalog.stats_version();
  EXPECT_TRUE(catalog.has_table("system_table"));
  EXPECT_TRUE(catalog.is_readable("system_table"));
  EXPECT_FALSE(catalog.is_writeable("system_table"));
  EXPECT_THROW(static_cast<void>(catalog.writeable_view("system_table")), std::logic_error);
  EXPECT_THROW(catalog.unregister_table("system_table"), std::logic_error);

  // Rejected user operations must not alter the entry or make a fresh stats projection necessary.
  EXPECT_TRUE(catalog.has_table("system_table"));
  EXPECT_EQ(catalog.column_names("system_table"), std::vector<std::string>{"a"});
  EXPECT_EQ(catalog.stats_version(), version);
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

// Registers one in-memory table with a fixed-width and a string column, injects known statistics,
// plus a parquet (external) table and the statistics tables themselves, then returns the catalog.
class StatsProjectionTest : public gqe::test::BaseFixture {
 protected:
  void SetUp() override
  {
    // register_stats_tables() registers the statistics tables as shared_segment_memory tables,
    // which (post-!604) open the node-manager-created shared-memory segment. There is no node
    // manager in this unit-test process, so create the segment here — mirroring
    // node_manager::context — so the registration (and the shared table metadata it constructs)
    // succeeds. This uses the production segment name because catalog::register_table opens exactly
    // that; do not run this suite concurrently with a live GQE server on the same host.
    boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
    _segment.emplace(gqe::create_only,
                     gqe::node_manager::shared_memory_name,
                     std::size_t{16} << 20,
                     gqe::segment_layout_policy_type::single_arena_unplaced);

    _catalog = std::make_unique<gqe::catalog>(get_task_manager_ctx());

    _catalog->register_table("data",
                             {{"id", int32_type()}, {"name", string_type()}},
                             storage_kind::system_memory{},
                             partitioning_schema_kind::none{});

    gqe::table_statistics stats(/*num_rows=*/100, std::array{int32_type(), string_type()});
    // Statistics for one written row group; append_table_statistics sums the field.
    stats.num_row_groups  = 1;
    stats.column_stats[0] = {.column_id = 0, .compression_stats = make_fixed_width(10)};
    stats.column_stats[1] = {
      .column_id         = 1,
      .compression_stats = string_compression_statistics{.offsets_stats = make_fixed_width(20),
                                                         .chars_stats   = make_fixed_width(30)}};
    _catalog->append_table_statistics("data", stats);

    // A file-backed table and the statistics tables must be excluded from the projection.
    _catalog->register_table("ext",
                             {{"x", int32_type()}},
                             storage_kind::parquet_file{{"/nonexistent/path"}},
                             partitioning_schema_kind::automatic{});
    register_stats_tables(*_catalog);
  }

  void TearDown() override
  {
    // Release the catalog (and its shared table metadata) before unmapping and removing the
    // segment.
    _catalog.reset();
    _segment.reset();
    boost::interprocess::shared_memory_object::remove(gqe::node_manager::shared_memory_name);
  }

  std::optional<gqe::shared_segment> _segment;
  std::unique_ptr<gqe::catalog> _catalog;
};

TEST_F(StatsProjectionTest, TableStatsExcludesFileBackedAndStatsTables)
{
  auto batch = project_table_stats(*_catalog);

  EXPECT_EQ(batch->num_rows(), 1) << "only the in-memory 'data' table should be projected";
  EXPECT_NE(find_table_row(*batch, "data"), -1);
  EXPECT_EQ(find_table_row(*batch, "ext"), -1) << "file-backed table must be excluded";
  EXPECT_EQ(find_table_row(*batch, std::string{table_stats_table_name}), -1)
    << "stats tables must be excluded";
  EXPECT_EQ(find_table_row(*batch, std::string{column_stats_table_name}), -1);
}

TEST_F(StatsProjectionTest, TableStatsValues)
{
  auto batch     = project_table_stats(*_catalog);
  auto const row = find_table_row(*batch, "data");
  ASSERT_NE(row, -1);

  EXPECT_EQ(int64_col(*batch, "num_rows")->Value(row), 100);
  EXPECT_EQ(int64_col(*batch, "num_row_groups")->Value(row), 1);  // one row group
  EXPECT_EQ(int64_col(*batch, "num_columns")->Value(row), 2);

  // Sizes are summed across all columns and (for the string column) across its buffers.
  // id: uncompressed=13; name: offsets uncompressed=23 + chars uncompressed=33 = 56. Total = 69.
  EXPECT_EQ(int64_col(*batch, "uncompressed_size")->Value(row), 13 + 23 + 33);
  // id: compressed=12; name: offsets 22 + chars 32 = 54. Total = 66.
  EXPECT_EQ(int64_col(*batch, "compressed_size")->Value(row), 12 + 22 + 32);
}

TEST_F(StatsProjectionTest, ColumnStatsFixedWidth)
{
  auto batch     = project_column_stats(*_catalog);
  auto const row = find_column_row(*batch, "data", "id");
  ASSERT_NE(row, -1);

  EXPECT_EQ(string_col(*batch, "column_type")->GetString(row), "INT32");

  // Aggregate equals the single buffer's stats (base 10 → counts/sizes 11..17).
  EXPECT_FALSE(int64_col(*batch, "num_compressed_row_groups")->IsNull(row));
  EXPECT_EQ(int64_col(*batch, "num_compressed_row_groups")->Value(row), 11);
  EXPECT_EQ(int64_col(*batch, "compressed_size")->Value(row), 12);
  EXPECT_EQ(int64_col(*batch, "uncompressed_size")->Value(row), 13);

  // Breakdown blocks are null for a fixed-width column.
  EXPECT_TRUE(int64_col(*batch, "offsets_compressed_size")->IsNull(row));
  EXPECT_TRUE(int64_col(*batch, "chars_compressed_size")->IsNull(row));
}

TEST_F(StatsProjectionTest, ColumnStatsString)
{
  auto batch     = project_column_stats(*_catalog);
  auto const row = find_column_row(*batch, "data", "name");
  ASSERT_NE(row, -1);

  EXPECT_EQ(string_col(*batch, "column_type")->GetString(row), "STRING");

  // Aggregate sizes are offsets + chars (offsets base 20 → size 22/23; chars base 30 → 32/33).
  EXPECT_EQ(int64_col(*batch, "compressed_size")->Value(row), 22 + 32);
  EXPECT_EQ(int64_col(*batch, "uncompressed_size")->Value(row), 23 + 33);

  // Aggregate row-group counts are null for a string column (authoritative counts live per-buffer).
  EXPECT_TRUE(int64_col(*batch, "num_compressed_row_groups")->IsNull(row));
  EXPECT_TRUE(int64_col(*batch, "num_primary_compressed_row_groups")->IsNull(row));
  EXPECT_TRUE(int64_col(*batch, "num_secondary_compressed_row_groups")->IsNull(row));

  // Per-buffer breakdown is populated.
  EXPECT_EQ(int64_col(*batch, "offsets_compressed_size")->Value(row), 22);
  EXPECT_EQ(int64_col(*batch, "offsets_uncompressed_size")->Value(row), 23);
  EXPECT_EQ(int64_col(*batch, "chars_compressed_size")->Value(row), 32);
  EXPECT_EQ(int64_col(*batch, "chars_uncompressed_size")->Value(row), 33);
  EXPECT_EQ(int64_col(*batch, "offsets_num_compressed_row_groups")->Value(row), 21);
  EXPECT_EQ(int64_col(*batch, "chars_num_compressed_row_groups")->Value(row), 31);
}

TEST_F(StatsProjectionTest, ColumnStatsExcludesFileBackedAndStatsTables)
{
  auto batch = project_column_stats(*_catalog);
  // Only the two columns of 'data' should appear.
  EXPECT_EQ(batch->num_rows(), 2);
  EXPECT_EQ(find_column_row(*batch, "ext", "x"), -1);
}

TEST_F(StatsProjectionTest, EmptyCatalogProducesZeroRows)
{
  gqe::catalog empty{get_task_manager_ctx()};
  register_stats_tables(empty);  // only the (excluded) stats tables exist
  EXPECT_EQ(project_table_stats(empty)->num_rows(), 0);
  EXPECT_EQ(project_column_stats(empty)->num_rows(), 0);
}

}  // namespace
}  // namespace gqe::node_manager
