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

#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

// Test code must live in the global namespace (repo guideline); bring gqe names into scope.
using namespace gqe;

TEST(TypesTest, IoEngineTypeRoundTrip)
{
  for (auto engine : {io_engine_type::io_uring, io_engine_type::psync, io_engine_type::automatic}) {
    EXPECT_EQ(from_string<io_engine_type>(to_string(engine)), engine);
  }
}

TEST(TypesTest, CompressionFormatRoundTrip)
{
  for (auto fmt : {compression_format::none,
                   compression_format::ans,
                   compression_format::lz4,
                   compression_format::snappy,
                   compression_format::gdeflate,
                   compression_format::deflate,
                   compression_format::cascaded,
                   compression_format::bitpacking,
                   compression_format::zstd,
                   compression_format::gzip,
                   compression_format::bitcomp}) {
    EXPECT_EQ(from_string<compression_format>(to_string(fmt)), fmt);
  }
}

TEST(TypesTest, DecompressionBackendRoundTrip)
{
  for (auto backend :
       {decompression_backend::auto_, decompression_backend::sm, decompression_backend::de}) {
    EXPECT_EQ(from_string<decompression_backend>(to_string(backend)), backend);
  }
}

TEST(TypesTest, DecimalTypeRoundTrip)
{
  for (auto dt : {decimal_type::double_, decimal_type::decimal}) {
    EXPECT_EQ(from_string<decimal_type>(to_string(dt)), dt);
  }
}

TEST(TypesTest, ExecutionBackendTypeRoundTrip)
{
  EXPECT_EQ(to_string(execution_backend_type::legacy_task_graph), "legacy_task_graph");
  EXPECT_EQ(to_string(execution_backend_type::qep_task_graph), "qep_task_graph");

  for (auto backend :
       {execution_backend_type::legacy_task_graph, execution_backend_type::qep_task_graph}) {
    EXPECT_EQ(from_string<execution_backend_type>(to_string(backend)), backend);
  }

  EXPECT_EQ(from_string<execution_backend_type>("QEP_TASK_GRAPH"),
            execution_backend_type::qep_task_graph);
  EXPECT_THROW(from_string<execution_backend_type>("unsupported"), std::invalid_argument);
}

TEST(TypesTest, StorageKindIsFileBacked)
{
  // parquet_file is the only file-backed kind.
  EXPECT_TRUE(storage_kind::is_file_backed(storage_kind::parquet_file{{"f.parquet"}}));

  // Every memory-backed kind is not file-backed.
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::system_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_memory{cpu_set{}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::pinned_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::device_memory{rmm::cuda_device_id{0}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::managed_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_pinned_memory{cpu_set{}}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::shared_segment_memory{}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::numa_pool_memory{0}));
  EXPECT_FALSE(storage_kind::is_file_backed(storage_kind::shared_numa_pool_memory{0}));
}

/**
 * @brief Row-group counts must sum when an aggregate is folded in, not count the append.
 *
 * Callers append both a single written row group and an already-aggregated
 * `table_statistics`. Incrementing by one discards the aggregate's own count and leaves
 * `num_compressed_row_groups` greater than `num_row_groups`.
 */
TEST(TableStatisticsTest, AppendSumsRowGroupsWhenFoldingAnAggregate)
{
  auto const types   = std::array{cudf::data_type{cudf::type_id::INT32}};
  auto one_row_group = [&](std::int64_t rows) {
    table_statistics s(rows, types);
    s.num_row_groups = 1;
    return s;
  };

  // Leaf: three row groups appended one at a time.
  table_statistics aggregate(0, types);
  for (int i = 0; i < 3; ++i) {
    aggregate.append_table_statistics(one_row_group(10));
  }
  EXPECT_EQ(aggregate.num_row_groups, 3u);
  EXPECT_EQ(aggregate.num_rows, 30);

  // Aggregation: folding that aggregate in must carry all three, not add one.
  table_statistics catalog_stats(0, types);
  catalog_stats.append_table_statistics(aggregate);
  EXPECT_EQ(catalog_stats.num_row_groups, 3u)
    << "folding an aggregate must sum its row groups, not count the append";
  EXPECT_EQ(catalog_stats.num_rows, 30);

  // Appending an empty aggregate must be a no-op rather than adding a phantom row group.
  catalog_stats.append_table_statistics(table_statistics(0, types));
  EXPECT_EQ(catalog_stats.num_row_groups, 3u);
  EXPECT_EQ(catalog_stats.num_rows, 30);
}

/**
 * @brief Appended table statistics must match the accumulated columns.
 *
 * Statistics carrying zero columns contribute their rows and row groups without a
 * per-column merge. A column-count or column-identity mismatch throws instead of merging
 * misaligned columns.
 */
TEST(TypesTest, AppendTableStatisticsGuardsColumnCounts)
{
  std::vector<cudf::data_type> const column_types = {cudf::data_type(cudf::type_id::INT32),
                                                     cudf::data_type(cudf::type_id::STRING)};
  table_statistics stats(100, column_types);

  table_statistics empty_stats;
  empty_stats.num_rows = 0;
  stats.append_table_statistics(empty_stats);
  EXPECT_EQ(stats.num_columns, 2u);
  EXPECT_EQ(stats.num_row_groups, 0u);
  EXPECT_TRUE(std::holds_alternative<fixed_width_compression_statistics>(
    stats.column_stats[0].compression_stats));
  EXPECT_TRUE(
    std::holds_alternative<string_compression_statistics>(stats.column_stats[1].compression_stats));

  // A rejected append must leave the accumulator as it was, so capture it to compare against.
  auto const rejected_rows       = stats.num_rows;
  auto const rejected_row_groups = stats.num_row_groups;
  auto const rejected_stats      = stats.column_stats;

  std::vector<cudf::data_type> const single_column = {cudf::data_type(cudf::type_id::INT32)};
  table_statistics mismatched(50, single_column);
  EXPECT_THROW(stats.append_table_statistics(mismatched), std::logic_error);
  EXPECT_EQ(stats.num_rows, rejected_rows);
  EXPECT_EQ(stats.num_row_groups, rejected_row_groups);
  EXPECT_EQ(stats.column_stats[0].column_id, rejected_stats[0].column_id);
  EXPECT_EQ(stats.column_stats[1].column_id, rejected_stats[1].column_id);

  table_statistics permuted(50, column_types);
  std::swap(permuted.column_stats[0].column_id, permuted.column_stats[1].column_id);
  EXPECT_THROW(stats.append_table_statistics(permuted), std::logic_error);
  EXPECT_EQ(stats.num_rows, rejected_rows);
  EXPECT_EQ(stats.num_row_groups, rejected_row_groups);
  EXPECT_EQ(stats.column_stats[0].column_id, rejected_stats[0].column_id);
  EXPECT_EQ(stats.column_stats[1].column_id, rejected_stats[1].column_id);
}
