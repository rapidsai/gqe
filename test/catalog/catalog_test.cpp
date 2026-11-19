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

#include <gqe/catalog.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/serialization.hpp>
#include <gqe_test/base_fixture.hpp>

#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace gqe {
namespace {

cudf::data_type int32_type() { return cudf::data_type{cudf::type_id::INT32}; }

// Removes its directory on scope exit, even if an assertion fails mid-test.
struct temp_dir_guard {
  std::filesystem::path path;
  ~temp_dir_guard()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

class CatalogTest : public test::BaseFixture {
 protected:
  void SetUp() override
  {
    // Serialization supports only compressed-sliced columns.
    auto params                         = make_optimization_parameters(false);
    params.compression_format           = compression_format::lz4;
    params.secondary_compression_format = compression_format::none;
    params.use_cpu_compression          = true;
    params.compression_chunk_size       = 64 * 1024;
    initialize_contexts(params);
    _catalog = std::make_unique<catalog>(get_task_manager_ctx());

    _snapshot_root = std::filesystem::temp_directory_path() /
                     ("gqe_catalog_test_" + std::to_string(std::random_device{}()));
    _guard.path = _snapshot_root;
  }

  /** Register an empty one-column pinned in-memory table. */
  void register_table(std::string const& table_name)
  {
    _catalog->register_table(table_name,
                             {{"id", int32_type()}},
                             storage_kind::pinned_memory{},
                             partitioning_schema_kind::none{});
  }

  /** Append a row group holding @p values to an in-memory table, compressed if requested. */
  void append_row_group(std::string const& table_name,
                        std::vector<int32_t> const& values,
                        bool is_compressed = true)
  {
    auto const stream  = cudf::get_default_stream();
    auto const mr      = get_task_manager_ctx()->get_table_memory_resource(memory_kind::pinned{});
    auto const& params = get_task_manager_ctx()->get_optimization_parameters();
    rmm::device_buffer data(values.data(), values.size() * sizeof(int32_t), stream, mr);
    cudf::column column(
      int32_type(), static_cast<cudf::size_type>(values.size()), std::move(data), {}, 0);

    std::vector<std::unique_ptr<storage::column_base>> columns;
    if (is_compressed) {
      auto const config = compression::configuration{
        .primary_compression_format            = params.compression_format,
        .secondary_compression_format          = params.secondary_compression_format,
        .compression_chunk_size                = params.compression_chunk_size,
        .compression_ratio_threshold           = params.compression_ratio_threshold,
        .secondary_compression_ratio_threshold = params.secondary_compression_ratio_threshold,
        .secondary_compression_multiplier_threshold =
          params.secondary_compression_multiplier_threshold,
        .use_cpu_compression = params.use_cpu_compression,
        .compression_level   = params.compression_level,
        .decompress_backend  = params.decompression_backend};
      columns.push_back(
        std::make_unique<storage::compressed_sliced_column>(std::move(column),
                                                            params.zone_map_partition_size,
                                                            memory_kind::pinned{},
                                                            config,
                                                            stream,
                                                            mr));
    } else {
      columns.push_back(std::make_unique<storage::contiguous_column>(std::move(column)));
    }

    auto table =
      std::dynamic_pointer_cast<storage::in_memory_table>(_catalog->table_storage(table_name));
    ASSERT_NE(table, nullptr);
    table->get_row_group_appender()(storage::row_group(std::move(columns)));
    stream.sync();
  }

  /** Number of row groups of an in-memory table. */
  [[nodiscard]] std::size_t num_row_groups(std::string const& table_name) const
  {
    auto table = std::dynamic_pointer_cast<storage::in_memory_table_interface>(
      _catalog->table_storage(table_name));
    return table->num_row_groups();
  }

  [[nodiscard]] std::filesystem::path marker() const
  {
    return utility::serialized_snapshot_marker_path(_snapshot_root);
  }

  std::unique_ptr<catalog> _catalog;
  std::filesystem::path _snapshot_root;
  temp_dir_guard _guard;
};

// ---------------------------------------------------------------------------
// Snapshot completion marker
// ---------------------------------------------------------------------------

TEST_F(CatalogTest, SerializeWritesMarkerAndRoundTrips)
{
  register_table("source");
  append_row_group("source", {1, 2, 3, 4});
  append_row_group("source", {5, 6, 7});

  _catalog->serialize_table("source", _snapshot_root.string());
  EXPECT_TRUE(std::filesystem::is_regular_file(marker()));

  register_table("restored");
  _catalog->deserialize_table("restored", _snapshot_root.string());
  EXPECT_EQ(num_row_groups("restored"), 2U);
  EXPECT_EQ(_catalog->statistics("restored")->statistics().num_rows, 7);
}

TEST_F(CatalogTest, SerializeReplacesPreviousSnapshot)
{
  register_table("large");
  append_row_group("large", {1, 2});
  append_row_group("large", {3, 4});
  _catalog->serialize_table("large", _snapshot_root.string());

  // A snapshot with fewer row groups must not pick up the previous snapshot's extra row group.
  register_table("small");
  append_row_group("small", {5});
  _catalog->serialize_table("small", _snapshot_root.string());

  register_table("restored");
  _catalog->deserialize_table("restored", _snapshot_root.string());
  EXPECT_EQ(num_row_groups("restored"), 1U);
  EXPECT_EQ(_catalog->statistics("restored")->statistics().num_rows, 1);
}

TEST_F(CatalogTest, DeserializeRefusesSnapshotWithoutMarker)
{
  register_table("source");
  append_row_group("source", {1, 2, 3});
  _catalog->serialize_table("source", _snapshot_root.string());
  ASSERT_TRUE(std::filesystem::remove(marker()));

  register_table("restored");
  EXPECT_THROW(_catalog->deserialize_table("restored", _snapshot_root.string()),
               std::invalid_argument);
  EXPECT_EQ(num_row_groups("restored"), 0U);
}

TEST_F(CatalogTest, FailedSerializationLeavesNoSnapshot)
{
  register_table("source");
  append_row_group("source", {1, 2, 3});
  _catalog->serialize_table("source", _snapshot_root.string());

  // Serialization rejects uncompressed columns after it has removed the previous snapshot's marker.
  register_table("uncompressed");
  append_row_group("uncompressed", {4, 5}, /*is_compressed=*/false);
  EXPECT_THROW(_catalog->serialize_table("uncompressed", _snapshot_root.string()),
               std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(_snapshot_root));

  register_table("restored");
  EXPECT_THROW(_catalog->deserialize_table("restored", _snapshot_root.string()),
               std::invalid_argument);
}

TEST_F(CatalogTest, MarkerRemovalFailureKeepsPreviousSnapshot)
{
  if (::geteuid() == 0) { GTEST_SKIP() << "root ignores the directory permissions this test uses"; }

  register_table("source");
  append_row_group("source", {1, 2});
  append_row_group("source", {3});
  _catalog->serialize_table("source", _snapshot_root.string());

  register_table("replacement");
  append_row_group("replacement", {4});

  // Without write permission on the snapshot root, the marker cannot be removed.
  auto const permissions = std::filesystem::status(_snapshot_root).permissions();
  std::filesystem::permissions(_snapshot_root,
                               std::filesystem::perms::owner_write |
                                 std::filesystem::perms::group_write |
                                 std::filesystem::perms::others_write,
                               std::filesystem::perm_options::remove);
  EXPECT_THROW(_catalog->serialize_table("replacement", _snapshot_root.string()),
               std::runtime_error);
  std::filesystem::permissions(_snapshot_root, permissions);

  register_table("restored");
  _catalog->deserialize_table("restored", _snapshot_root.string());
  EXPECT_EQ(num_row_groups("restored"), 2U);
  EXPECT_EQ(_catalog->statistics("restored")->statistics().num_rows, 3);
}

// ---------------------------------------------------------------------------
// Write commit point and unusable tables
// ---------------------------------------------------------------------------

TEST_F(CatalogTest, RejectedStatisticsAppendLeavesStatisticsUnchanged)
{
  register_table("t");
  table_statistics stats(/*num_rows=*/5, std::array{int32_type()});
  _catalog->append_table_statistics("t", stats);
  auto const version = _catalog->stats_version();

  table_statistics mismatched(/*num_rows=*/3, std::array{int32_type(), int32_type()});
  EXPECT_THROW(_catalog->append_table_statistics("t", mismatched), std::logic_error);
  EXPECT_EQ(_catalog->statistics("t")->statistics().num_rows, 5);
  EXPECT_EQ(_catalog->stats_version(), version);
}

TEST_F(CatalogTest, StatisticsAppendFailingMidMergeLeavesStatisticsUnchanged)
{
  _catalog->register_table("t",
                           {{"a", int32_type()}, {"b", int32_type()}},
                           storage_kind::pinned_memory{},
                           partitioning_schema_kind::none{});
  table_statistics stats(/*num_rows=*/5, std::array{int32_type(), int32_type()});
  stats.num_row_groups = 1;
  std::get<fixed_width_compression_statistics>(stats.column_stats[0].compression_stats)
    .uncompressed_size = 100;
  _catalog->append_table_statistics("t", stats);
  auto const version = _catalog->stats_version();

  // The column counts and IDs match, so the merge starts. It updates the row counts and the first
  // column before the second column's string statistics fail to merge into fixed-width ones.
  table_statistics mismatched(/*num_rows=*/3,
                              std::array{int32_type(), cudf::data_type{cudf::type_id::STRING}});
  mismatched.num_row_groups = 1;
  std::get<fixed_width_compression_statistics>(mismatched.column_stats[0].compression_stats)
    .uncompressed_size = 7;
  EXPECT_THROW(_catalog->append_table_statistics("t", mismatched), std::logic_error);

  auto const& current = _catalog->statistics("t")->statistics();
  EXPECT_EQ(current.num_rows, 5);
  EXPECT_EQ(current.num_row_groups, 1U);
  EXPECT_EQ(std::get<fixed_width_compression_statistics>(current.column_stats[0].compression_stats)
              .uncompressed_size,
            100);
  EXPECT_EQ(_catalog->stats_version(), version);
}

TEST_F(CatalogTest, UnusableFlagLastsUntilTableIsDropped)
{
  register_table("t");
  EXPECT_FALSE(_catalog->is_unusable("t"));

  _catalog->mark_unusable("t");
  EXPECT_TRUE(_catalog->is_unusable("t"));

  _catalog->unregister_table("t");
  register_table("t");
  EXPECT_FALSE(_catalog->is_unusable("t"));

  EXPECT_THROW(_catalog->mark_unusable("missing"), std::logic_error);
  EXPECT_THROW(static_cast<void>(_catalog->is_unusable("missing")), std::logic_error);
}

}  // namespace
}  // namespace gqe
