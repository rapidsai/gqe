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

// Unit tests for parse_ddl_command (CreateTableExtension decoding, storage_kind parsing).

#include <gqe/logical/from_substrait.hpp>

#include <gtest/gtest.h>

#include <proto/ddl_extension.pb.h>

#include <google/protobuf/any.pb.h>
#include <substrait/algebra.pb.h>
#include <unistd.h>

#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr std::string_view kTypeUrl = "type.googleapis.com/gqe.proto.CreateTableExtension";

// Build a DdlRel with a 3-column INT64 schema and no AdvancedExtension.
substrait::DdlRel make_ddl_rel(std::string const& table_name,
                               int num_cols,
                               substrait::DdlRel::DdlOp op = substrait::DdlRel::DDL_OP_CREATE)
{
  substrait::DdlRel ddl;
  ddl.mutable_named_object()->add_names(table_name);
  ddl.set_op(op);
  auto* schema = ddl.mutable_table_schema();
  for (int i = 0; i < num_cols; ++i) {
    schema->add_names("col" + std::to_string(i));
    schema->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
      substrait::Type::NULLABILITY_REQUIRED);
  }
  return ddl;
}

// Pack a CreateTableExtension into the DdlRel's AdvancedExtension.
void attach_extension(substrait::DdlRel& ddl, gqe::proto::CreateTableExtension const& ext)
{
  std::string encoded;
  ASSERT_TRUE(ext.SerializeToString(&encoded));
  auto* enh = ddl.mutable_advanced_extension()->mutable_enhancement();
  enh->set_type_url(std::string(kTypeUrl));
  enh->set_value(encoded);
}

}  // namespace

class SubstraitDdlTest : public ::testing::Test {
 protected:
  // parse_ddl_command never dereferences the catalog; nullptr is safe here.
  gqe::substrait_parser parser{static_cast<gqe::catalog*>(nullptr)};
};

// ---------------------------------------------------------------------------
// No extension → empty unique_keys
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, NoExtensionReturnsEmptyUniqueKeys)
{
  auto ddl = make_ddl_rel("t", 3);
  auto cmd = parser.parse_ddl_command(ddl);
  EXPECT_TRUE(cmd.unique_keys.empty());
}

// ---------------------------------------------------------------------------
// Single-column unique key
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, SingleColumnUniqueKeyRoundTrips)
{
  auto ddl = make_ddl_rel("t", 3);
  gqe::proto::CreateTableExtension ext;
  ext.add_unique_keys()->add_column_indices(1);  // UNIQUE (col1)
  attach_extension(ddl, ext);

  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_EQ(cmd.unique_keys.size(), 1u);
  ASSERT_EQ(cmd.unique_keys[0].size(), 1u);
  EXPECT_EQ(cmd.unique_keys[0][0], 1u);
}

// ---------------------------------------------------------------------------
// Composite unique key
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, CompositeUniqueKeyRoundTrips)
{
  auto ddl = make_ddl_rel("t", 3);
  gqe::proto::CreateTableExtension ext;
  auto* uk = ext.add_unique_keys();
  uk->add_column_indices(0);
  uk->add_column_indices(1);
  attach_extension(ddl, ext);

  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_EQ(cmd.unique_keys.size(), 1u);
  ASSERT_EQ(cmd.unique_keys[0].size(), 2u);
  EXPECT_EQ(cmd.unique_keys[0][0], 0u);
  EXPECT_EQ(cmd.unique_keys[0][1], 1u);
}

// ---------------------------------------------------------------------------
// Multiple keys — order preserved
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, MultipleKeysOrderPreserved)
{
  auto ddl = make_ddl_rel("t", 4);
  gqe::proto::CreateTableExtension ext;
  // key 0: singleton {2}
  ext.add_unique_keys()->add_column_indices(2);
  // key 1: composite {0, 1}
  auto* uk = ext.add_unique_keys();
  uk->add_column_indices(0);
  uk->add_column_indices(1);
  attach_extension(ddl, ext);

  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_EQ(cmd.unique_keys.size(), 2u);
  ASSERT_EQ(cmd.unique_keys[0], (std::vector<std::size_t>{2u}));
  ASSERT_EQ(cmd.unique_keys[1], (std::vector<std::size_t>{0u, 1u}));
}

// ---------------------------------------------------------------------------
// Out-of-bounds column index → throws std::invalid_argument
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, IndexOutOfBoundsThrows)
{
  auto ddl = make_ddl_rel("t", 3);  // columns 0, 1, 2
  gqe::proto::CreateTableExtension ext;
  ext.add_unique_keys()->add_column_indices(99);  // column 99 doesn't exist
  attach_extension(ddl, ext);

  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Empty column_indices in a key → throws std::invalid_argument
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, EmptyColumnIndicesThrows)
{
  auto ddl = make_ddl_rel("t", 3);
  gqe::proto::CreateTableExtension ext;
  ext.add_unique_keys();  // no column_indices
  attach_extension(ddl, ext);

  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Wrong type_url → extension silently ignored
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, WrongTypeUrlIgnoredSilently)
{
  auto ddl = make_ddl_rel("t", 3);
  gqe::proto::CreateTableExtension ext;
  ext.add_unique_keys()->add_column_indices(0);
  std::string encoded;
  ASSERT_TRUE(ext.SerializeToString(&encoded));

  auto* enh = ddl.mutable_advanced_extension()->mutable_enhancement();
  enh->set_type_url("type.googleapis.com/some.other.Message");
  enh->set_value(encoded);

  auto cmd = parser.parse_ddl_command(ddl);
  EXPECT_TRUE(cmd.unique_keys.empty());
}

// ---------------------------------------------------------------------------
// Malformed Any payload → silently ignored (ParseFromString returns false)
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, MalformedPayloadIgnoredSilently)
{
  auto ddl  = make_ddl_rel("t", 3);
  auto* enh = ddl.mutable_advanced_extension()->mutable_enhancement();
  enh->set_type_url(std::string(kTypeUrl));
  enh->set_value("not-valid-protobuf-bytes\xff\xfe");

  auto cmd = parser.parse_ddl_command(ddl);
  EXPECT_TRUE(cmd.unique_keys.empty());
}

// ---------------------------------------------------------------------------
// CREATE OR REPLACE also parses extension
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, CreateOrReplaceAlsoDecodesExtension)
{
  auto ddl = make_ddl_rel("t", 2, substrait::DdlRel::DDL_OP_CREATE_OR_REPLACE);
  gqe::proto::CreateTableExtension ext;
  auto* uk = ext.add_unique_keys();
  uk->add_column_indices(0);
  uk->add_column_indices(1);
  attach_extension(ddl, ext);

  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_EQ(cmd.unique_keys.size(), 1u);
  EXPECT_EQ(cmd.unique_keys[0], (std::vector<std::size_t>{0u, 1u}));
}

// ---------------------------------------------------------------------------
// DROP TABLE: schema branch is never entered → empty unique_keys
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, DropTableNoExtensionParsed)
{
  substrait::DdlRel ddl;
  ddl.mutable_named_object()->add_names("t");
  ddl.set_op(substrait::DdlRel::DDL_OP_DROP);

  auto cmd = parser.parse_ddl_command(ddl);
  EXPECT_TRUE(cmd.unique_keys.empty());
}

// ===========================================================================
// storage_kind tests
// ===========================================================================

namespace {

gqe::proto::StorageOptionValue str_opt(std::string s)
{
  gqe::proto::StorageOptionValue v;
  v.set_string_val(std::move(s));
  return v;
}

gqe::proto::StorageOptionValue int_opt(int64_t n)
{
  gqe::proto::StorageOptionValue v;
  v.set_int_val(n);
  return v;
}

gqe::proto::StorageOptionValue int_list_opt(std::initializer_list<int64_t> ns)
{
  gqe::proto::StorageOptionValue v;
  for (auto n : ns)
    v.mutable_int_list_val()->add_values(n);
  return v;
}

// Build a DdlRel with typed storage_options packed into the extension.
substrait::DdlRel make_ddl_with_storage(
  std::map<std::string, gqe::proto::StorageOptionValue> const& storage_options)
{
  auto ddl = make_ddl_rel("t", 1);
  gqe::proto::CreateTableExtension ext;
  for (auto const& [k, v] : storage_options)
    (*ext.mutable_storage_options())[k] = v;
  attach_extension(ddl, ext);
  return ddl;
}

}  // namespace

// ---------------------------------------------------------------------------
// Default: no extension → boost_shared_memory
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, NoExtensionDefaultsToBoostSharedMemory)
{
  auto ddl = make_ddl_rel("t", 1);
  auto cmd = parser.parse_ddl_command(ddl);
  EXPECT_TRUE(std::holds_alternative<gqe::storage_kind::boost_shared_memory>(cmd.storage));
}

// ---------------------------------------------------------------------------
// device_memory — requires device_id
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, StorageKindDeviceMemory)
{
  auto ddl =
    make_ddl_with_storage({{"storage_kind", str_opt("device_memory")}, {"device_id", int_opt(2)}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::device_memory>(cmd.storage));
  EXPECT_EQ(std::get<gqe::storage_kind::device_memory>(cmd.storage).device_id.value(), 2);
}

// ---------------------------------------------------------------------------
// numa_pool_memory / shared_numa_pool_memory — require numa_node_id
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, StorageKindNumaPoolMemory)
{
  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("numa_pool_memory")}, {"numa_node_id", int_opt(1)}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::numa_pool_memory>(cmd.storage));
  EXPECT_EQ(std::get<gqe::storage_kind::numa_pool_memory>(cmd.storage).numa_node_id, 1);
}

TEST_F(SubstraitDdlTest, StorageKindSharedNumaPoolMemory)
{
  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("shared_numa_pool_memory")}, {"numa_node_id", int_opt(3)}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::shared_numa_pool_memory>(cmd.storage));
  EXPECT_EQ(std::get<gqe::storage_kind::shared_numa_pool_memory>(cmd.storage).numa_node_id, 3);
}

// ---------------------------------------------------------------------------
// numa_memory / numa_pinned_memory — require numa_node_set, optional page_kind
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, StorageKindNumaMemoryDefaultPageKind)
{
  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("numa_memory")}, {"numa_node_set", int_list_opt({0, 1, 2})}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::numa_memory>(cmd.storage));
  auto const& m = std::get<gqe::storage_kind::numa_memory>(cmd.storage);
  EXPECT_TRUE(m.numa_node_set.contains(0));
  EXPECT_TRUE(m.numa_node_set.contains(1));
  EXPECT_TRUE(m.numa_node_set.contains(2));
  EXPECT_EQ(m.page_kind, gqe::page_kind::system_default);
}

TEST_F(SubstraitDdlTest, StorageKindNumaPinnedMemoryWithPageKind)
{
  auto ddl = make_ddl_with_storage({{"storage_kind", str_opt("numa_pinned_memory")},
                                    {"numa_node_set", int_list_opt({0, 1})},
                                    {"page_kind", str_opt("huge2mb")}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::numa_pinned_memory>(cmd.storage));
  auto const& m = std::get<gqe::storage_kind::numa_pinned_memory>(cmd.storage);
  EXPECT_TRUE(m.numa_node_set.contains(0));
  EXPECT_TRUE(m.numa_node_set.contains(1));
  EXPECT_EQ(m.page_kind, gqe::page_kind::huge2mb);
}

// ---------------------------------------------------------------------------
// Case insensitivity (page_kind only; storage_kind covered by integration tests)
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, PageKindCaseInsensitive)
{
  auto ddl = make_ddl_with_storage({{"storage_kind", str_opt("numa_pinned_memory")},
                                    {"numa_node_set", int_list_opt({0})},
                                    {"page_kind", str_opt("HUGE2MB")}});
  auto cmd = parser.parse_ddl_command(ddl);
  ASSERT_TRUE(std::holds_alternative<gqe::storage_kind::numa_pinned_memory>(cmd.storage));
  EXPECT_EQ(std::get<gqe::storage_kind::numa_pinned_memory>(cmd.storage).page_kind,
            gqe::page_kind::huge2mb);
}

// ---------------------------------------------------------------------------
// Range validation for narrowed int values
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, StorageKindOverflowDeviceIdRejected)
{
  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("device_memory")},
     {"device_id", int_opt(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1)}});
  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);
}

TEST_F(SubstraitDdlTest, NumaNodeSetNegativeEntryRejected)
{
  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("numa_memory")}, {"numa_node_set", int_list_opt({0, -1})}});
  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// parquet_file storage_kind
// ---------------------------------------------------------------------------

TEST_F(SubstraitDdlTest, StorageKindParquetFileEmptyDirectoryRejected)
{
  // A unique, freshly-emptied temp directory so get_parquet_files finds no files and throws.
  // The PID keeps concurrent test processes from sharing a directory; remove_all clears any
  // stale leftovers from a previous run.
  auto const tmp_dir = std::filesystem::temp_directory_path() /
                       ("gqe_substrait_ddl_test_parquet_" + std::to_string(::getpid()));
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);

  auto ddl = make_ddl_with_storage(
    {{"storage_kind", str_opt("parquet_file")}, {"location", str_opt(tmp_dir.string())}});
  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);

  std::filesystem::remove_all(tmp_dir);
}

TEST_F(SubstraitDdlTest, StorageKindParquetFileMissingLocationRejected)
{
  auto ddl = make_ddl_with_storage({{"storage_kind", str_opt("parquet_file")}});
  EXPECT_THROW(parser.parse_ddl_command(ddl), std::invalid_argument);
}
