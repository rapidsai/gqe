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

#include <gqe/task_manager/remote_table_cache.hpp>

#include <gtest/gtest.h>

using namespace gqe;
using namespace gqe::task_manager;

namespace {

storage::parquet_file_descriptor make_parquet(std::string name, std::vector<std::string> paths)
{
  return {std::move(name), std::move(paths)};
}

}  // namespace

class RemoteTableCacheTest : public ::testing::Test {
 protected:
  // Parquet descriptors don't require a task_manager_context.
  remote_table_cache cache{nullptr};
};

TEST_F(RemoteTableCacheTest, UpdateAddsNewTable)
{
  cache.update({make_parquet("t1", {"/data/t1.parquet"})});

  auto result = cache.get_table("t1");
  EXPECT_NE(result, nullptr);
}

TEST_F(RemoteTableCacheTest, SameDescriptorReturnsCachedTable)
{
  cache.update({make_parquet("t1", {"/data/t1.parquet"})});
  auto first = cache.get_table("t1");

  // Update with the identical descriptor.
  cache.update({make_parquet("t1", {"/data/t1.parquet"})});
  auto second = cache.get_table("t1");

  EXPECT_EQ(first, second);
}

TEST_F(RemoteTableCacheTest, ChangedDescriptorRefreshesTable)
{
  cache.update({make_parquet("t1", {"/data/t1_v1.parquet"})});
  auto first = cache.get_table("t1");

  // Update with a different descriptor for the same table name.
  cache.update({make_parquet("t1", {"/data/t1_v2.parquet"})});
  auto second = cache.get_table("t1");

  EXPECT_NE(first, second);
}

TEST_F(RemoteTableCacheTest, MultipleTables)
{
  cache.update({
    make_parquet("t1", {"/data/t1.parquet"}),
    make_parquet("t2", {"/data/t2.parquet"}),
  });

  auto r1 = cache.get_table("t1");
  auto r2 = cache.get_table("t2");
  EXPECT_NE(r1, nullptr);
  EXPECT_NE(r2, nullptr);
  EXPECT_NE(r1, r2);
}

TEST_F(RemoteTableCacheTest, UnknownTableThrows)
{
  cache.update({make_parquet("t1", {"/data/t1.parquet"})});
  EXPECT_THROW(std::ignore = cache.get_table("nonexistent"), std::logic_error);
}

TEST_F(RemoteTableCacheTest, OnlyChangedTableIsRefreshed)
{
  cache.update({
    make_parquet("t1", {"/data/t1.parquet"}),
    make_parquet("t2", {"/data/t2.parquet"}),
  });
  auto t1_first = cache.get_table("t1");
  auto t2_first = cache.get_table("t2");

  // Change t2 but leave t1 unchanged.
  cache.update({
    make_parquet("t1", {"/data/t1.parquet"}),
    make_parquet("t2", {"/data/t2_v2.parquet"}),
  });
  auto t1_second = cache.get_table("t1");
  auto t2_second = cache.get_table("t2");

  EXPECT_EQ(t1_first, t1_second);
  EXPECT_NE(t2_first, t2_second);
}
