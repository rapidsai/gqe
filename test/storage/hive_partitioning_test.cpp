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

#include <gqe/storage/hive_partitioning.hpp>

#include "../utility.hpp"

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/query_context.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/mr/callback_memory_resource.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<std::size_t, int64_t>> typed_partitioned_files(
  std::vector<std::pair<std::size_t, std::shared_ptr<void>>> const& files)
{
  std::vector<std::pair<std::size_t, int64_t>> out;
  out.reserve(files.size());
  for (auto const& [file_idx, key] : files) {
    out.emplace_back(file_idx, *std::static_pointer_cast<int64_t>(key));
  }
  return out;
}

}  // namespace

TEST(HivePartitioning, AllFilesNonPartitioned)
{
  auto const info = gqe::storage::detail::all_files_non_partitioned(3);

  EXPECT_THAT(info.partitioned_files, ::testing::IsEmpty());
  EXPECT_EQ(info.partition_column_idx, 0);
  EXPECT_THAT(info.non_partitioned_files, ::testing::ElementsAre(0, 1, 2));
}

TEST(HivePartitioning, FilterHiveFilesBySignedIntegralHaystack)
{
  auto const file_paths =
    std::vector<std::string>{"root/part_col=-2/data.parquet",
                             "root/part_col=+3/data.parquet",
                             "root/part_col=42abc/data.parquet",
                             "root/part_col=__HIVE_DEFAULT_PARTITION__/data.parquet",
                             "root/part_col=7/data.parquet"};

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({-2, 7, 42});
  auto haystack = make_table_from_columns(haystack_col.release());

  auto const info = gqe::storage::detail::filter_hive_files_by_haystack(
    file_paths, 1, "part_col", haystack->view());

  EXPECT_EQ(info.partition_column_idx, 1);
  EXPECT_THAT(typed_partitioned_files(info.partitioned_files),
              ::testing::ElementsAre(std::pair<std::size_t, int64_t>{0, -2},
                                     std::pair<std::size_t, int64_t>{4, 7}));
  EXPECT_THAT(info.non_partitioned_files, ::testing::ElementsAre(2));
}

TEST(HivePartitioning, FilterHiveFilesDropsNullHaystackEntries)
{
  auto const file_paths =
    std::vector<std::string>{"root/part_col=2/data.parquet", "root/part_col=3/data.parquet"};

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({2, 3}, {0, 1});
  auto haystack = make_table_from_columns(haystack_col.release());

  auto const info = gqe::storage::detail::filter_hive_files_by_haystack(
    file_paths, 0, "part_col", haystack->view());

  EXPECT_THAT(typed_partitioned_files(info.partitioned_files),
              ::testing::ElementsAre(std::pair<std::size_t, int64_t>{1, 3}));
  EXPECT_THAT(info.non_partitioned_files, ::testing::IsEmpty());
}

TEST(HivePartitioning, FilterHiveFilesRejectsNonIntegralHaystack)
{
  auto const file_paths = std::vector<std::string>{"root/part_col=1/data.parquet"};

  cudf::test::fixed_width_column_wrapper<float> haystack_col({1.0F});
  auto haystack = make_table_from_columns(haystack_col.release());

  EXPECT_THROW(std::ignore = gqe::storage::detail::filter_hive_files_by_haystack(
                 file_paths, 0, "part_col", haystack->view()),
               std::runtime_error);
}

TEST(HivePartitioning, ComputeHiveChunkSegmentsSpansFilesAndAdvancesCursor)
{
  gqe::storage::detail::hive_partitioning_state state{.partition_column_idx         = 0,
                                                      .partition_keys               = {10, 20},
                                                      .rows_per_file                = {3, 2},
                                                      .cursor_file_idx              = 0,
                                                      .cursor_rows_consumed_in_file = 0};

  auto first = gqe::storage::detail::compute_hive_chunk_segments(state, 4);
  EXPECT_THAT(first.keys, ::testing::ElementsAre(10, 20));
  EXPECT_THAT(first.rows, ::testing::ElementsAre(3, 1));
  EXPECT_EQ(state.cursor_file_idx, 1);
  EXPECT_EQ(state.cursor_rows_consumed_in_file, 1);

  auto second = gqe::storage::detail::compute_hive_chunk_segments(state, 1);
  EXPECT_THAT(second.keys, ::testing::ElementsAre(20));
  EXPECT_THAT(second.rows, ::testing::ElementsAre(1));
  EXPECT_EQ(state.cursor_file_idx, 2);
  EXPECT_EQ(state.cursor_rows_consumed_in_file, 0);
}

TEST(HivePartitioning, ComputeHiveChunkSegmentsThrowsWhenChunkExceedsRemainingRows)
{
  gqe::storage::detail::hive_partitioning_state state{.partition_column_idx         = 0,
                                                      .partition_keys               = {10},
                                                      .rows_per_file                = {1},
                                                      .cursor_file_idx              = 0,
                                                      .cursor_rows_consumed_in_file = 0};

  EXPECT_THROW(std::ignore = gqe::storage::detail::compute_hive_chunk_segments(state, 2),
               std::logic_error);
}

TEST(HivePartitioning, PruningUsesQueryStreamAndResource)
{
  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({2, 3}, {0, 1});
  auto haystack = make_table_from_columns(haystack_col.release());
  cudf::get_default_stream().sync();
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const stream_handle     = context.create_stream("hive-pruning-test");
  auto upstream                = cudf::get_current_device_resource_ref();
  std::size_t allocation_count = 0;
  rmm::mr::callback_memory_resource resource{
    [&](std::size_t bytes, cuda::stream_ref stream, void*) {
      EXPECT_EQ(stream.get(), stream_handle.ref().get());
      ++allocation_count;
      return upstream.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    },
    [&](void* pointer, std::size_t bytes, cuda::stream_ref stream, void*) {
      EXPECT_EQ(stream.get(), stream_handle.ref().get());
      upstream.deallocate(stream, pointer, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }};
  auto const info = gqe::storage::detail::filter_hive_files_by_haystack(
    {"root/part_col=2/data.parquet", "root/part_col=3/data.parquet"},
    0,
    "part_col",
    haystack->view(),
    stream_handle.ref(),
    resource);
  EXPECT_GT(allocation_count, 0);
  EXPECT_THAT(typed_partitioned_files(info.partitioned_files),
              ::testing::ElementsAre(std::pair<std::size_t, int64_t>{1, 3}));
}
