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

#include "../utility.hpp"

#include <gqe/qep/shapes/deferred_column.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/storage/parquet_footer_cache.hpp>
#include <gqe/storage/parquet_reference_iterate.hpp>
#include <gqe/storage/parquet_resolve.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

namespace {

[[nodiscard]] gqe::storage::parquet_reference const* as_parquet_reference(
  gqe::qep::state_container const& container)
{
  if (container.empty() || container.front() == nullptr) { return nullptr; }
  auto const* slot = std::get_if<gqe::qep::state_kind::reference>(container.front().get());
  if (slot == nullptr) { return nullptr; }
  return dynamic_cast<gqe::storage::parquet_reference const*>(slot->state().get());
}

}  // namespace

class ParquetReferenceReadTest : public gqe::test::BaseFixture {
 protected:
  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  static std::vector<cudf::data_type> int64_types(std::size_t count)
  {
    return std::vector<cudf::data_type>(count, cudf::data_type{cudf::type_id::INT64});
  }

  std::shared_ptr<gqe::storage::parquet_footer_cache> _footer_cache =
    std::make_shared<gqe::storage::parquet_footer_cache>();
};

TEST_F(ParquetReferenceReadTest, EmitsReferencesAndFetchMaterializes)
{
  cudf::test::fixed_width_column_wrapper<int64_t> file_0_col({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int64_t> file_1_col({4, 5});
  auto file_0 = make_table_from_columns(file_0_col.release());
  auto file_1 = make_table_from_columns(file_1_col.release());

  auto const path_0  = temp_env->get_temp_filepath("ref_read_0.parquet");
  auto const path_1  = temp_env->get_temp_filepath("ref_read_1.parquet");
  auto const columns = std::vector<std::string>{"col_0"};
  write_table_to_file(file_0->view(), columns, path_0);
  write_table_to_file(file_1->view(), columns, path_1);

  gqe::storage::parquet_reference_iterate_task iterate{
    {path_0, path_1}, columns, int64_types(1), std::nullopt, _footer_cache};
  gqe::storage::parquet_resolve_task resolve{_footer_cache, columns, int64_types(1)};

  auto iterator = iterate.initialize({}, ctx_ref());
  EXPECT_EQ(_footer_cache->size(), 2) << "initialize must populate the shared footer cache";

  std::vector<std::pair<std::string, std::vector<int64_t>>> const expected = {{path_0, {1, 2, 3}},
                                                                              {path_1, {4, 5}}};
  for (auto const& [expected_path, expected_values] : expected) {
    auto reference = iterate.next(iterator, ctx_ref());
    ASSERT_TRUE(reference.has_value());
    auto const* ref = as_parquet_reference(*reference);
    ASSERT_NE(ref, nullptr);
    EXPECT_EQ(ref->file_path(), expected_path);
    EXPECT_EQ(gqe::qep::get_row_count(*reference),
              static_cast<cudf::size_type>(expected_values.size()));

    auto deferred = resolve.next(*reference, ctx_ref());
    ASSERT_TRUE(deferred.has_value());
    auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
    auto actual       = gqe::qep::to_table_view(materialized);
    ASSERT_TRUE(actual.has_value());
    cudf::test::fixed_width_column_wrapper<int64_t> expected_col(expected_values.begin(),
                                                                 expected_values.end());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view{{expected_col}}, *actual);
  }
  EXPECT_FALSE(iterate.next(iterator, ctx_ref()).has_value());
}

TEST_F(ParquetReferenceReadTest, MaterializeTwiceYieldsIndependentColumns)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({7, 8, 9});
  auto table = make_table_from_columns(col.release());

  auto const path = temp_env->get_temp_filepath("ref_read_cached.parquet");
  write_table_to_file(table->view(), {"col_0"}, path);

  gqe::storage::parquet_reference_iterate_task iterate{
    {path}, {"col_0"}, int64_types(1), std::nullopt, _footer_cache};
  gqe::storage::parquet_resolve_task resolve{_footer_cache, {"col_0"}, int64_types(1)};

  auto iterator  = iterate.initialize({}, ctx_ref());
  auto reference = iterate.next(iterator, ctx_ref());
  ASSERT_TRUE(reference.has_value());
  auto deferred = resolve.next(*reference, ctx_ref());
  ASSERT_TRUE(deferred.has_value());
  auto first  = gqe::qep::materialize_deferred_columns(*deferred);
  auto second = gqe::qep::materialize_deferred_columns(*deferred);
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_NE(first[i].get(), second[i].get()) << "the source retains no decoded columns";
  }
  EXPECT_EQ(gqe::qep::get_row_count(first), gqe::qep::get_row_count(second));
}

TEST_F(ParquetReferenceReadTest, FetchWithFreshCacheResolvesStolenReference)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({11, 12});
  auto table = make_table_from_columns(col.release());

  auto const path = temp_env->get_temp_filepath("ref_read_stolen.parquet");
  write_table_to_file(table->view(), {"col_0"}, path);

  gqe::storage::parquet_reference_iterate_task iterate{
    {path}, {"col_0"}, int64_types(1), std::nullopt, _footer_cache};
  auto iterator  = iterate.initialize({}, ctx_ref());
  auto reference = iterate.next(iterator, ctx_ref());
  ASSERT_TRUE(reference.has_value());

  // A stealing rank has its own cache; the resolve task populates it through the same miss path.
  auto thief_cache = std::make_shared<gqe::storage::parquet_footer_cache>();
  gqe::storage::parquet_resolve_task resolve{thief_cache, {"col_0"}, int64_types(1)};
  auto deferred = resolve.next(*reference, ctx_ref());
  ASSERT_TRUE(deferred.has_value());
  EXPECT_EQ(thief_cache->size(), 1);

  auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
  EXPECT_EQ(gqe::qep::get_row_count(materialized), 2);
}

TEST_F(ParquetReferenceReadTest, ResolveRejectsColumnsMissingFromTheFile)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2});
  auto table = make_table_from_columns(col.release());

  auto const path = temp_env->get_temp_filepath("ref_read_missing_column.parquet");
  write_table_to_file(table->view(), {"col_0"}, path);

  gqe::storage::parquet_reference_iterate_task iterate{
    {path}, {"col_0"}, int64_types(1), std::nullopt, _footer_cache};
  auto iterator  = iterate.initialize({}, ctx_ref());
  auto reference = iterate.next(iterator, ctx_ref());
  ASSERT_TRUE(reference.has_value());

  gqe::storage::parquet_resolve_task resolve{_footer_cache, {"not_a_column"}, int64_types(1)};
  EXPECT_THROW(std::ignore = resolve.next(*reference, ctx_ref()), std::invalid_argument);
}

TEST_F(ParquetReferenceReadTest, CountStarEmitsTotalRowCountWithoutReferences)
{
  cudf::test::fixed_width_column_wrapper<int64_t> file_0_col({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int64_t> file_1_col({4, 5});
  auto file_0 = make_table_from_columns(file_0_col.release());
  auto file_1 = make_table_from_columns(file_1_col.release());

  auto const path_0 = temp_env->get_temp_filepath("ref_count_0.parquet");
  auto const path_1 = temp_env->get_temp_filepath("ref_count_1.parquet");
  write_table_to_file(file_0->view(), {"col_0"}, path_0);
  write_table_to_file(file_1->view(), {"col_0"}, path_1);

  gqe::storage::parquet_reference_iterate_task task{
    {path_0, path_1}, {}, {}, std::nullopt, _footer_cache};

  auto iterator = task.initialize({}, ctx_ref());
  auto chunk    = task.next(iterator, ctx_ref());
  ASSERT_TRUE(chunk.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 5);
  EXPECT_FALSE(task.next(iterator, ctx_ref()).has_value());
  EXPECT_EQ(_footer_cache->size(), 2);
}

TEST_F(ParquetReferenceReadTest, EmptyFileListEmitsOneEmptyReferenceAndEmptyTable)
{
  gqe::storage::parquet_reference_iterate_task iterate{
    {}, {"col_0"}, int64_types(1), std::nullopt, _footer_cache};
  gqe::storage::parquet_resolve_task resolve{_footer_cache, {"col_0"}, int64_types(1)};

  auto iterator  = iterate.initialize({}, ctx_ref());
  auto reference = iterate.next(iterator, ctx_ref());
  ASSERT_TRUE(reference.has_value());
  auto const* ref = as_parquet_reference(*reference);
  ASSERT_NE(ref, nullptr);
  EXPECT_TRUE(ref->is_empty());
  EXPECT_FALSE(iterate.next(iterator, ctx_ref()).has_value());

  auto empty_table = resolve.next(*reference, ctx_ref());
  ASSERT_TRUE(empty_table.has_value());
  auto view = gqe::qep::to_table_view(*empty_table);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->num_columns(), 1);
  EXPECT_EQ(view->num_rows(), 0);
}

TEST_F(ParquetReferenceReadTest, HivePruningMintsPerFileKeysAndSynthesizesColumn)
{
  cudf::test::fixed_width_column_wrapper<int64_t> file_a_col({1, 2});
  cudf::test::fixed_width_column_wrapper<int64_t> file_b_col({3, 4, 5});
  auto file_a = make_table_from_columns(file_a_col.release());
  auto file_b = make_table_from_columns(file_b_col.release());

  auto const dir_a = temp_env->get_temp_dir() + "part=7";
  auto const dir_b = temp_env->get_temp_dir() + "part=9";
  std::filesystem::create_directories(dir_a);
  std::filesystem::create_directories(dir_b);
  auto const path_a = dir_a + "/data.parquet";
  auto const path_b = dir_b + "/data.parquet";
  write_table_to_file(file_a->view(), {"col_0"}, path_a);
  write_table_to_file(file_b->view(), {"col_0"}, path_b);

  auto const columns = std::vector<std::string>{"col_0", "part"};
  auto const types   = int64_types(2);
  auto const pruner  = gqe::storage::hive_pruner_options{.partition_column_idx = 1};

  gqe::storage::parquet_reference_iterate_task iterate{
    {path_a, path_b}, columns, types, pruner, _footer_cache};
  gqe::storage::parquet_resolve_task resolve{_footer_cache, columns, types, pruner};

  // Haystack keeps only partition 9.
  cudf::test::fixed_width_column_wrapper<int64_t> haystack_col({9});
  auto const haystack =
    gqe::qep::state_container_builder().add_state(cudf::table_view{{haystack_col}}).build();

  auto split_reader = iterate.split({0, 2});
  auto clone        = gqe::qep::clone_qep_task_as<gqe::qep::iterate_task>(*split_reader);
  for (auto const* source : {static_cast<gqe::qep::iterate_task const*>(&iterate),
                             static_cast<gqe::qep::iterate_task const*>(clone.get())}) {
    auto iterator  = source->initialize(haystack, ctx_ref());
    auto reference = source->next(iterator, ctx_ref());
    ASSERT_TRUE(reference.has_value());
    auto const* ref = as_parquet_reference(*reference);
    ASSERT_NE(ref, nullptr);
    EXPECT_EQ(ref->file_path(), path_b);
    EXPECT_EQ(ref->hive_partition_key(), std::optional<std::int64_t>{9});
    EXPECT_FALSE(source->next(iterator, ctx_ref()).has_value());

    auto deferred = resolve.next(*reference, ctx_ref());
    ASSERT_TRUE(deferred.has_value());
    auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
    auto actual       = gqe::qep::to_table_view(materialized);
    ASSERT_TRUE(actual.has_value());

    cudf::test::fixed_width_column_wrapper<int64_t> expected_data({3, 4, 5});
    cudf::test::fixed_width_column_wrapper<int64_t> expected_keys({9, 9, 9});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(cudf::table_view({expected_data, expected_keys}), *actual);
    source->finalize(std::move(iterator), ctx_ref());
  }
}
TEST_F(ParquetReferenceReadTest, RejectsInvalidConstructorArguments)
{
  EXPECT_THROW(gqe::storage::parquet_reference_iterate_task(
                 {}, {"col_0"}, int64_types(1), std::nullopt, nullptr),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::parquet_reference_iterate_task(
                 {}, {"col_0"}, int64_types(2), std::nullopt, _footer_cache),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::parquet_resolve_task(nullptr, {"col_0"}, int64_types(1)),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::storage::parquet_footer_cache{}.get_or_load(""),
               std::invalid_argument);
}

TEST_F(ParquetReferenceReadTest, FetchRejectsOutOfRangeReferences)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2});
  auto table = make_table_from_columns(col.release());

  auto const path = temp_env->get_temp_filepath("ref_read_bounds.parquet");
  write_table_to_file(table->view(), {"col_0"}, path);

  gqe::storage::parquet_resolve_task resolve{_footer_cache, {"col_0"}, int64_types(1)};
  auto out_of_range =
    gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::reference{gqe::storage::parquet_reference::make(
        path, /*row_group_begin=*/5, /*row_group_count=*/1, /*num_rows=*/2)})
      .build();
  EXPECT_THROW(std::ignore = resolve.next(out_of_range, ctx_ref()), std::invalid_argument);
}

TEST_F(ParquetReferenceReadTest, SourceSplitsCoverFilesAndShareFooterCache)
{
  std::vector<std::string> paths;
  for (int64_t value = 0; value < 3; ++value) {
    cudf::test::fixed_width_column_wrapper<int64_t> column({value});
    auto table = make_table_from_columns(column.release());
    auto path  = temp_env->get_temp_filepath("partition_ref_" + std::to_string(value) + ".parquet");
    write_table_to_file(table->view(), {"col_0"}, path);
    paths.push_back(std::move(path));
  }
  gqe::storage::parquet_reference_iterate_task task{
    paths, {"col_0"}, int64_types(1), std::nullopt, _footer_cache};
  EXPECT_EQ(task.work_unit_count(), 3);
  gqe::storage::parquet_resolve_task resolve{_footer_cache, {"col_0"}, int64_types(1)};
  for (auto const range : std::vector<gqe::qep::source_work_range>{{0, 1}, {1, 3}}) {
    auto split_reader = task.split(range);
    auto iterator     = split_reader->initialize({}, ctx_ref());
    EXPECT_EQ(_footer_cache->size(), range.end) << "Partitions must populate the shared cache";
    for (auto index = range.begin; index < range.end; ++index) {
      auto chunk = split_reader->next(iterator, ctx_ref());
      ASSERT_TRUE(chunk);
      auto const* ref = as_parquet_reference(*chunk);
      ASSERT_NE(ref, nullptr);
      EXPECT_EQ(ref->file_path(), paths[index]);
      auto deferred = resolve.next(*chunk, ctx_ref());
      ASSERT_TRUE(deferred);
      auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
      auto actual       = gqe::qep::to_table_view(materialized);
      ASSERT_TRUE(actual);
      cudf::test::fixed_width_column_wrapper<int64_t> expected({static_cast<int64_t>(index)});
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT((cudf::table_view{{expected}}), *actual);
    }
    EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
    split_reader->finalize(std::move(iterator), ctx_ref());
  }
  EXPECT_EQ(_footer_cache->size(), paths.size());
  gqe::storage::parquet_reference_iterate_task count{paths, {}, {}, std::nullopt, _footer_cache};
  auto split_reader = count.split({1, 3});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  auto chunk        = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 2);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  auto empty_split    = count.split({3, 3});
  auto empty_iterator = empty_split->initialize({}, ctx_ref());
  auto empty_chunk    = empty_split->next(empty_iterator, ctx_ref());
  ASSERT_TRUE(empty_chunk);
  EXPECT_EQ(gqe::qep::get_row_count(*empty_chunk), 0);
  EXPECT_FALSE(empty_split->next(empty_iterator, ctx_ref()));
  empty_split->finalize(std::move(empty_iterator), ctx_ref());
  EXPECT_THROW(std::ignore = task.split({2, 1}), std::invalid_argument);
  EXPECT_THROW(std::ignore = task.split({0, 4}), std::invalid_argument);
  gqe::storage::parquet_reference_iterate_task empty{{}, {}, {}, std::nullopt, _footer_cache};
  EXPECT_EQ(empty.work_unit_count(), 0);
  EXPECT_THROW(std::ignore = empty.split({0, 1}), std::invalid_argument);
}

TEST_F(ParquetReferenceReadTest, EmptySourceSplitPreservesOutputShape)
{
  auto const type = cudf::data_type{cudf::type_id::INT32};
  gqe::storage::parquet_reference_iterate_task source{
    {}, {"value"}, {type}, std::nullopt, _footer_cache};
  EXPECT_EQ(source.work_unit_count(), 0);
  auto split_reader = source.split({0, 0});
  ASSERT_NE(dynamic_cast<gqe::storage::parquet_reference_iterate_task*>(split_reader.get()),
            nullptr);
  auto iterator = split_reader->initialize({}, ctx_ref());
  auto chunk    = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto const* slot = std::get_if<gqe::qep::state_kind::reference>(chunk->front().get());
  ASSERT_NE(slot, nullptr);
  auto const* reference = dynamic_cast<gqe::storage::parquet_reference const*>(slot->state().get());
  ASSERT_NE(reference, nullptr);
  EXPECT_TRUE(reference->is_empty());
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  EXPECT_THROW(std::ignore = source.split({1, 1}), std::invalid_argument);
}
