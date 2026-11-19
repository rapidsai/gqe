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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/shapes/deferred_column.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/in_memory_reference_iterate.hpp>
#include <gqe/storage/in_memory_resolve.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/source_references.hpp>
#include <gqe/storage/table_provider.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

/**
 * @brief Minimal in-memory table over externally owned row groups, for provider resolution.
 */
class fake_in_memory_table final : public gqe::storage::in_memory_table_interface {
 public:
  explicit fake_in_memory_table(std::deque<gqe::storage::row_group> const* row_groups)
    : _row_groups{row_groups}
  {
  }

  [[nodiscard]] bool is_readable() const override { return true; }
  [[nodiscard]] bool is_writeable() const override { return false; }
  [[nodiscard]] int32_t max_concurrent_readers() const override { return 1; }
  [[nodiscard]] int32_t max_concurrent_writers() const override { return 0; }
  std::unique_ptr<gqe::storage::readable_view> readable_view() override
  {
    throw std::logic_error{"fake_in_memory_table: readable_view is unused"};
  }
  std::unique_ptr<gqe::storage::writeable_view> writeable_view() override
  {
    throw std::logic_error{"fake_in_memory_table: writeable_view is unused"};
  }

  [[nodiscard]] cudf::size_type get_column_index(std::string const&) const override
  {
    throw std::logic_error{"fake_in_memory_table: get_column_index is unused"};
  }
  [[nodiscard]] gqe::memory_kind::type get_memory_kind() const override
  {
    return gqe::memory_kind::system{};
  }
  [[nodiscard]] std::size_t num_row_groups() const override { return _row_groups->size(); }
  [[nodiscard]] gqe::storage::row_group_interface const* row_group_at(
    std::size_t index) const override
  {
    return &_row_groups->at(index);
  }
  [[nodiscard]] std::shared_mutex& row_group_latch() const override { return _latch; }
  void serialize_table_to_disk(std::string const&,
                               std::string const&,
                               gqe::optimization_parameters const&,
                               cuda::stream_ref) override
  {
    throw std::logic_error{"fake_in_memory_table: serialize_table_to_disk is unused"};
  }
  void deserialize_table_from_disk(std::string const&,
                                   std::string const&,
                                   gqe::optimization_parameters const&,
                                   rmm::device_async_resource_ref,
                                   cuda::stream_ref,
                                   gqe::table_statistics_manager*) override
  {
    throw std::logic_error{"fake_in_memory_table: deserialize_table_from_disk is unused"};
  }
  void deallocate(cuda::stream_ref, rmm::device_async_resource_ref) override {}

 private:
  std::deque<gqe::storage::row_group> const* _row_groups;
  mutable std::shared_mutex _latch;
};

/**
 * @brief Name-keyed table provider over test-owned tables.
 */
class fake_table_provider final : public gqe::storage::table_provider {
 public:
  void add_table(std::string name, std::shared_ptr<gqe::storage::table> table)
  {
    _tables.emplace(std::move(name), std::move(table));
  }

  [[nodiscard]] std::shared_ptr<gqe::storage::table> get_table(
    std::string_view table_name) const override
  {
    auto const entry = _tables.find(std::string{table_name});
    if (entry == _tables.end()) { throw std::logic_error{"fake_table_provider: table not found"}; }
    return entry->second;
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<gqe::storage::table>> _tables;
};

}  // namespace

class InMemoryReferenceReadTest : public gqe::test::BaseFixture {
 protected:
  using row_group_storage = std::deque<gqe::storage::row_group>;

  InMemoryReferenceReadTest()
  {
    // Deferred sources reject use_overlap_mtx (see in_memory_deferred_source), so pin it off
    // instead of inheriting the fixture's default parameters.
    auto parameters            = gqe::make_optimization_parameters(/*only_defaults=*/true);
    parameters.use_overlap_mtx = false;
    initialize_contexts(parameters);
  }

  gqe::context_reference ctx_ref() { return {get_task_manager_ctx(), get_query_ctx()}; }

  static std::vector<gqe::storage::row_group_interface const*> row_group_refs(
    row_group_storage const& row_groups)
  {
    std::vector<gqe::storage::row_group_interface const*> refs;
    refs.reserve(row_groups.size());
    for (auto const& row_group : row_groups) {
      refs.push_back(&row_group);
    }
    return refs;
  }

  static std::vector<std::size_t> iota_indexes(std::size_t count)
  {
    std::vector<std::size_t> indexes(count);
    std::iota(indexes.begin(), indexes.end(), std::size_t{0});
    return indexes;
  }

  static void append_row_group(
    row_group_storage& row_groups,
    std::vector<std::vector<int32_t>> const& values,
    std::optional<cudf::size_type> zone_map_partition_size = std::nullopt)
  {
    if (values.empty()) { throw std::invalid_argument("row group must have at least one column"); }

    std::vector<std::unique_ptr<cudf::column>> cudf_columns;
    cudf_columns.reserve(values.size());
    for (auto const& column : values) {
      cudf::test::fixed_width_column_wrapper<int32_t> wrapper(column.begin(), column.end());
      cudf_columns.push_back(wrapper.release());
    }

    std::unique_ptr<gqe::zone_map> zone_map;
    if (zone_map_partition_size) {
      std::vector<cudf::column_view> views;
      for (auto const& column : cudf_columns) {
        views.push_back(column->view());
      }
      zone_map =
        std::make_unique<gqe::local_zone_map>(cudf::table_view{views}, *zone_map_partition_size);
    }

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.reserve(cudf_columns.size());
    for (auto& column : cudf_columns) {
      columns.push_back(std::make_unique<gqe::storage::contiguous_column>(std::move(*column)));
    }
    row_groups.emplace_back(std::move(columns), std::move(zone_map));
  }

  static std::vector<cudf::data_type> int32_types(std::size_t count)
  {
    return std::vector<cudf::data_type>(count, cudf::data_type{cudf::type_id::INT32});
  }

  static gqe::qep::source_morsel_metadata host_metadata()
  {
    return {.residency = gqe::qep::data_residency::host};
  }
};

TEST_F(InMemoryReferenceReadTest, EmitsOneReferencePerRowGroup)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}, {10, 20, 30}});
  append_row_group(row_groups, {{4, 5}, {40, 50}});

  gqe::storage::in_memory_reference_iterate_task task{
    "orders", row_group_refs(row_groups), iota_indexes(2), {0, 1}, int32_types(2), host_metadata()};

  auto iterator = task.initialize({}, ctx_ref());

  auto first = task.next(iterator, ctx_ref());
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(gqe::qep::get_row_count(*first), 3);
  auto const* first_slot = std::get_if<gqe::qep::state_kind::reference>(first->front().get());
  ASSERT_NE(first_slot, nullptr);
  auto const* first_ref =
    dynamic_cast<gqe::storage::in_memory_reference const*>(first_slot->state().get());
  ASSERT_NE(first_ref, nullptr);
  EXPECT_EQ(first_ref->table_name(), "orders");
  EXPECT_EQ(first_ref->row_group_index(), 0);
  EXPECT_FALSE(first_ref->is_empty());

  auto second = task.next(iterator, ctx_ref());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(gqe::qep::get_row_count(*second), 2);

  EXPECT_FALSE(task.next(iterator, ctx_ref()).has_value());
  EXPECT_FALSE(task.next(iterator, ctx_ref()).has_value());
}

TEST_F(InMemoryReferenceReadTest, ResolveAndMaterializeReferencedRowGroups)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}, {10, 20, 30}});
  append_row_group(row_groups, {{4, 5}, {40, 50}});

  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));

  gqe::storage::in_memory_reference_iterate_task iterate{
    "orders", row_group_refs(row_groups), iota_indexes(2), {0, 1}, int32_types(2), host_metadata()};
  gqe::storage::in_memory_resolve_task resolve{&provider, {0, 1}, int32_types(2)};

  auto iterator = iterate.initialize({}, ctx_ref());

  std::vector<std::vector<std::vector<int32_t>>> const expected = {{{1, 2, 3}, {10, 20, 30}},
                                                                   {{4, 5}, {40, 50}}};
  for (auto const& expected_chunk : expected) {
    auto reference = iterate.next(iterator, ctx_ref());
    ASSERT_TRUE(reference.has_value());

    auto deferred = resolve.next(*reference, ctx_ref());
    ASSERT_TRUE(deferred.has_value());
    EXPECT_EQ(gqe::qep::get_row_count(*deferred),
              static_cast<cudf::size_type>(expected_chunk[0].size()));

    auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
    auto actual       = gqe::qep::to_table_view(materialized);
    ASSERT_TRUE(actual.has_value());

    cudf::test::fixed_width_column_wrapper<int32_t> column_a(expected_chunk[0].begin(),
                                                             expected_chunk[0].end());
    cudf::test::fixed_width_column_wrapper<int32_t> column_b(expected_chunk[1].begin(),
                                                             expected_chunk[1].end());
    cudf::table_view const expected_view{{column_a, column_b}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_view, *actual);
  }
  EXPECT_FALSE(iterate.next(iterator, ctx_ref()).has_value());
}

TEST_F(InMemoryReferenceReadTest, MaterializeTwiceYieldsIndependentColumns)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}});

  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));

  gqe::storage::in_memory_reference_iterate_task iterate{
    "orders", row_group_refs(row_groups), iota_indexes(1), {0}, int32_types(1), host_metadata()};
  gqe::storage::in_memory_resolve_task resolve{&provider, {0}, int32_types(1)};

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

TEST_F(InMemoryReferenceReadTest, CountOnlyEmitsAggregateRowCountWithoutReferences)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}});
  append_row_group(row_groups, {{4, 5}});

  gqe::storage::in_memory_reference_iterate_task task{
    "orders", row_group_refs(row_groups), iota_indexes(2), {}, {}, host_metadata()};

  auto iterator = task.initialize({}, ctx_ref());
  auto chunk    = task.next(iterator, ctx_ref());
  ASSERT_TRUE(chunk.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 5);
  EXPECT_FALSE(task.next(iterator, ctx_ref()).has_value());
}

TEST_F(InMemoryReferenceReadTest, CountOnlyChunksPassThroughTheResolve)
{
  fake_table_provider provider;
  // A count-only plan constructs the resolve with the same empty projection as its iterate.
  gqe::storage::in_memory_resolve_task resolve{&provider, {}, {}};

  auto count_only = gqe::qep::make_row_count_container(5);
  auto passed     = resolve.next(count_only, ctx_ref());
  ASSERT_TRUE(passed.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*passed), 5);

  gqe::storage::pruning_result_t const pruning{
    std::vector<gqe::zone_map::partition>{{false, 0, 3, {0}}}, 3};
  auto reference = gqe::qep::state_container_builder()
                     .add_state(gqe::qep::state_kind::reference{
                       gqe::storage::in_memory_reference::make("orders", 0, pruning)})
                     .build();
  EXPECT_THROW(std::ignore = resolve.next(reference, ctx_ref()), std::invalid_argument);
}

TEST_F(InMemoryReferenceReadTest, EmptyInputEmitsOneEmptyReferenceAndEmptyTable)
{
  fake_table_provider provider;
  row_group_storage row_groups;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));

  gqe::storage::in_memory_reference_iterate_task iterate{
    "orders", {}, {}, {0, 1}, int32_types(2), host_metadata()};
  gqe::storage::in_memory_resolve_task resolve{&provider, {0, 1}, int32_types(2)};

  auto iterator  = iterate.initialize({}, ctx_ref());
  auto reference = iterate.next(iterator, ctx_ref());
  ASSERT_TRUE(reference.has_value());
  EXPECT_EQ(gqe::qep::get_row_count(*reference), 0);
  auto const* slot = std::get_if<gqe::qep::state_kind::reference>(reference->front().get());
  ASSERT_NE(slot, nullptr);
  auto const* ref = dynamic_cast<gqe::storage::in_memory_reference const*>(slot->state().get());
  ASSERT_NE(ref, nullptr);
  EXPECT_TRUE(ref->is_empty());
  EXPECT_FALSE(iterate.next(iterator, ctx_ref()).has_value());

  auto empty_table = resolve.next(*reference, ctx_ref());
  ASSERT_TRUE(empty_table.has_value());
  auto view = gqe::qep::to_table_view(*empty_table);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->num_columns(), 2);
  EXPECT_EQ(view->num_rows(), 0);
}

TEST_F(InMemoryReferenceReadTest, ResolveRejectsUnknownAndMismatchedReferences)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}});

  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));

  auto make_reference_container = [](std::string name, std::size_t row_group_index) {
    // A select-all pruning result over the three-row, one-column row group above.
    gqe::storage::pruning_result_t const pruning{
      std::vector<gqe::zone_map::partition>{{false, 0, 3, {0}}}, 3};
    auto ref = gqe::storage::in_memory_reference::make(std::move(name), row_group_index, pruning);
    return gqe::qep::state_container_builder()
      .add_state(gqe::qep::state_kind::reference{std::move(ref)})
      .build();
  };

  gqe::storage::in_memory_resolve_task resolve{&provider, {0}, int32_types(1)};

  auto unknown = make_reference_container("missing", 0);
  EXPECT_THROW(std::ignore = resolve.next(unknown, ctx_ref()), std::logic_error);

  auto out_of_range = make_reference_container("orders", 7);
  EXPECT_THROW(std::ignore = resolve.next(out_of_range, ctx_ref()), std::out_of_range);

  auto reference = make_reference_container("orders", 0);
  gqe::storage::in_memory_resolve_task mistyped_resolve{
    &provider, {0}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_THROW(std::ignore = mistyped_resolve.next(reference, ctx_ref()), std::invalid_argument);

  gqe::storage::in_memory_resolve_task out_of_schema_resolve{&provider, {0, 1}, int32_types(2)};
  EXPECT_THROW(std::ignore = out_of_schema_resolve.next(reference, ctx_ref()), std::out_of_range);

  // The reference claims a three-row pruning extent; a shorter resolved row group must fail
  // loudly instead of slicing past its end.
  row_group_storage short_row_groups;
  append_row_group(short_row_groups, {{1, 2}});
  fake_table_provider short_provider;
  short_provider.add_table("orders", std::make_shared<fake_in_memory_table>(&short_row_groups));
  gqe::storage::in_memory_resolve_task short_resolve{&short_provider, {0}, int32_types(1)};
  auto extent_mismatch = make_reference_container("orders", 0);
  EXPECT_THROW(std::ignore = short_resolve.next(extent_mismatch, ctx_ref()), std::logic_error);
}

TEST_F(InMemoryReferenceReadTest, ResolveRejectsNullableColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> wrapper({1, 2, 3}, {true, false, true});
  std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
  columns.push_back(
    std::make_unique<gqe::storage::contiguous_column>(std::move(*wrapper.release())));
  row_group_storage row_groups;
  row_groups.emplace_back(std::move(columns), nullptr);

  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));

  gqe::storage::pruning_result_t const pruning{
    std::vector<gqe::zone_map::partition>{{false, 0, 3, {1}}}, 3};
  auto reference = gqe::qep::state_container_builder()
                     .add_state(gqe::qep::state_kind::reference{
                       gqe::storage::in_memory_reference::make("orders", 0, pruning)})
                     .build();

  gqe::storage::in_memory_resolve_task resolve{&provider, {0}, int32_types(1)};
  EXPECT_THROW(std::ignore = resolve.next(reference, ctx_ref()), std::invalid_argument);
}

TEST_F(InMemoryReferenceReadTest, RejectsInvalidConstructorArguments)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1, 2, 3}});

  EXPECT_THROW(
    gqe::storage::in_memory_reference_iterate_task(
      "", row_group_refs(row_groups), iota_indexes(1), {0}, int32_types(1), host_metadata()),
    std::invalid_argument);
  EXPECT_THROW(gqe::storage::in_memory_reference_iterate_task(
                 "orders", row_group_refs(row_groups), {}, {0}, int32_types(1), host_metadata()),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::in_memory_reference_iterate_task("orders",
                                                              row_group_refs(row_groups),
                                                              iota_indexes(1),
                                                              {0, 1},
                                                              int32_types(1),
                                                              host_metadata()),
               std::invalid_argument);

  fake_table_provider provider;
  EXPECT_THROW(gqe::storage::in_memory_resolve_task(nullptr, {0}, int32_types(1)),
               std::invalid_argument);
  EXPECT_THROW(gqe::storage::in_memory_resolve_task(&provider, {0}, int32_types(2)),
               std::invalid_argument);
}

TEST_F(InMemoryReferenceReadTest, InitializeRejectsNullableColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> wrapper({1, 2, 3}, {true, false, true});
  std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
  columns.push_back(
    std::make_unique<gqe::storage::contiguous_column>(std::move(*wrapper.release())));
  row_group_storage row_groups;
  row_groups.emplace_back(std::move(columns), nullptr);

  gqe::storage::in_memory_reference_iterate_task task{
    "orders", row_group_refs(row_groups), iota_indexes(1), {0}, int32_types(1), host_metadata()};
  EXPECT_THROW(std::ignore = task.initialize({}, ctx_ref()), std::invalid_argument);
}

TEST_F(InMemoryReferenceReadTest, SourceSplitsPreserveOriginalIndexesAndProjection)
{
  row_group_storage row_groups;
  for (int32_t value = 0; value < 6; ++value) {
    append_row_group(row_groups, {{value}, {value * 10}});
  }
  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));
  gqe::storage::in_memory_resolve_task resolve{&provider, {1, 0}, int32_types(2)};
  auto all_groups = row_group_refs(row_groups);
  std::vector<std::size_t> const indexes{1, 3, 5};
  gqe::storage::in_memory_reference_iterate_task task{"orders",
                                                      {all_groups[1], all_groups[3], all_groups[5]},
                                                      indexes,
                                                      {1, 0},
                                                      int32_types(2),
                                                      host_metadata()};
  EXPECT_EQ(task.work_unit_count(), 3);
  for (auto const range : std::vector<gqe::qep::source_work_range>{{0, 1}, {1, 3}}) {
    auto split_reader = task.split(range);
    auto const* residency =
      dynamic_cast<gqe::qep::input_residency_trait const*>(split_reader.get());
    ASSERT_NE(residency, nullptr);
    EXPECT_EQ(residency->input_residency().residency, gqe::qep::data_residency::host);
    auto iterator = split_reader->initialize({}, ctx_ref());
    for (auto index = range.begin; index < range.end; ++index) {
      auto chunk = split_reader->next(iterator, ctx_ref());
      ASSERT_TRUE(chunk);
      auto const* slot = std::get_if<gqe::qep::state_kind::reference>(chunk->front().get());
      ASSERT_NE(slot, nullptr);
      auto const* reference =
        dynamic_cast<gqe::storage::in_memory_reference const*>(slot->state().get());
      ASSERT_NE(reference, nullptr);
      EXPECT_EQ(reference->table_name(), "orders");
      EXPECT_EQ(reference->row_group_index(), indexes[index]);
      auto deferred = resolve.next(*chunk, ctx_ref());
      ASSERT_TRUE(deferred);
      auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
      auto actual       = gqe::qep::to_table_view(materialized);
      ASSERT_TRUE(actual);
      auto const value = static_cast<int32_t>(indexes[index]);
      cudf::test::fixed_width_column_wrapper<int32_t> first({value * 10});
      cudf::test::fixed_width_column_wrapper<int32_t> second({value});
      CUDF_TEST_EXPECT_TABLES_EQUAL((cudf::table_view{{first, second}}), *actual);
    }
    EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
    split_reader->finalize(std::move(iterator), ctx_ref());
  }
}

TEST_F(InMemoryReferenceReadTest, SourceSplitCountOnlyAndInvalidRanges)
{
  row_group_storage row_groups;
  append_row_group(row_groups, {{1}});
  append_row_group(row_groups, {{2, 3}});
  gqe::storage::in_memory_reference_iterate_task task{
    "orders", row_group_refs(row_groups), {4, 7}, {}, {}, host_metadata()};
  auto split_reader = task.split({1, 2});
  auto iterator     = split_reader->initialize({}, ctx_ref());
  auto chunk        = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  EXPECT_EQ(gqe::qep::try_row_count(*chunk), 2);
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  auto empty_split    = task.split({2, 2});
  auto empty_iterator = empty_split->initialize({}, ctx_ref());
  auto empty_chunk    = empty_split->next(empty_iterator, ctx_ref());
  ASSERT_TRUE(empty_chunk);
  EXPECT_EQ(gqe::qep::get_row_count(*empty_chunk), 0);
  EXPECT_FALSE(empty_split->next(empty_iterator, ctx_ref()));
  empty_split->finalize(std::move(empty_iterator), ctx_ref());
  EXPECT_THROW(std::ignore = task.split({2, 1}), std::invalid_argument);
  EXPECT_THROW(std::ignore = task.split({0, 3}), std::invalid_argument);
  gqe::storage::in_memory_reference_iterate_task empty{"orders", {}, {}, {}, {}, host_metadata()};
  EXPECT_EQ(empty.work_unit_count(), 0);
  EXPECT_THROW(std::ignore = empty.split({0, 1}), std::invalid_argument);
}

TEST_F(InMemoryReferenceReadTest, SourceSplitPreservesPruningAndOriginalIndexAfterClone)
{
  get_query_ctx()->parameters.use_partition_pruning   = true;
  get_query_ctx()->parameters.zone_map_partition_size = 2;
  row_group_storage row_groups;
  append_row_group(row_groups, {{100, 101}}, 2);
  append_row_group(row_groups, {{0, 1}}, 2);
  append_row_group(row_groups, {{12, 13, 20, 21}}, 2);
  fake_table_provider provider;
  provider.add_table("orders", std::make_shared<fake_in_memory_table>(&row_groups));
  gqe::storage::in_memory_resolve_task resolve{&provider, {0}, int32_types(1)};
  auto split_reader = [&] {
    auto filter = std::make_unique<gqe::greater_expression>(
      std::make_unique<gqe::column_reference_expression>(0),
      std::make_unique<gqe::literal_expression<int32_t>>(15));
    gqe::storage::in_memory_reference_iterate_task task{"orders",
                                                        row_group_refs(row_groups),
                                                        iota_indexes(3),
                                                        {0},
                                                        int32_types(1),
                                                        host_metadata(),
                                                        std::move(filter)};
    return task.split({1, 3});
  }();
  auto clone    = gqe::qep::clone_qep_task_as<gqe::qep::iterate_task>(*split_reader);
  auto iterator = clone->initialize({}, ctx_ref());
  auto chunk    = clone->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto const* slot = std::get_if<gqe::qep::state_kind::reference>(chunk->front().get());
  ASSERT_NE(slot, nullptr);
  auto const* reference =
    dynamic_cast<gqe::storage::in_memory_reference const*>(slot->state().get());
  ASSERT_NE(reference, nullptr);
  EXPECT_EQ(reference->row_group_index(), 2);
  auto deferred = resolve.next(*chunk, ctx_ref());
  ASSERT_TRUE(deferred);
  auto materialized = gqe::qep::materialize_deferred_columns(*deferred);
  auto actual       = gqe::qep::to_table_view(materialized);
  ASSERT_TRUE(actual);
  cudf::test::fixed_width_column_wrapper<int32_t> expected({20, 21});
  CUDF_TEST_EXPECT_TABLES_EQUAL((cudf::table_view{{expected}}), *actual);
  EXPECT_FALSE(clone->next(iterator, ctx_ref()));
  clone->finalize(std::move(iterator), ctx_ref());
}

TEST_F(InMemoryReferenceReadTest, EmptySourceSplitPreservesOutputShape)
{
  auto const type = cudf::data_type{cudf::type_id::INT32};
  gqe::storage::in_memory_reference_iterate_task source{
    "orders", {}, {}, {0}, {type}, host_metadata()};
  EXPECT_EQ(source.work_unit_count(), 0);
  auto split_reader = source.split({0, 0});
  ASSERT_NE(dynamic_cast<gqe::storage::in_memory_reference_iterate_task*>(split_reader.get()),
            nullptr);
  auto iterator = split_reader->initialize({}, ctx_ref());
  auto chunk    = split_reader->next(iterator, ctx_ref());
  ASSERT_TRUE(chunk);
  auto const* slot = std::get_if<gqe::qep::state_kind::reference>(chunk->front().get());
  ASSERT_NE(slot, nullptr);
  auto const* reference =
    dynamic_cast<gqe::storage::in_memory_reference const*>(slot->state().get());
  ASSERT_NE(reference, nullptr);
  EXPECT_TRUE(reference->is_empty());
  EXPECT_FALSE(split_reader->next(iterator, ctx_ref()));
  split_reader->finalize(std::move(iterator), ctx_ref());
  EXPECT_THROW(std::ignore = source.split({1, 1}), std::invalid_argument);
}
