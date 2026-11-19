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

#include <gqe/qep/column_materialization.hpp>

#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

/** @brief Build an owned QEP column state from one owning cuDF column. */
gqe::qep::shared_state owned_state(std::unique_ptr<cudf::column> column)
{
  return gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column{std::move(column)},
                                     gqe::query_stream::default_stream());
}

/** @brief Build an owned INT64 QEP column state. */
gqe::qep::shared_state owned_int64_materialization(std::initializer_list<int64_t> values)
{
  return owned_state(int64_column_wrapper(values).release());
}

/** @brief Build a typed immutable position-list handle. */
gqe::qep::position_list_handle materialization_positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

/** @brief Build a gather source over supplied direct column states. */
gqe::qep::indirect_gather_source materialization_source(
  gqe::qep::state_container source_columns,
  gqe::qep::position_list_handle positions,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(source_columns));
  return gqe::qep::indirect_gather_source_state::make(
    std::move(relation), std::move(positions), policy);
}

/** @brief Build one indirect logical-column state from a gather source and source index. */
gqe::qep::shared_state materialization_indirect(gqe::qep::indirect_gather_source source,
                                                cudf::size_type source_column_index)
{
  return gqe::qep::indirect_column_state::make(std::move(source), source_column_index);
}

/** @brief Validate and return the logical facade for a test-owned container. */
gqe::qep::logical_table_view logical_view(gqe::qep::state_container const& input)
{
  auto result = gqe::qep::logical_table_view::try_from(input);
  EXPECT_TRUE(result.has_value());
  return *result;
}

TEST(QepColumnMaterialization, SelectiveMaterializationPreservesRequestOrderAndDuplicates)
{
  auto direct0 = owned_int64_materialization({1, 2});
  string_column_wrapper borrowed_owner{"zéro", "二"};
  cudf::column_view const borrowed_view = borrowed_owner;
  auto direct1 = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{borrowed_view});
  auto shared_source =
    materialization_source({owned_int64_materialization({10, 20, 30}),
                            owned_state(int32_column_wrapper{100, 200, 300}.release())},
                           materialization_positions({2, 0}));
  auto indirect0 = materialization_indirect(shared_source, 0);
  auto indirect1 = materialization_indirect(shared_source, 1);
  gqe::qep::state_container input{direct0, indirect0, direct1, indirect1};
  std::vector<cudf::size_type> request{3, 0, 1, 3, 2, 1};

  gqe::qep::column_materialization result =
    gqe::qep::materialize_columns(logical_view(input), request);

  EXPECT_EQ(result.row_count(), 2);
  EXPECT_EQ(
    std::vector(result.logical_column_indices().begin(), result.logical_column_indices().end()),
    request);
  ASSERT_EQ(result.states().size(), request.size());
  EXPECT_EQ(result.states()[1], direct0);
  EXPECT_EQ(result.states()[4], direct1);

  int32_column_wrapper expected_indirect1{300, 100};
  int64_column_wrapper expected_direct0{1, 2};
  int64_column_wrapper expected_indirect0{30, 10};
  string_column_wrapper expected_direct1{"zéro", "二"};
  auto table = result.table_view();
  ASSERT_EQ(table.num_columns(), 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_indirect1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_direct0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), expected_indirect0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(3), expected_indirect1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(4), expected_direct1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(5), expected_indirect0);
  EXPECT_EQ(table.column(1).head(), gqe::qep::to_column_view(direct0)->head());
  EXPECT_EQ(table.column(4).head(), borrowed_view.head());
}

TEST(QepColumnMaterialization, FullPreservingLeavesMaskUnappliedAndReportsInputGeometry)
{
  auto mask = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{
      bool_column_wrapper{{true, false, true}, {true, false, true}}.release()},
    gqe::query_stream::default_stream());
  auto direct   = owned_state(int64_column_wrapper{{1, 2, 3}, {true, false, true}}.release());
  auto source   = materialization_source({owned_int64_materialization({10, 20, 30})},
                                       materialization_positions({2, 1, 0}));
  auto indirect = materialization_indirect(source, 0);
  gqe::qep::state_container input{mask, direct, indirect};
  auto logical = logical_view(input);

  gqe::qep::column_materialization result = gqe::qep::materialize_all_columns(logical);

  EXPECT_EQ(result.row_count(), 3);
  EXPECT_EQ(result.states().size(), 2);
  EXPECT_EQ(result.states()[0], direct);
  EXPECT_EQ(result.logical_column_indices()[0], 0);
  EXPECT_EQ(result.logical_column_indices()[1], 1);
  ASSERT_TRUE(logical.valid_mask().has_value());
  int64_column_wrapper expected_direct{{1, 2, 3}, {true, false, true}};
  int64_column_wrapper expected_indirect{30, 20, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(0), expected_direct);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(1), expected_indirect);
}

TEST(QepColumnMaterialization, EmptyRequestsPreserveGeometryAndPerformNoIndirectGather)
{
  auto unsafe_source =
    materialization_source({owned_int64_materialization({1})}, materialization_positions({100}));
  gqe::qep::state_container input{materialization_indirect(unsafe_source, 0)};

  gqe::qep::column_materialization result = gqe::qep::materialize_columns(logical_view(input), {});
  EXPECT_EQ(result.row_count(), 1);
  EXPECT_TRUE(result.logical_column_indices().empty());
  EXPECT_EQ(result.table_view().num_columns(), 0);
  EXPECT_THROW(std::ignore = std::move(result).release_column_states(), std::logic_error);

  gqe::qep::state_container zero_input{materialization_indirect(
    materialization_source({owned_int64_materialization({1})}, materialization_positions({})), 0)};
  gqe::qep::column_materialization zero =
    gqe::qep::materialize_columns(logical_view(zero_input), {});
  EXPECT_EQ(zero.row_count(), 0);
}

TEST(QepColumnMaterialization, CompactionAppliesValidAndTrueMaskBeforeIndirectGather)
{
  auto mask = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{
      bool_column_wrapper{{true, true, true}, {false, true, true}}.release()},
    gqe::query_stream::default_stream());

  int64_column_wrapper direct_owner{0, 10, 20, 30, 40};
  auto direct_slice = cudf::slice(direct_owner, {1, 4}).front();
  auto direct = gqe::qep::make_shared_state(gqe::qep::state_kind::cudf_column_view{direct_slice});

  auto positions = materialization_positions({100, 0, 1});
  auto source    = materialization_source({owned_int64_materialization({200, 300})}, positions);
  auto indirect  = materialization_indirect(source, 0);
  gqe::qep::state_container input{mask, direct, indirect};

  gqe::qep::column_materialization result =
    gqe::qep::materialize_compacted_columns(logical_view(input),
                                            std::vector<cudf::size_type>{0, 1},
                                            true,
                                            gqe::query_stream::default_stream(),
                                            cudf::get_current_device_resource_ref());

  EXPECT_EQ(result.row_count(), 2);
  ASSERT_NE(result.row_positions(), nullptr);
  int32_column_wrapper expected_rows{1, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.row_positions()->column().view(), expected_rows);
  int64_column_wrapper expected_direct{20, 30};
  int64_column_wrapper expected_indirect{200, 300};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(0), expected_direct);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(1), expected_indirect);
}

TEST(QepColumnMaterialization, CompactionHandlesDistinctSourcesWithSharedInputPositions)
{
  auto mask = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
    gqe::query_stream::default_stream());
  auto positions = materialization_positions({2, 0, 1});
  auto source0   = materialization_source({owned_int64_materialization({10, 20, 30})}, positions);
  auto source1   = materialization_source({owned_int64_materialization({40, 50, 60})}, positions);
  gqe::qep::state_container input{
    mask, materialization_indirect(source0, 0), materialization_indirect(source1, 0)};

  gqe::qep::column_materialization result =
    gqe::qep::materialize_all_compacted_columns(logical_view(input));

  int64_column_wrapper expected0{30, 20};
  int64_column_wrapper expected1{60, 50};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(0), expected0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.table_view().column(1), expected1);
}

TEST(QepColumnMaterialization, EmptyCompactedRequestWithNoSurvivingRows)
{
  std::vector<bool_column_wrapper> masks;
  masks.emplace_back(std::initializer_list<bool>{});
  masks.push_back(bool_column_wrapper{{true, true}, {false, false}});
  masks.emplace_back(std::initializer_list<bool>{false, false});
  for (auto& mask : masks) {
    std::vector<int64_t> values(static_cast<cudf::column_view>(mask).size(), 1);
    auto data = owned_state(int64_column_wrapper(values.begin(), values.end()).release());
    gqe::qep::state_container input{
      gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask{mask.release()},
                                  gqe::query_stream::default_stream()),
      data};
    auto result = gqe::qep::materialize_compacted_columns(logical_view(input), {});
    EXPECT_EQ(result.row_count(), 0);
    EXPECT_TRUE(result.states().empty());
  }
}

TEST(QepColumnMaterialization, EmptyCompactedRequestCountsSurvivingRowsWithoutDataGather)
{
  auto mask = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{
      bool_column_wrapper{{true, true, false, true}, {true, false, true, true}}.release()},
    gqe::query_stream::default_stream());
  auto source = materialization_source({owned_int64_materialization({1})},
                                       materialization_positions({100, 100, 100, 100}));
  gqe::qep::state_container input{mask, materialization_indirect(source, 0)};

  gqe::qep::column_materialization result =
    gqe::qep::materialize_compacted_columns(logical_view(input), {});
  EXPECT_EQ(result.row_count(), 2);
  EXPECT_TRUE(result.states().empty());

  gqe::qep::state_container unmasked{owned_int64_materialization({1, 2, 3})};
  gqe::qep::column_materialization unmasked_result =
    gqe::qep::materialize_compacted_columns(logical_view(unmasked), {});
  EXPECT_EQ(unmasked_result.row_count(), 3);
}

TEST(QepColumnMaterialization, MoveAndReleasePreserveViewsAndPublishOwningStates)
{
  auto source = materialization_source({owned_int64_materialization({10, 20})},
                                       materialization_positions({1, 0}));
  gqe::qep::state_container input{materialization_indirect(source, 0)};
  gqe::qep::column_materialization original =
    gqe::qep::materialize_all_columns(logical_view(input));
  auto const published_view = original.table_view().column(0);

  gqe::qep::column_materialization moved = std::move(original);
  EXPECT_EQ(moved.table_view().column(0).head(), published_view.head());

  auto states = std::move(moved).release_column_states();
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*states.front()));
  int64_column_wrapper expected{20, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_column_view(states.front()).value(), expected);
}

TEST(QepColumnMaterialization, ReleasedCompactedStatesRemainValidAfterInputRelease)
{
  std::optional<gqe::qep::state_container> released;
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("materialization-test");
  {
    auto mask = gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
      gqe::query_stream::default_stream());
    auto source = materialization_source({owned_int64_materialization({10, 20, 30})},
                                         materialization_positions({2, 1, 0}));
    gqe::qep::state_container input{
      mask, owned_int64_materialization({1, 2, 3}), materialization_indirect(source, 0)};
    auto materialized =
      gqe::qep::materialize_all_compacted_columns(logical_view(input), stream_handle);
    released.emplace(std::move(materialized).release_column_states());
  }

  stream_handle.ref().sync();
  auto const table = gqe::qep::to_table_view(*released);
  ASSERT_TRUE(table.has_value());
  int64_column_wrapper expected_direct{1, 3};
  int64_column_wrapper expected_indirect{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_direct);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_indirect);
}

TEST(QepColumnMaterialization, UsesExplicitStreamAndResultMemoryResource)
{
  auto source = materialization_source({owned_int64_materialization({10, 20, 30})},
                                       materialization_positions({2, 1, 0}));
  auto mask   = gqe::qep::make_shared_state(
    gqe::qep::state_kind::valid_mask{bool_column_wrapper{true, false, true}.release()},
    gqe::query_stream::default_stream());
  gqe::qep::state_container input{
    mask, owned_int64_materialization({1, 2, 3}), materialization_indirect(source, 0)};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("materialization-test");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  gqe::qep::column_materialization result = gqe::qep::materialize_compacted_columns(
    logical_view(input), std::vector<cudf::size_type>{0, 1}, true, stream_handle, result_mr);
  stream_handle.ref().sync();

  ASSERT_NE(result.row_positions(), nullptr);
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(result.row_positions()->column().view().head())));
  auto const output = result.table_view();
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(output.column(0).head())));
  EXPECT_TRUE(
    result_mr.get_outstanding_allocations().contains(const_cast<void*>(output.column(1).head())));
  int64_column_wrapper expected_direct{1, 3};
  int64_column_wrapper expected_indirect{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output.column(0), expected_direct);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output.column(1), expected_indirect);
}

TEST(QepColumnMaterialization, RejectsInvalidLogicalIndicesBeforeMaterialization)
{
  auto source =
    materialization_source({owned_int64_materialization({1})}, materialization_positions({100}));
  gqe::qep::state_container input{materialization_indirect(source, 0)};
  std::vector<cudf::size_type> negative{-1};
  std::vector<cudf::size_type> too_large{1};

  EXPECT_THROW(std::ignore = gqe::qep::materialize_columns(logical_view(input), negative),
               std::out_of_range);
  EXPECT_THROW(
    std::ignore = gqe::qep::materialize_compacted_columns(logical_view(input), too_large),
    std::out_of_range);
}

TEST(QepColumnMaterialization, RetainedRowPositionsHandleEmptyAndUnmaskedInputs)
{
  auto const stream = gqe::query_stream::default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  gqe::qep::state_container unmasked{owned_int64_materialization({10, 20, 30})};
  auto unchanged =
    gqe::qep::materialize_compacted_columns(logical_view(unmasked), {}, true, stream, mr);
  EXPECT_EQ(unchanged.row_positions(), nullptr);
  EXPECT_EQ(unchanged.row_count(), 3);

  for (bool keep : {false, true}) {
    auto mask = gqe::qep::make_shared_state(
      gqe::qep::state_kind::valid_mask{bool_column_wrapper{keep, false, keep}.release()}, stream);
    gqe::qep::state_container input{mask, unmasked.front()};
    auto selected =
      gqe::qep::materialize_compacted_columns(logical_view(input), {}, true, stream, mr);
    ASSERT_NE(selected.row_positions(), nullptr);
    EXPECT_TRUE(selected.states().empty());
    auto rows = selected.row_positions();
    input.clear();
    int32_column_wrapper expected = keep ? int32_column_wrapper{0, 2} : int32_column_wrapper{};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(rows->column().view(), expected);
    EXPECT_EQ(selected.row_count(), keep ? 2 : 0);
  }
}
