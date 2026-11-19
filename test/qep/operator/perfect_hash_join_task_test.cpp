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

#include "../lifetime_probe.hpp"
#include "join_task_test_util.hpp"

#include <gqe/qep/operator/perfect_hash_join.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe_test {

constexpr auto materialized_build = gqe::qep::perfect_hash_join_build_input::materialized;
constexpr auto streaming_build    = gqe::qep::perfect_hash_join_build_input::streaming;

/** @brief Provides fold, probe, and gather helpers for perfect-hash join task tests. */
class QepPerfectHashJoinTaskTest : public QepJoinTaskTestBase {};

TEST_F(QepPerfectHashJoinTaskTest, LogicalLeftBuildSharesRepeatedExecutionOrderProjections)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  // Exercise the cloned build layout across chunks and repeated projections.
  auto cloned_build =
    gqe::qep::clone_qep_task_as<gqe::qep::perfect_hash_join_build_task>(build_task);
  auto build =
    fold(*cloned_build, {two_int64_columns({1, 2}, {10, 20}), two_int64_columns({3}, {30})});
  gqe::qep::perfect_hash_join_probe_task probe_task{{0},
                                                    materialized_build,
                                                    {1, 3, 0, 1, 3},
                                                    {cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64}}};
  auto output = probe(probe_task, build, two_int64_columns({3, 1, 4}, {300, 100, 400}));

  ASSERT_EQ(output.size(), 5);
  auto const& build_value = *std::get<gqe::qep::state_kind::indirect_column>(*output[0]).state();
  auto const& probe_value = *std::get<gqe::qep::state_kind::indirect_column>(*output[1]).state();
  auto const& build_key   = *std::get<gqe::qep::state_kind::indirect_column>(*output[2]).state();
  auto const& repeated_build_value =
    *std::get<gqe::qep::state_kind::indirect_column>(*output[3]).state();
  auto const& repeated_probe_value =
    *std::get<gqe::qep::state_kind::indirect_column>(*output[4]).state();
  EXPECT_EQ(build_value.gather_source(), build_key.gather_source());
  EXPECT_EQ(build_value.gather_source(), repeated_build_value.gather_source());
  EXPECT_EQ(probe_value.gather_source(), repeated_probe_value.gather_source());
  EXPECT_NE(build_value.gather_source(), probe_value.gather_source());
  EXPECT_NE(build_value.gather_source()->relation_source(),
            probe_value.gather_source()->relation_source());
  EXPECT_EQ(build_value.source_column_index(), 0);
  EXPECT_EQ(probe_value.source_column_index(), 0);
  EXPECT_EQ(build_key.source_column_index(), 1);
  EXPECT_EQ(repeated_build_value.source_column_index(), 0);
  EXPECT_EQ(repeated_probe_value.source_column_index(), 0);
  EXPECT_EQ(build_value.gather_source()->relation_source()->num_columns(), 2);
  EXPECT_EQ(probe_value.gather_source()->relation_source()->num_columns(), 1);

  auto gathered = gather(output);
  auto table    = *gqe::qep::to_table_view(gathered);
  int64_column_wrapper expected_build_values{30, 10};
  int64_column_wrapper expected_probe_values{300, 100};
  int64_column_wrapper expected_build_keys{3, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_build_values);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_probe_values);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), expected_build_keys);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(3), expected_build_values);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(4), expected_probe_values);
}

TEST_F(QepPerfectHashJoinTaskTest, LogicalRightBuildUsesExecutionOrderProjection)
{
  std::vector<std::unique_ptr<cudf::column>> logical_right_columns;
  logical_right_columns.push_back(int64_column_wrapper{1, 2}.release());
  logical_right_columns.push_back(int32_column_wrapper{5, 6}.release());
  logical_right_columns.push_back(string_column_wrapper{"right-1", "right-2"}.release());
  gqe::qep::perfect_hash_join_build_task build_task{3, {0, 1}, {2}};
  auto build = fold(build_task, {table_state(std::move(logical_right_columns))});

  std::vector<std::unique_ptr<cudf::column>> logical_left_columns;
  logical_left_columns.push_back(int64_column_wrapper{2, 1, 9}.release());
  logical_left_columns.push_back(int32_column_wrapper{6, 5, 9}.release());
  logical_left_columns.push_back(string_column_wrapper{"left-2", "left-1", "none"}.release());
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0, 1},
    materialized_build,
    {5, 2},
    {cudf::data_type{cudf::type_id::STRING}, cudf::data_type{cudf::type_id::STRING}}};
  auto output   = probe(probe_task, build, table_state(std::move(logical_left_columns)));
  auto gathered = gather(output);
  auto table    = *gqe::qep::to_table_view(gathered);
  string_column_wrapper expected_left{"left-2", "left-1"};
  string_column_wrapper expected_right{"right-2", "right-1"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_left);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_right);
}

TEST_F(QepPerfectHashJoinTaskTest, StreamingBuildPreservesProjectionOrderAcrossCloning)
{
  for (bool const indirect : {false, true}) {
    SCOPED_TRACE(indirect);
    gqe::qep::perfect_hash_join_build_task materialize_task{2, {0}, {1}};
    auto materialized_input =
      indirect ? indirect_columns(two_int64_columns({1, 2, 2}, {100, 201, 200}), {2, 0, 1})
               : two_int64_columns({2, 1, 2}, {200, 100, 201});
    auto materialized    = fold(materialize_task, {materialized_input});
    auto streaming_input = indirect ? indirect_columns(two_int64_columns({2, 1}, {20, 10}), {1, 0})
                                    : two_int64_columns({1, 2}, {10, 20});
    gqe::qep::perfect_hash_join_probe_task probe_task{{0},
                                                      streaming_build,
                                                      {1, 3, 0, 2},
                                                      {cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64}}};

    auto output   = probe(probe_task, materialized, streaming_input);
    auto gathered = gather(output);
    auto table    = *gqe::qep::to_table_view(gathered);
    int64_column_wrapper expected_materialized_values{200, 100, 201};
    int64_column_wrapper expected_streaming_values{20, 10, 20};
    int64_column_wrapper expected_materialized_keys{2, 1, 2};
    int64_column_wrapper expected_streaming_keys{2, 1, 2};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_materialized_values);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_streaming_values);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(2), expected_materialized_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(3), expected_streaming_keys);

    auto clone = probe_task.clone();
    auto const* cloned_probe =
      dynamic_cast<gqe::qep::perfect_hash_join_probe_task const*>(clone.get());
    ASSERT_NE(cloned_probe, nullptr);
    auto cloned_output   = probe(*cloned_probe, materialized, streaming_input);
    auto cloned_gathered = gather(cloned_output);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*gqe::qep::to_table_view(cloned_gathered), table);
  }
}

TEST_F(QepPerfectHashJoinTaskTest, BuildAndProbeMasksTreatFalseAndNullAsInvalid)
{
  // Build keys 2 and 3 are excluded by a false and null mask entry, respectively.
  auto first_build_chunk =
    masked_two_int64_columns(bool_column_wrapper{true, false}.release(), {1, 2}, {10, 20});
  auto second_build_chunk =
    masked_two_int64_columns(bool_column_wrapper{{true, true, true}, {false, true, true}}.release(),
                             {3, 4, 5},
                             {30, 40, 50});
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build = fold(build_task, {first_build_chunk, second_build_chunk});
  // Probe keys 4 and 5 are excluded by a false and null mask entry, respectively.
  auto probe_chunk = masked_two_int64_columns(
    bool_column_wrapper{{true, true, true, false, true}, {true, true, true, true, false}}.release(),
    {1, 2, 3, 4, 5},
    {100, 200, 300, 400, 500});
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0},
    materialized_build,
    {1, 3},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};

  auto output   = probe(probe_task, build, probe_chunk);
  auto gathered = gather(output);
  auto table    = *gqe::qep::to_table_view(gathered);
  int64_column_wrapper expected_build{10};
  int64_column_wrapper expected_probe{100};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_build);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_probe);
}

TEST_F(QepPerfectHashJoinTaskTest, EmptyBuildProbeAndNoMatchRetainSchema)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, materialized_build, {1}, {cudf::data_type{cudf::type_id::INT64}}};

  auto empty_build       = fold(build_task, {});
  auto from_empty_build  = probe(probe_task, empty_build, two_int64_columns({1, 2}, {10, 20}));
  auto empty_build_table = *gqe::qep::to_table_view(from_empty_build);
  ASSERT_EQ(empty_build_table.num_columns(), 1);
  EXPECT_EQ(empty_build_table.num_rows(), 0);
  EXPECT_EQ(empty_build_table.column(0).type().id(), cudf::type_id::INT64);

  auto build       = fold(build_task, {two_int64_columns({1, 2}, {10, 20})});
  auto empty_probe = probe(probe_task, build, two_int64_columns({}, {}));
  EXPECT_EQ(gqe::qep::get_row_count(empty_probe), 0);
  auto gathered_empty_probe = gather(empty_probe);
  EXPECT_EQ(gqe::qep::to_table_view(gathered_empty_probe)->column(0).type().id(),
            cudf::type_id::INT64);

  auto no_match = probe(probe_task, build, two_int64_columns({8}, {80}));
  EXPECT_EQ(gqe::qep::get_row_count(no_match), 0);
  auto gathered_no_match = gather(no_match);
  EXPECT_EQ(gqe::qep::to_table_view(gathered_no_match)->num_rows(), 0);
}

TEST_F(QepPerfectHashJoinTaskTest, SchemaBearingZeroRowBuildProducesIndirectResult)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build = fold(build_task, {two_int64_columns({}, {})});
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, materialized_build, {1}, {cudf::data_type{cudf::type_id::INT64}}};

  auto output = probe(probe_task, build, two_int64_columns({1, 2}, {10, 20}));
  ASSERT_EQ(output.size(), 1);
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::indirect_column>(*output.front()));
  EXPECT_EQ(gqe::qep::get_row_count(output), 0);

  auto gathered = gather(output);
  auto table    = gqe::qep::to_table_view(gathered);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 1);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->column(0).type().id(), cudf::type_id::INT64);
}

TEST_F(QepPerfectHashJoinTaskTest, EmptyProjectionPreservesMatchCount)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {}};
  bool is_unused_build_payload_destroyed = false;
  auto build_input                       = two_int64_columns({1, 2}, {10, 20});
  build_input[1] =
    gqe::test::track_lifetime(std::move(build_input[1]), is_unused_build_payload_destroyed);
  auto build = fold(build_task, {build_input});
  build_input.clear();
  EXPECT_TRUE(is_unused_build_payload_destroyed);
  gqe::qep::perfect_hash_join_probe_task probe_task{{0}, materialized_build, {}, {}};
  auto output = probe(probe_task, build, two_int64_columns({2, 1, 9}, {20, 10, 90}));
  EXPECT_EQ(gqe::qep::try_row_count(output), 2);
  auto gathered = gather(output);
  EXPECT_EQ(gqe::qep::try_row_count(gathered), 2);
}

TEST_F(QepPerfectHashJoinTaskTest, SupportsSlicedInputsAndNullableProjectedColumns)
{
  int64_column_wrapper build_key_owner{0, 1, 2, 3};
  int64_column_wrapper build_value_owner{{0, 10, 20, 30}, {true, true, false, true}};
  auto build_keys   = cudf::slice(build_key_owner, {1, 4}).front();
  auto build_values = cudf::slice(build_value_owner, {1, 4}).front();
  auto build_chunk  = gqe::qep::state_container_builder()
                       .add_state(cudf::table_view{{build_keys, build_values}})
                       .build();
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build = fold(build_task, {build_chunk});

  int64_column_wrapper probe_key_owner{9, 3, 2, 8};
  int64_column_wrapper probe_value_owner{90, 30, 20, 80};
  auto probe_keys   = cudf::slice(probe_key_owner, {1, 3}).front();
  auto probe_values = cudf::slice(probe_value_owner, {1, 3}).front();
  auto probe_chunk  = gqe::qep::state_container_builder()
                       .add_state(cudf::table_view{{probe_keys, probe_values}})
                       .build();
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, materialized_build, {1}, {cudf::data_type{cudf::type_id::INT64}}};
  auto output   = probe(probe_task, build, probe_chunk);
  auto gathered = gather(output);
  auto table    = *gqe::qep::to_table_view(gathered);
  int64_column_wrapper expected{{30, 0}, {true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(0), expected);
}

TEST_F(QepPerfectHashJoinTaskTest, ComposesMaskedIndirectProbePositionsAndPreservesSource)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build_input = indirect_columns(two_int64_columns({3, 1, 2}, {3000, 1000, 2000}), {1, 2, 0});
  auto build       = fold(build_task, {build_input});

  std::vector<std::unique_ptr<cudf::column>> probe_source_columns;
  probe_source_columns.push_back(int64_column_wrapper{1, 2, 3}.release());
  probe_source_columns.push_back(int64_column_wrapper{10, 20, 30}.release());
  probe_source_columns.push_back(int64_column_wrapper{100, 200, 300}.release());
  auto probe_columns = indirect_columns(table_state(std::move(probe_source_columns)),
                                        {2, 0, 2, 1},
                                        cudf::out_of_bounds_policy::NULLIFY);
  auto const& original_descriptor =
    *std::get<gqe::qep::state_kind::indirect_column>(*probe_columns[1]).state();
  auto const original_relation = original_descriptor.gather_source()->relation_source();
  auto probe_input = with_valid_mask(bool_column_wrapper{true, false, true, true}.release(),
                                     std::move(probe_columns));

  gqe::qep::perfect_hash_join_probe_task probe_task{{0},
                                                    materialized_build,
                                                    {3, 1, 4, 3},
                                                    {cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT64}}};
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto output = probe(probe_task, build, probe_input, stream, result_mr);
  stream.ref().sync();

  auto const& first_probe    = *std::get<gqe::qep::state_kind::indirect_column>(*output[0]).state();
  auto const& second_probe   = *std::get<gqe::qep::state_kind::indirect_column>(*output[2]).state();
  auto const& repeated_probe = *std::get<gqe::qep::state_kind::indirect_column>(*output[3]).state();
  EXPECT_EQ(first_probe.gather_source(), second_probe.gather_source());
  EXPECT_EQ(first_probe.gather_source(), repeated_probe.gather_source());
  EXPECT_EQ(first_probe.gather_source()->relation_source(), original_relation);
  EXPECT_EQ(first_probe.gather_source()->bounds_policy(), cudf::out_of_bounds_policy::NULLIFY);
  EXPECT_EQ(first_probe.source_column_index(), 1);
  EXPECT_EQ(second_probe.source_column_index(), 2);
  EXPECT_EQ(repeated_probe.source_column_index(), 1);
  int32_column_wrapper expected_positions{2, 2, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_probe.gather_source()->position_list()->column().view(),
                                 expected_positions);
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(first_probe.gather_source()->position_list()->column().view().head())));

  auto gathered = gather(output);
  auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{20, 30, 30}.release());
  expected_columns.push_back(int64_column_wrapper{2000, 3000, 3000}.release());
  expected_columns.push_back(int64_column_wrapper{200, 300, 300}.release());
  expected_columns.push_back(int64_column_wrapper{20, 30, 30}.release());
  cudf::table expected{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted->view(), expected.view());
}

TEST_F(QepPerfectHashJoinTaskTest, IndirectProbeResultRetainsOnlyProjectedSourceColumns)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {}};
  auto build = fold(build_task, {two_int64_columns({1, 2}, {10, 20})});
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, materialized_build, {3}, {cudf::data_type{cudf::type_id::INT64}}};
  auto probe_input        = two_int64_columns({2, 1}, {200, 100});
  bool is_key_destroyed   = false;
  bool is_value_destroyed = false;
  probe_input[0]          = gqe::test::track_lifetime(std::move(probe_input[0]), is_key_destroyed);
  probe_input[1] = gqe::test::track_lifetime(std::move(probe_input[1]), is_value_destroyed);

  auto output = probe(probe_task, build, probe_input);
  probe_input.clear();

  // Match positions own independent storage after the join has ordered its temporary inputs.
  EXPECT_TRUE(is_key_destroyed);
  EXPECT_FALSE(is_value_destroyed);
  auto const& indirect_state =
    *std::get<gqe::qep::state_kind::indirect_column>(*output.front()).state();
  EXPECT_EQ(indirect_state.source_column_index(), 0);
  EXPECT_EQ(indirect_state.gather_source()->relation_source()->num_columns(), 1);

  auto gathered = gather(output);
  int64_column_wrapper expected{200, 100};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(gqe::qep::to_table_view(gathered)->column(0), expected);
  gathered.clear();
  output.clear();
  EXPECT_TRUE(is_key_destroyed);
  EXPECT_TRUE(is_value_destroyed);
}

TEST_F(QepPerfectHashJoinTaskTest, PublishedPositionListsUseResultResource)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build = fold(build_task, {two_int64_columns({1, 2}, {10, 20})});
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0},
    materialized_build,
    {1, 3},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};

  auto output =
    probe(probe_task, build, two_int64_columns({2, 1}, {20, 10}), stream_handle, result_mr);
  stream_handle.ref().sync();
  auto const& build_indirect = *std::get<gqe::qep::state_kind::indirect_column>(*output[0]).state();
  auto const& probe_indirect = *std::get<gqe::qep::state_kind::indirect_column>(*output[1]).state();
  auto const& build_positions = build_indirect.gather_source()->position_list();
  auto const& probe_positions = probe_indirect.gather_source()->position_list();
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(build_positions->column().view().head())));
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(probe_positions->column().view().head())));
}

TEST_F(QepPerfectHashJoinTaskTest, CompactsMaskedBuildInBothOrientations)
{
  // Two of five build rows are selected. Build keys 2 and 4 are rejected by false entries and key
  // 3 by a null entry.
  auto const make_build_chunks = [&] {
    return std::vector{
      masked_two_int64_columns(bool_column_wrapper{true, false}.release(), {1, 2}, {10, 20}),
      masked_two_int64_columns(
        bool_column_wrapper{{true, false, true}, {false, true, true}}.release(),
        {3, 4, 5},
        {30, 40, 50})};
  };
  // Probe key 5 appears twice, and its second row is rejected by the probe mask.
  auto const make_probe_chunk = [&] {
    return masked_two_int64_columns(
      bool_column_wrapper{true, true, true, true, true, false}.release(),
      {5, 2, 1, 4, 3, 5},
      {500, 200, 100, 400, 300, 501});
  };

  for (auto const build_input : {materialized_build, streaming_build}) {
    SCOPED_TRACE(::testing::Message() << "streaming=" << (build_input == streaming_build));
    gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
    auto build = fold(build_task, make_build_chunks());
    gqe::qep::perfect_hash_join_probe_task probe_task{{0},
                                                      build_input,
                                                      {0, 1, 3},
                                                      {cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64}}};

    auto output = probe(probe_task, build, make_probe_chunk());
    // Only the two selected rows remain in the build relation.
    auto const& build_indirect =
      *std::get<gqe::qep::state_kind::indirect_column>(*output[0]).state();
    EXPECT_EQ(build_indirect.gather_source()->relation_source()->row_count(), 2);
    auto gathered = gather(output);
    auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
    int64_column_wrapper expected_build_keys{1, 5};
    int64_column_wrapper expected_build_values{10, 50};
    int64_column_wrapper expected_probe_values{100, 500};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0).view(), expected_build_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(1).view(), expected_build_values);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(2).view(), expected_probe_values);
  }
}

TEST_F(QepPerfectHashJoinTaskTest, CompactsBuildWithNoSelectedRowsToEmptyJoin)
{
  // One chunk rejects every row with false entries, the other with null entries, and the empty
  // chunk adds no rows.
  auto const make_build_chunks = [&] {
    return std::vector{
      masked_two_int64_columns(bool_column_wrapper{false, false}.release(), {1, 2}, {10, 20}),
      masked_two_int64_columns(bool_column_wrapper{}.release(), {}, {}),
      masked_two_int64_columns(
        bool_column_wrapper{{true, true}, {false, false}}.release(), {3, 4}, {30, 40})};
  };

  for (auto const build_input : {materialized_build, streaming_build}) {
    SCOPED_TRACE(::testing::Message() << "streaming=" << (build_input == streaming_build));
    gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
    auto build = fold(build_task, make_build_chunks());
    gqe::qep::perfect_hash_join_probe_task probe_task{{0},
                                                      build_input,
                                                      {0, 1, 3},
                                                      {cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64},
                                                       cudf::data_type{cudf::type_id::INT64}}};

    auto output   = probe(probe_task, build, two_int64_columns({1, 2, 3, 4}, {100, 200, 300, 400}));
    auto gathered = gather(output);
    EXPECT_EQ(gqe::qep::to_table_view(gathered)->num_rows(), 0);
  }
}

TEST_F(QepPerfectHashJoinTaskTest, AcceptsMaskedAndUnmaskedBuildChunksInOneFold)
{
  // Compaction removes the mask from each chunk, so masked and unmasked chunks can share a fold.
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build =
    fold(build_task,
         {masked_two_int64_columns(bool_column_wrapper{false, true}.release(), {1, 2}, {10, 20}),
          two_int64_columns({3}, {30})});
  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, materialized_build, {1}, {cudf::data_type{cudf::type_id::INT64}}};

  auto output   = probe(probe_task, build, two_int64_columns({1, 2, 3}, {100, 200, 300}));
  auto gathered = gather(output);
  auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
  int64_column_wrapper expected_build_values{20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0).view(), expected_build_values);
}

TEST_F(QepPerfectHashJoinTaskTest, RejectsInvalidConfigurationAndInputs)
{
  EXPECT_THROW(std::ignore = gqe::qep::perfect_hash_join_build_task(0, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::qep::perfect_hash_join_probe_task({0}, materialized_build, {0}, {}),
    std::invalid_argument);
  for (auto const type_id :
       {cudf::type_id::LIST, cudf::type_id::STRUCT, cudf::type_id::DICTIONARY32}) {
    EXPECT_THROW(std::ignore = gqe::qep::perfect_hash_join_probe_task(
                   {0}, materialized_build, {0}, {cudf::data_type{type_id}}),
                 std::invalid_argument);
  }

  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {}};
  auto accumulator = build_task.initialize(ctx_ref);
  EXPECT_THROW(build_task.next(gqe::qep::make_row_count_container(1), accumulator, ctx_ref),
               std::logic_error);

  auto build = fold(build_task, {two_int64_columns({1}, {10})});
  gqe::qep::perfect_hash_join_probe_task bad_key{
    {2}, materialized_build, {0}, {cudf::data_type{cudf::type_id::INT64}}};
  auto probe_accumulator = bad_key.initialize(build, ctx_ref);
  EXPECT_THROW(std::ignore = bad_key.next(two_int64_columns({1}, {10}), probe_accumulator, ctx_ref),
               std::logic_error);

  gqe::qep::state_container invalid_payload = gqe::qep::make_row_count_container(1);
  gqe::qep::perfect_hash_join_probe_task valid{{0}, materialized_build, {}, {}};
  EXPECT_THROW(std::ignore = valid.initialize(invalid_payload, ctx_ref), std::logic_error);
}

TEST_F(QepPerfectHashJoinTaskTest, RejectsConfiguredOutputTypeMismatch)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {1}};
  auto build       = fold(build_task, {two_int64_columns({1}, {10})});
  auto probe_input = two_int64_columns({1}, {100});

  auto expect_mismatch = [&](cudf::size_type projection) {
    gqe::qep::perfect_hash_join_probe_task probe_task{
      {0}, materialized_build, {projection}, {cudf::data_type{cudf::type_id::INT32}}};
    auto accumulator = probe_task.initialize(build, ctx_ref);
    EXPECT_THROW(std::ignore = probe_task.next(probe_input, accumulator, ctx_ref),
                 std::invalid_argument);
    probe_task.finalize(std::move(accumulator), ctx_ref);
  };

  expect_mismatch(1);  // Build-side INT64 value.
  expect_mismatch(3);  // Probe-side INT64 value.
}

TEST_F(QepPerfectHashJoinTaskTest, ClonePreservesTaskTypes)
{
  gqe::qep::perfect_hash_join_build_task build_task{2, {0}, {}};
  auto cloned_build = build_task.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::perfect_hash_join_build_task*>(cloned_build.get()), nullptr);

  gqe::qep::perfect_hash_join_probe_task probe_task{
    {0}, streaming_build, {0}, {cudf::data_type{cudf::type_id::INT64}}};
  auto cloned_probe = probe_task.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::perfect_hash_join_probe_task*>(cloned_probe.get()), nullptr);
}

}  // namespace gqe_test
