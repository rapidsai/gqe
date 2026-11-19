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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/qep/operator/filter_join_indices.hpp>
#include <gqe/qep/operator/hash_join.hpp>
#include <gqe/qep/operator/unique_key_inner_join.hpp>
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
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gqe_test {

/** @brief Checks general and unique-key hash joins against the same input and output contracts. */
class QepHashJoinTaskTest : public QepJoinTaskTestBase {};

/**
 * @brief Runs shared join contracts for both concrete task pairs.
 *
 * @tparam UseUniqueKeys Boolean constant selecting the concrete task pair under test.
 */
template <typename UseUniqueKeys>
class QepHashJoinImplementationsTest : public QepHashJoinTaskTest {
 protected:
  static auto make_build_task(cudf::size_type column_count,
                              std::vector<cudf::size_type> keys,
                              std::vector<cudf::size_type> payloads,
                              cudf::null_equality null_equality,
                              bool cache,
                              std::vector<cudf::data_type> key_types)
  {
    if constexpr (UseUniqueKeys::value) {
      return gqe::qep::unique_key_inner_join_build_task{column_count,
                                                        std::move(keys),
                                                        std::move(payloads),
                                                        null_equality,
                                                        cache,
                                                        std::move(key_types)};
    } else {
      return gqe::qep::hash_join_build_task{
        column_count, std::move(keys), std::move(payloads), null_equality, cache};
    }
  }

  static auto make_inner_probe_task(std::vector<cudf::size_type> keys,
                                    std::vector<cudf::size_type> projections,
                                    std::vector<cudf::data_type> output_types)
  {
    if constexpr (UseUniqueKeys::value) {
      return gqe::qep::unique_key_inner_join_probe_task{
        std::move(keys), std::move(projections), std::move(output_types)};
    } else {
      return gqe::qep::hash_join_probe_task{std::move(keys),
                                            gqe::join_type_type::inner,
                                            std::move(projections),
                                            std::move(output_types)};
    }
  }
};

using HashJoinImplementations = ::testing::Types<std::false_type, std::true_type>;
TYPED_TEST_SUITE(QepHashJoinImplementationsTest, HashJoinImplementations);

TEST_F(QepHashJoinTaskTest, InnerJoinSupportsDuplicateKeysAndRepeatedProjections)
{
  gqe::qep::hash_join_build_task build_task{
    2, {0}, {1}, cudf::null_equality::UNEQUAL, /*is_hash_map_cache_enabled=*/true};
  // Exercise the cloned build layout across chunks and repeated projections.
  auto cloned_build = gqe::qep::clone_qep_task_as<gqe::qep::hash_join_build_task>(build_task);
  auto build =
    fold(*cloned_build, {two_int64_columns({1, 1}, {10, 11}), two_int64_columns({2}, {20})});
  gqe::qep::hash_join_probe_task probe_task{{0},
                                            gqe::join_type_type::inner,
                                            {1, 3, 1},
                                            {cudf::data_type{cudf::type_id::INT64},
                                             cudf::data_type{cudf::type_id::INT64},
                                             cudf::data_type{cudf::type_id::INT64}}};
  auto output = gather(probe(probe_task, build, two_int64_columns({1, 2}, {100, 200})));
  auto sorted = cudf::sort(*gqe::qep::to_table_view(output));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{10, 11, 20}.release());
  expected_columns.push_back(int64_column_wrapper{100, 100, 200}.release());
  expected_columns.push_back(int64_column_wrapper{10, 11, 20}.release());
  cudf::table expected{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected.view());
}

TEST_F(QepHashJoinTaskTest, LeftJoinNullifiesUnmatchedBuildRows)
{
  for (bool const cache_enabled : {false, true}) {
    gqe::qep::hash_join_build_task build_task{
      2, {0}, {1}, cudf::null_equality::UNEQUAL, cache_enabled};
    auto build = fold(build_task, {two_int64_columns({1, 2}, {10, 20})});
    gqe::qep::hash_join_probe_task probe_task{
      {0},
      gqe::join_type_type::left,
      {1, 3},
      {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
    auto output = gather(probe(probe_task, build, two_int64_columns({2, 3}, {200, 300})));
    auto table  = *gqe::qep::to_table_view(output);
    int64_column_wrapper expected_build{{20, 0}, {true, false}};
    int64_column_wrapper expected_probe{200, 300};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(0), expected_build);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_probe);
  }
}

TEST_F(QepHashJoinTaskTest, SwappedInputsProvideRightJoinEquivalentSemantics)
{
  gqe::qep::hash_join_build_task build_task{
    2, {0}, {1}, cudf::null_equality::UNEQUAL, /*is_hash_map_cache_enabled=*/true};
  auto logical_left = fold(build_task, {two_int64_columns({1, 2}, {10, 20})});
  gqe::qep::hash_join_probe_task probe_task{
    {0},
    gqe::join_type_type::left,
    {3, 1},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  auto output = gather(probe(probe_task, logical_left, two_int64_columns({2, 3}, {200, 300})));
  auto table  = *gqe::qep::to_table_view(output);
  int64_column_wrapper expected_right{200, 300};
  int64_column_wrapper expected_left{{20, 0}, {true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_right);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(table.column(1), expected_left);
}

TYPED_TEST(QepHashJoinImplementationsTest, CompositeKeysHonorBothNullEqualityPolicies)
{
  for (auto const null_equality : {cudf::null_equality::UNEQUAL, cudf::null_equality::EQUAL}) {
    std::vector<std::unique_ptr<cudf::column>> build_columns;
    build_columns.push_back(int64_column_wrapper{{1, 2}, {true, false}}.release());
    build_columns.push_back(int32_column_wrapper{5, 6}.release());
    build_columns.push_back(int64_column_wrapper{10, 20}.release());
    auto build_task = this->make_build_task(
      3,
      {0, 1},
      {2},
      null_equality,
      true,
      {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}});
    auto build = this->fold(build_task, {table_state(std::move(build_columns))});

    std::vector<std::unique_ptr<cudf::column>> probe_columns;
    probe_columns.push_back(int64_column_wrapper{{1, 9}, {true, false}}.release());
    probe_columns.push_back(int32_column_wrapper{5, 6}.release());
    probe_columns.push_back(int64_column_wrapper{100, 200}.release());
    auto probe_task = this->make_inner_probe_task(
      {0, 1},
      {2, 5},
      {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}});
    auto output =
      this->gather(this->probe(probe_task, build, table_state(std::move(probe_columns))));
    auto table = *gqe::qep::to_table_view(output);
    EXPECT_EQ(table.num_rows(), null_equality == cudf::null_equality::EQUAL ? 2 : 1);
  }
}

TYPED_TEST(QepHashJoinImplementationsTest, MasksCompactFalseAndNullRowsOnBothInputs)
{
  auto build_chunk = masked_two_int64_columns(
    bool_column_wrapper{{true, false, true}, {true, true, false}}.release(),
    {1, 2, 3},
    {10, 20, 30});
  auto build_task = this->make_build_task(
    2, {0}, {1}, cudf::null_equality::UNEQUAL, false, {cudf::data_type{cudf::type_id::INT64}});
  auto build = this->fold(build_task, {build_chunk, two_int64_columns({4}, {40})});

  auto probe_chunk = masked_two_int64_columns(
    bool_column_wrapper{{true, true, false, true}, {true, true, true, false}}.release(),
    {1, 2, 3, 4},
    {100, 200, 300, 400});
  auto probe_task = this->make_inner_probe_task(
    {0}, {1, 3}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}});
  auto output = this->gather(this->probe(probe_task, build, probe_chunk));
  auto table  = *gqe::qep::to_table_view(output);
  int64_column_wrapper expected_build{10};
  int64_column_wrapper expected_probe{100};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), expected_build);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), expected_probe);
}

TYPED_TEST(QepHashJoinImplementationsTest,
           SelectivelyCompactsMaskedIndirectInputsInLogicalCoordinates)
{
  bool is_build_unused_destroyed = false;
  bool is_probe_unused_destroyed = false;
  auto build_columns = indirect_columns(two_int64_columns({1, 2, 3}, {10, 20, 30}), {2, 0, 1});
  auto build_unused  = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{70, 80, 90}.release()},
    gqe::query_stream::default_stream());
  build_unused = gqe::test::track_lifetime(std::move(build_unused), is_build_unused_destroyed);
  build_columns.push_back(std::move(build_unused));
  auto build_input =
    with_valid_mask(bool_column_wrapper{true, false, true}.release(), std::move(build_columns));

  auto build_task = this->make_build_task(
    3,
    {0, 0},
    {1, 1},
    cudf::null_equality::UNEQUAL,
    true,
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}});
  auto build = this->fold(build_task, {build_input});
  build_input.clear();
  EXPECT_TRUE(is_build_unused_destroyed);

  auto probe_columns = indirect_columns(two_int64_columns({2, 3, 9}, {200, 300, 900}), {1, 0, 2});
  auto probe_unused  = gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper{7, 8, 9}.release()},
    gqe::query_stream::default_stream());
  probe_unused = gqe::test::track_lifetime(std::move(probe_unused), is_probe_unused_destroyed);
  probe_columns.push_back(std::move(probe_unused));
  auto probe_input =
    with_valid_mask(bool_column_wrapper{true, true, false}.release(), std::move(probe_columns));

  auto probe_task = this->make_inner_probe_task({0, 0},
                                                {1, 4, 1},
                                                {cudf::data_type{cudf::type_id::INT64},
                                                 cudf::data_type{cudf::type_id::INT64},
                                                 cudf::data_type{cudf::type_id::INT64}});
  auto output     = this->probe(probe_task, build, probe_input);
  probe_input.clear();
  EXPECT_TRUE(is_probe_unused_destroyed);

  auto gathered = this->gather(output);
  auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{20, 30}.release());
  expected_columns.push_back(int64_column_wrapper{200, 300}.release());
  expected_columns.push_back(int64_column_wrapper{20, 30}.release());
  cudf::table expected{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected.view());
}

TEST_F(QepHashJoinTaskTest, ChainedMaskedProbesPreserveStringSourcesAndDuplicateMatches)
{
  gqe::qep::hash_join_build_task build_task{2, {0}, {1}, cudf::null_equality::UNEQUAL, true};
  auto build = fold(build_task, {two_int64_columns({2, 2, 3}, {20, 21, 30})});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{1, 2, 3, 9}.release());
  columns.push_back(string_column_wrapper{"one", "café", "東京", "nine"}.release());
  auto input_columns = indirect_columns(table_state(std::move(columns)), {3, 1, 0, 2, 1});
  auto const source  = std::get<gqe::qep::state_kind::indirect_column>(*input_columns[1])
                        .state()
                        ->gather_source()
                        ->relation_source();
  // Keep noncontiguous rows. A null mask entry must be discarded like a false entry.
  auto input = with_valid_mask(
    bool_column_wrapper{{true, true, true, true, false}, {true, true, false, true, true}}.release(),
    std::move(input_columns));
  gqe::qep::hash_join_probe_task first{{0},
                                       gqe::join_type_type::left,
                                       {2, 3, 1, 3},
                                       {cudf::data_type{cudf::type_id::INT64},
                                        cudf::data_type{cudf::type_id::STRING},
                                        cudf::data_type{cudf::type_id::INT64},
                                        cudf::data_type{cudf::type_id::STRING}}};
  cudf::get_default_stream().sync();
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto first_output = probe(first, build, input, stream, result_mr);
  input.clear();
  auto const first_source =
    std::get<gqe::qep::state_kind::indirect_column>(*first_output[1]).state()->gather_source();
  EXPECT_EQ(first_source->relation_source(), source);
  EXPECT_EQ(
    first_source,
    std::get<gqe::qep::state_kind::indirect_column>(*first_output[3]).state()->gather_source());
  // A second left join preserves the first join's null-producing build payload as probe data.
  gqe::qep::hash_join_probe_task second{
    {0},
    gqe::join_type_type::left,
    {3, 4},
    {cudf::data_type{cudf::type_id::STRING}, cudf::data_type{cudf::type_id::INT64}}};
  auto output = probe(second, build, first_output, stream, result_mr);
  first_output.clear();
  build.clear();
  stream.ref().sync();
  auto const output_source =
    std::get<gqe::qep::state_kind::indirect_column>(*output[0]).state()->gather_source();
  EXPECT_EQ(output_source->relation_source(), source);
  EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
    const_cast<void*>(output_source->position_list()->column().view().head())));
  auto gathered = gather(output);
  auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(
    string_column_wrapper{"café", "café", "café", "café", "nine", "東京"}.release());
  expected_columns.push_back(
    int64_column_wrapper{{20, 20, 21, 21, 0, 30}, {true, true, true, true, false, true}}.release());
  cudf::table expected{std::move(expected_columns)};
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted->view(), expected.view());
}

TYPED_TEST(QepHashJoinImplementationsTest, FullyMaskedProbeProducesEmptyIndirectPayload)
{
  auto build_task = this->make_build_task(
    2, {0}, {}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto build    = this->fold(build_task, {two_int64_columns({1}, {10})});
  auto input    = with_valid_mask(bool_column_wrapper{false, false}.release(),
                               indirect_columns(two_int64_columns({1, 2}, {10, 20}), {1, 0}));
  auto task     = this->make_inner_probe_task({0}, {3}, {cudf::data_type{cudf::type_id::INT64}});
  auto output   = this->probe(task, build, input);
  auto gathered = this->gather(output);
  EXPECT_EQ(gqe::qep::to_table_view(gathered)->num_rows(), 0);
}

TEST_F(QepHashJoinTaskTest, FullyMaskedLeftProbeProducesEmptyIndirectPayload)
{
  gqe::qep::hash_join_build_task build_task(2, {0}, {}, cudf::null_equality::UNEQUAL, true);
  auto build = fold(build_task, {two_int64_columns({1}, {10})});
  auto input = with_valid_mask(bool_column_wrapper{false, false}.release(),
                               indirect_columns(two_int64_columns({1, 2}, {10, 20}), {1, 0}));
  gqe::qep::hash_join_probe_task probe_task(
    {0}, gqe::join_type_type::left, {3}, {cudf::data_type{cudf::type_id::INT64}});
  auto output = gather(probe(probe_task, build, input));
  EXPECT_EQ(gqe::qep::to_table_view(output)->num_rows(), 0);
}

TYPED_TEST(QepHashJoinImplementationsTest, EmptyAndNoMatchInputsRetainSchemaAndCounts)
{
  auto build_task = this->make_build_task(
    2, {0}, {1}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto build       = this->fold(build_task, {two_int64_columns({1}, {10})});
  auto inner_probe = this->make_inner_probe_task({0}, {}, {});
  auto count       = this->probe(inner_probe, build, two_int64_columns({9}, {90}));
  EXPECT_EQ(gqe::qep::try_row_count(count), 0);

  auto projected_probe =
    this->make_inner_probe_task({0}, {1}, {cudf::data_type{cudf::type_id::INT64}});
  auto empty_probe = this->gather(this->probe(projected_probe, build, two_int64_columns({}, {})));
  EXPECT_EQ(gqe::qep::to_table_view(empty_probe)->num_rows(), 0);
  EXPECT_EQ(gqe::qep::to_table_view(empty_probe)->column(0).type().id(), cudf::type_id::INT64);
}

TYPED_TEST(QepHashJoinImplementationsTest, EmptyBuildWithOrWithoutSchemaProducesNoMatches)
{
  for (bool cache : {false, true}) {
    auto build_task = this->make_build_task(
      2, {0}, {1}, cudf::null_equality::UNEQUAL, cache, {cudf::data_type{cudf::type_id::INT64}});
    for (bool has_chunk : {false, true}) {
      std::vector<gqe::qep::state_container> chunks;
      if (has_chunk) { chunks.push_back(two_int64_columns({}, {})); }
      auto build       = this->fold(build_task, chunks);
      auto count_probe = this->make_inner_probe_task({0}, {}, {});
      EXPECT_EQ(
        gqe::qep::try_row_count(this->probe(count_probe, build, two_int64_columns({1}, {10}))), 0);
      auto projected_probe =
        this->make_inner_probe_task({0}, {1}, {cudf::data_type{cudf::type_id::INT64}});
      auto output = this->gather(this->probe(projected_probe, build, two_int64_columns({1}, {10})));
      EXPECT_EQ(gqe::qep::to_table_view(output)->num_rows(), 0);
      EXPECT_EQ(gqe::qep::to_table_view(output)->column(0).type(),
                cudf::data_type{cudf::type_id::INT64});
    }
  }
}

TEST_F(QepHashJoinTaskTest, EmptyBuildPreservesLeftProbeRows)
{
  gqe::qep::hash_join_build_task build_task(2, {0}, {1}, cudf::null_equality::UNEQUAL, true);
  auto empty_build = fold(build_task, {two_int64_columns({}, {})});
  gqe::qep::hash_join_probe_task left_probe{
    {0},
    gqe::join_type_type::left,
    {1, 3},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  auto left_output = gather(probe(left_probe, empty_build, two_int64_columns({1}, {100})));
  auto left_table  = *gqe::qep::to_table_view(left_output);
  EXPECT_EQ(left_table.num_rows(), 1);
  EXPECT_EQ(left_table.column(0).null_count(), 1);
}

TEST_F(QepHashJoinTaskTest, CountOnlyOutputRejectsRowsBeyondCurrentStateLimit)
{
  constexpr std::size_t rows_per_input = 46'341;
  std::vector<int64_t> repeated_keys(rows_per_input, 1);
  gqe::qep::hash_join_build_task build_task{2, {0}, {}, cudf::null_equality::UNEQUAL, true};
  auto build = fold(build_task, {two_int64_columns(repeated_keys, repeated_keys)});
  gqe::qep::hash_join_probe_task probe_task{{0}, gqe::join_type_type::inner, {}, {}};
  auto accumulator = probe_task.initialize(build, ctx_ref);

  EXPECT_THROW(std::ignore = probe_task.next(
                 two_int64_columns(repeated_keys, repeated_keys), accumulator, ctx_ref),
               std::overflow_error);
  probe_task.finalize(std::move(accumulator), ctx_ref);
}

TYPED_TEST(QepHashJoinImplementationsTest,
           SupportsSlicedStringPayloadsAndRetainsOnlyProjectedColumns)
{
  int64_column_wrapper build_key_owner{0, 1, 2, 3};
  string_column_wrapper build_value_owner{"zero", "un caf\xC3\xA9", "two", "three"};
  auto build_keys   = cudf::slice(build_key_owner, {1, 4}).front();
  auto build_values = cudf::slice(build_value_owner, {1, 4}).front();
  auto build_chunk  = gqe::qep::state_container_builder()
                       .add_state(cudf::table_view{{build_keys, build_values}})
                       .build();
  auto build_task = this->make_build_task(
    2, {0}, {1}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto build = this->fold(build_task, {build_chunk});

  auto probe_input              = two_int64_columns({3, 1}, {300, 100});
  bool is_unprojected_destroyed = false;
  probe_input[1]  = gqe::test::track_lifetime(std::move(probe_input[1]), is_unprojected_destroyed);
  auto probe_task = this->make_inner_probe_task({0}, {1}, {cudf::data_type{cudf::type_id::STRING}});
  auto output     = this->probe(probe_task, build, probe_input);
  probe_input.clear();
  EXPECT_TRUE(is_unprojected_destroyed);
  auto gathered = this->gather(output);
  auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
  string_column_wrapper expected{"three", "un caf\xC3\xA9"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expected);
}

TYPED_TEST(QepHashJoinImplementationsTest, CachedStateSupportsConcurrentProbes)
{
  auto build_task = this->make_build_task(
    2, {0}, {}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto build = this->fold(build_task, {two_int64_columns({1, 2, 3}, {10, 20, 30})});
  cudf::get_default_stream().sync();
  auto probe_task = this->make_inner_probe_task({0}, {}, {});

  auto run_probe = [&](int64_t key) {
    gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
    auto const stream = stream_context.create_stream("test/stream");
    auto result = this->probe(probe_task, build, two_int64_columns({key}, {key * 100}), stream);
    stream.ref().sync();
    return gqe::qep::try_row_count(result);
  };
  auto first  = std::async(std::launch::async, run_probe, int64_t{1});
  auto second = std::async(std::launch::async, run_probe, int64_t{3});
  EXPECT_EQ(first.get(), 1);
  EXPECT_EQ(second.get(), 1);
}

TYPED_TEST(QepHashJoinImplementationsTest, PositionListsUseSuppliedResultResource)
{
  auto build_task = this->make_build_task(
    2, {0}, {1}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto build      = this->fold(build_task, {two_int64_columns({1}, {10})});
  auto probe_task = this->make_inner_probe_task(
    {0}, {1, 3}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}});
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto output = this->probe(probe_task, build, two_int64_columns({1}, {100}), stream, result_mr);
  stream.ref().sync();
  for (auto const& state : output) {
    auto const& indirect = *std::get<gqe::qep::state_kind::indirect_column>(*state).state();
    EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
      const_cast<void*>(indirect.gather_source()->position_list()->column().view().head())));
  }
}

TEST_F(QepHashJoinTaskTest, MixedCandidatesFilterBeforeProjectionAndSharePositionLists)
{
  std::vector<std::unique_ptr<cudf::column>> build_columns;
  build_columns.push_back(int64_column_wrapper{1, 1, 2, 3}.release());
  build_columns.push_back(int64_column_wrapper{{5, 20, 7, 0}, {true, true, true, false}}.release());
  build_columns.push_back(int64_column_wrapper{50, 200, 70, 300}.release());
  gqe::qep::hash_join_build_task build_task{
    3, {0}, {0, 1, 2}, cudf::null_equality::UNEQUAL, /*is_hash_map_cache_enabled=*/true};
  auto build = fold(build_task, {table_state(std::move(build_columns))});

  std::vector<std::unique_ptr<cudf::column>> probe_columns;
  probe_columns.push_back(int64_column_wrapper{1, 2, 3}.release());
  probe_columns.push_back(int64_column_wrapper{10, 8, 9}.release());
  probe_columns.push_back(int64_column_wrapper{100, 80, 90}.release());
  auto probe_input = table_state(std::move(probe_columns));
  auto const int64 = cudf::data_type{cudf::type_id::INT64};
  gqe::qep::hash_join_probe_task probe_task{
    {0}, gqe::join_type_type::inner, {4, 1, 5, 2}, {int64, int64, int64, int64}};
  auto candidates = probe(probe_task, build, probe_input);

  auto residual = std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(1),
    std::make_shared<gqe::column_reference_expression>(4));
  gqe::qep::filter_join_indices_task filter_task{
    gqe::join_type_type::inner,
    std::move(residual),
    3,
    3,
    gqe::qep::mixed_join_candidate_layout{.logical_candidate_indices = {1, 4, 2, 5},
                                          .left_conditional_indices  = {0},
                                          .right_conditional_indices = {1},
                                          .output_indices            = {2, 3, 2}},
    {int64, int64, int64, int64},
    {int64, int64, int64}};
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor result_mr{cudf::get_current_device_resource_ref()};
  auto filtered = filter_task.next(candidates, ctx_ref, stream, result_mr);
  ASSERT_TRUE(filtered.has_value());
  auto const indirect = gqe::qep::indirect_table_view::try_from(*filtered);
  ASSERT_TRUE(indirect.has_value());
  ASSERT_EQ(indirect->columns.size(), 3);
  auto const& first_left =
    *std::get<gqe::qep::state_kind::indirect_column>(*indirect->columns[0]).state();
  auto const& repeated_left =
    *std::get<gqe::qep::state_kind::indirect_column>(*indirect->columns[2]).state();
  EXPECT_EQ(first_left.gather_source().get(), repeated_left.gather_source().get());
  for (auto const& output : indirect->indirect_columns) {
    EXPECT_TRUE(result_mr.get_outstanding_allocations().contains(
      const_cast<void*>(output.gather_source->position_list()->column().view().head())));
  }

  stream.ref().sync();
  auto output = gather(*filtered);
  auto table  = *gqe::qep::to_table_view(output);
  auto sorted = cudf::sort(table);
  std::vector<std::unique_ptr<cudf::column>> expected_columns;
  expected_columns.push_back(int64_column_wrapper{80, 100}.release());
  expected_columns.push_back(int64_column_wrapper{70, 50}.release());
  expected_columns.push_back(int64_column_wrapper{80, 100}.release());
  auto expected = table_state(std::move(expected_columns));
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), *gqe::qep::to_table_view(expected));

  // Equivalent source wrappers must produce the same result.
  gqe::qep::state_container split_sources;
  for (auto const& column : candidates) {
    auto const& candidate = *std::get<gqe::qep::state_kind::indirect_column>(*column).state();
    auto const& source    = candidate.gather_source();
    auto separate         = gqe::qep::indirect_gather_source_state::make(
      source->relation_source(), source->position_list(), source->bounds_policy());
    split_sources.push_back(
      gqe::qep::indirect_column_state::make(separate, candidate.source_column_index()));
  }
  auto split_filtered = filter_task.next(split_sources, ctx_ref, stream, result_mr);
  ASSERT_TRUE(split_filtered.has_value());
  stream.ref().sync();
  auto split_output = gather(*split_filtered);
  auto split_sorted = cudf::sort(*gqe::qep::to_table_view(split_output));
  CUDF_TEST_EXPECT_TABLES_EQUAL(split_sorted->view(), *gqe::qep::to_table_view(expected));

  gqe::qep::filter_join_indices_task count_filter{
    gqe::join_type_type::inner,
    std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::column_reference_expression>(4)),
    3,
    3,
    gqe::qep::mixed_join_candidate_layout{.logical_candidate_indices = {1, 4, 2, 5},
                                          .left_conditional_indices  = {0},
                                          .right_conditional_indices = {1},
                                          .output_indices            = {}},
    {int64, int64, int64, int64},
    {}};
  auto count = count_filter.next(candidates, ctx_ref, stream, result_mr);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*count), 2);

  auto split_count = count_filter.next(split_sources, ctx_ref, stream, result_mr);
  ASSERT_TRUE(split_count.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*split_count), 2);

  auto clone        = count_filter.clone();
  auto cloned_count = static_cast<gqe::qep::filter_join_indices_task const&>(*clone).next(
    split_sources, ctx_ref, stream, result_mr);
  ASSERT_TRUE(cloned_count.has_value());
  EXPECT_EQ(gqe::qep::try_row_count(*cloned_count), 2);

  auto const candidate_types = std::vector<cudf::data_type>{int64, int64, int64, int64};
  auto empty_candidates      = gqe::qep::make_empty_state_container(candidate_types);
  auto empty_filtered        = filter_task.next(empty_candidates, ctx_ref, stream, result_mr);
  ASSERT_TRUE(empty_filtered.has_value());
  EXPECT_EQ(gqe::qep::to_table_view(*empty_filtered)->num_rows(), 0);
  EXPECT_EQ(gqe::qep::to_table_view(*empty_filtered)->num_columns(), 3);
}

TEST_F(QepHashJoinTaskTest, MixedLeftCandidatesAcceptEquivalentSourceWrappers)
{
  auto const int64 = cudf::data_type{cudf::type_id::INT64};
  gqe::qep::hash_join_build_task build_task{2, {0}, {0, 1}, cudf::null_equality::UNEQUAL, false};
  auto build = fold(build_task, {two_int64_columns({1, 2}, {5, 30})});
  gqe::qep::hash_join_probe_task probe_task{
    {0}, gqe::join_type_type::left, {2, 3, 0, 1}, {int64, int64, int64, int64}};
  auto candidates = probe(probe_task, build, two_int64_columns({1, 2, 3}, {10, 20, 40}));
  gqe::qep::state_container split_sources;
  for (auto const& column : candidates) {
    auto const& candidate = *std::get<gqe::qep::state_kind::indirect_column>(*column).state();
    auto const& source    = candidate.gather_source();
    auto separate         = gqe::qep::indirect_gather_source_state::make(
      source->relation_source(), source->position_list(), source->bounds_policy());
    split_sources.push_back(
      gqe::qep::indirect_column_state::make(separate, candidate.source_column_index()));
  }
  gqe::qep::filter_join_indices_task filter{
    gqe::join_type_type::left,
    std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::column_reference_expression>(3)),
    2,
    2,
    gqe::qep::mixed_join_candidate_layout{.logical_candidate_indices = {0, 1, 2, 3},
                                          .left_conditional_indices  = {1},
                                          .right_conditional_indices = {3},
                                          .output_indices            = {0, 3}},
    {int64, int64, int64, int64},
    {int64, int64}};
  auto filtered = filter.next(split_sources, ctx_ref);
  ASSERT_TRUE(filtered.has_value());
  auto output = gather(*filtered);
  auto sorted = cudf::sort(*gqe::qep::to_table_view(output));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), int64_column_wrapper({1, 2, 3}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view().column(1),
                                      int64_column_wrapper({5, 0, 0}, {true, false, false}));
}

TEST_F(QepHashJoinTaskTest, MixedCandidateBindingRejectsInvalidLogicalMetadata)
{
  auto const make_filter = [](cudf::size_type left_width,
                              cudf::size_type right_width,
                              std::vector<cudf::size_type> logical_indices) {
    return gqe::qep::filter_join_indices_task(
      gqe::join_type_type::inner,
      std::make_unique<gqe::greater_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::column_reference_expression>(4)),
      left_width,
      right_width,
      gqe::qep::mixed_join_candidate_layout{.logical_candidate_indices = std::move(logical_indices),
                                            .left_conditional_indices  = {0},
                                            .right_conditional_indices = {1},
                                            .output_indices            = {}},
      std::vector<cudf::data_type>(4, cudf::data_type{cudf::type_id::INT64}),
      std::vector<cudf::data_type>{});
  };
  EXPECT_THROW((void)make_filter(-1, 3, {1, 4, 2, 5}), std::invalid_argument);
  EXPECT_THROW((void)make_filter(std::numeric_limits<cudf::size_type>::max(), 1, {1, 4, 2, 5}),
               std::invalid_argument);
  EXPECT_THROW((void)make_filter(3, 3, {1, 6, 2, 5}), std::invalid_argument);
  EXPECT_THROW((void)make_filter(3, 3, {4, 1, 2, 5}), std::invalid_argument);
  EXPECT_THROW((void)make_filter(3, 3, {0, 4, 2, 5}), std::invalid_argument);
}

TEST_F(QepHashJoinTaskTest, RejectsInvalidConfigurationTypesAndStateShapes)
{
  EXPECT_THROW(
    std::ignore = gqe::qep::hash_join_build_task(0, {}, {}, cudf::null_equality::UNEQUAL, false),
    std::invalid_argument);
  for (auto const type : {gqe::join_type_type::full,
                          gqe::join_type_type::left_semi,
                          gqe::join_type_type::left_anti,
                          gqe::join_type_type::single}) {
    EXPECT_THROW(std::ignore = gqe::qep::hash_join_probe_task({0}, type, {}, {}),
                 std::invalid_argument);
  }
  EXPECT_THROW(std::ignore = gqe::qep::hash_join_probe_task(
                 {0}, gqe::join_type_type::inner, {0}, {cudf::data_type{cudf::type_id::LIST}}),
               std::invalid_argument);

  gqe::qep::hash_join_build_task build_task{2, {0}, {}, cudf::null_equality::UNEQUAL, false};
  auto build = fold(build_task, {two_int64_columns({1}, {10})});
  std::vector<std::unique_ptr<cudf::column>> wrong_type;
  wrong_type.push_back(int32_column_wrapper{1}.release());
  wrong_type.push_back(int64_column_wrapper{100}.release());
  gqe::qep::hash_join_probe_task probe_task{{0}, gqe::join_type_type::inner, {}, {}};
  auto accumulator = probe_task.initialize(build, ctx_ref);
  EXPECT_THROW(
    std::ignore = probe_task.next(table_state(std::move(wrong_type)), accumulator, ctx_ref),
    std::logic_error);
  EXPECT_THROW(std::ignore = probe_task.initialize(gqe::qep::make_row_count_container(1), ctx_ref),
               std::logic_error);
}

TEST_F(QepHashJoinTaskTest, ClonePreservesTaskKinds)
{
  gqe::qep::hash_join_build_task build_task{2, {0}, {}, cudf::null_equality::EQUAL, true};
  auto cloned_build = build_task.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::hash_join_build_task*>(cloned_build.get()), nullptr);
  gqe::qep::hash_join_probe_task probe_task{{0}, gqe::join_type_type::left, {}, {}};
  auto cloned_probe = probe_task.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::hash_join_probe_task*>(cloned_probe.get()), nullptr);

  gqe::qep::unique_key_inner_join_build_task unique_build(
    2, {0}, {}, cudf::null_equality::EQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto cloned_unique_build = unique_build.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::unique_key_inner_join_build_task*>(cloned_unique_build.get()),
            nullptr);
  gqe::qep::unique_key_inner_join_probe_task unique_probe({0}, {}, {});
  auto cloned_unique_probe = unique_probe.clone();
  EXPECT_NE(dynamic_cast<gqe::qep::unique_key_inner_join_probe_task*>(cloned_unique_probe.get()),
            nullptr);
}

TEST_F(QepHashJoinTaskTest, UniqueKeyJoinPreservesDuplicateProbesAcrossBlocks)
{
  std::vector<int64_t> keys(8192);
  std::iota(keys.begin(), keys.end(), int64_t{0});
  auto probe_keys = keys;
  probe_keys.insert(probe_keys.end(), keys.begin(), keys.end());
  std::vector<int64_t> build_payload(keys.size());
  std::iota(build_payload.begin(), build_payload.end(), int64_t{10000});
  std::vector<int64_t> probe_payload(probe_keys.size());
  std::iota(probe_payload.begin(), probe_payload.end(), int64_t{20000});
  auto expected_build = build_payload;
  expected_build.insert(expected_build.end(), build_payload.begin(), build_payload.end());
  auto expected        = two_int64_columns(expected_build, probe_payload);
  auto expected_sorted = cudf::sort(*gqe::qep::to_table_view(expected));
  for (bool cache : {false, true}) {
    SCOPED_TRACE(cache);
    gqe::qep::unique_key_inner_join_build_task build_task(
      2, {0}, {1}, cudf::null_equality::UNEQUAL, cache, {cudf::data_type{cudf::type_id::INT64}});
    auto build = fold(build_task, {two_int64_columns(keys, build_payload)});
    gqe::qep::unique_key_inner_join_probe_task probe_task(
      {0}, {1, 3}, {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}});
    auto result   = probe(probe_task, build, two_int64_columns(probe_keys, probe_payload));
    auto gathered = gather(result);
    auto sorted   = cudf::sort(*gqe::qep::to_table_view(gathered));
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected_sorted->view());
  }
}

TEST_F(QepHashJoinTaskTest, UniqueKeyConstructorRejectsUnsupportedOrMissingTypes)
{
  using build_task = gqe::qep::unique_key_inner_join_build_task;
  for (auto type : {cudf::type_id::STRING, cudf::type_id::TIMESTAMP_DAYS, cudf::type_id::LIST}) {
    EXPECT_THROW(std::ignore = build_task(
                   2, {0}, {}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{type}}),
                 std::invalid_argument);
  }
  EXPECT_THROW(std::ignore = build_task(2, {0}, {}, cudf::null_equality::UNEQUAL, true, {}),
               std::invalid_argument);
}

TEST_F(QepHashJoinTaskTest, UniqueKeyBuildRejectsMismatchedInputTypes)
{
  gqe::qep::unique_key_inner_join_build_task unique_build(
    2, {0}, {}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT32}});
  auto accumulator = unique_build.initialize(ctx_ref);
  EXPECT_THROW(unique_build.next(two_int64_columns({1}, {10}), accumulator, ctx_ref),
               std::logic_error);
}

TEST_F(QepHashJoinTaskTest, ProbeTasksRejectMismatchedBuildStates)
{
  gqe::qep::hash_join_build_task general_build(2, {0}, {}, cudf::null_equality::UNEQUAL, true);
  auto build = fold(general_build, {two_int64_columns({1}, {10})});
  gqe::qep::unique_key_inner_join_probe_task unique_probe({0}, {}, {});
  EXPECT_THROW(std::ignore = unique_probe.initialize(build, ctx_ref), std::logic_error);
  gqe::qep::unique_key_inner_join_build_task matching_unique_build(
    2, {0}, {}, cudf::null_equality::UNEQUAL, true, {cudf::data_type{cudf::type_id::INT64}});
  auto unique_state = fold(matching_unique_build, {two_int64_columns({1}, {10})});
  gqe::qep::hash_join_probe_task general_probe({0}, gqe::join_type_type::inner, {}, {});
  EXPECT_THROW(std::ignore = general_probe.initialize(unique_state, ctx_ref), std::logic_error);
}

}  // namespace gqe_test
