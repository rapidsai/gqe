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

#include "join_task_test_util.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/conditional_inner_join.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>

#include <cudf/sorting.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

using namespace gqe_test;
namespace qep = gqe::qep;

/** @brief Checks conditional pair multiplicity, orientation, and deferred ownership. */
class QepConditionalInnerJoinTaskTest : public QepJoinTaskTestBase {
 protected:
  static auto predicate()
  {
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::column_reference_expression>(3));
  }
  static auto make_probe(bool is_build_left, bool count = false)
  {
    return qep::conditional_inner_join_probe_task{
      2,
      2,
      predicate(),
      is_build_left,
      count ? std::vector<cudf::size_type>{} : std::vector<cudf::size_type>{2, 0, 2},
      std::vector<cudf::data_type>(count ? 0 : 3, cudf::data_type{cudf::type_id::INT64})};
  }
};

TEST_F(QepConditionalInnerJoinTaskTest, DuplicateMatchesPreserveLogicalProjectionForBothBuildSides)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    auto left  = two_int64_columns({100, 200, 300}, {1, 5, 20});
    auto right = two_int64_columns({10, 10, 30}, {10, 10, 2});
    qep::conditional_inner_join_build_task task{2, {1}, {0}};
    auto build    = fold(task, {is_build_left ? left : right});
    auto output   = probe(make_probe(is_build_left), build, is_build_left ? right : left);
    auto gathered = gather(output);
    auto sorted   = cudf::sort(*qep::to_table_view(gathered));
    int64_column_wrapper expected_right{10, 10, 10, 10, 30};
    int64_column_wrapper expected_left{100, 100, 200, 200, 100};
    cudf::table_view expected{{expected_right, expected_left, expected_right}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
    EXPECT_EQ(qep::try_row_count(
                probe(make_probe(is_build_left, true), build, is_build_left ? right : left)),
              5);
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, NullableResidualsAndMasksExcludeUnknownPairs)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{1, 2, 3}.release());
    columns.push_back(int64_column_wrapper{{1, 2, 3}, {true, false, true}}.release());
    auto left = with_valid_mask(bool_column_wrapper{true, true, false}.release(),
                                table_state(std::move(columns)));
    columns.push_back(int64_column_wrapper{10, 20, 30}.release());
    columns.push_back(int64_column_wrapper{{10, 20, 30}, {true, false, true}}.release());
    auto right = with_valid_mask(bool_column_wrapper{true, true, false}.release(),
                                 table_state(std::move(columns)));
    qep::conditional_inner_join_build_task task{2, {1}, {}};
    auto build = fold(task, {is_build_left ? left : right});
    EXPECT_EQ(qep::try_row_count(
                probe(make_probe(is_build_left, true), build, is_build_left ? right : left)),
              1);
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, EmptyBuildAndProbeProduceTypedEmptyAndZeroCount)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    qep::conditional_inner_join_build_task task{2, {1}, {0}};
    for (auto chunks : {std::vector<qep::state_container>{},
                        std::vector<qep::state_container>{two_int64_columns({}, {})}}) {
      auto build  = fold(task, chunks);
      auto input  = two_int64_columns({1}, {10});
      auto output = gather(probe(make_probe(is_build_left), build, input));
      int64_column_wrapper empty{};
      cudf::table_view expected{{empty, empty, empty}};
      CUDF_TEST_EXPECT_TABLES_EQUAL(*qep::to_table_view(output), expected);
      EXPECT_EQ(qep::try_row_count(probe(make_probe(is_build_left, true), build, input)), 0);
    }
    auto build = fold(task, {two_int64_columns({1}, {10})});
    EXPECT_EQ(
      qep::try_row_count(probe(make_probe(is_build_left, true), build, two_int64_columns({}, {}))),
      0);
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, PredicatesMustReferenceBothInputs)
{
  EXPECT_THROW((qep::conditional_inner_join_build_task{2, {}, {0}}), std::invalid_argument);
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    for (int side : {-1, 0, 1}) {
      std::shared_ptr<gqe::expression> operand =
        side < 0
          ? std::shared_ptr<gqe::expression>{std::make_shared<gqe::literal_expression<int64_t>>(1)}
          : std::shared_ptr<gqe::expression>{
              std::make_shared<gqe::column_reference_expression>(side * 2)};
      auto residual = std::make_unique<gqe::less_expression>(
        operand, std::make_shared<gqe::literal_expression<int64_t>>(2));
      EXPECT_THROW(
        (qep::conditional_inner_join_probe_task{2, 2, std::move(residual), is_build_left, {}, {}}),
        std::invalid_argument);
    }
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, MaskedIndirectUtf8PayloadsSurviveInputsAndShareSelection)
{
  qep::conditional_inner_join_build_task task{2, {1}, {0}};
  auto clone = task.clone();
  auto build = fold(static_cast<qep::conditional_inner_join_build_task const&>(*clone),
                    {two_int64_columns({10}, {10}), two_int64_columns({20}, {20})});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(string_column_wrapper{"café", "東京", "skip"}.release());
  columns.push_back(int64_column_wrapper{1, 15, 99}.release());
  auto input  = indirect_columns(table_state(std::move(columns)), {1, 2, 0, 0});
  auto source = std::get<qep::state_kind::indirect_column>(*input[0])
                  .state()
                  ->gather_source()
                  ->relation_source();
  input =
    with_valid_mask(bool_column_wrapper{true, false, true, false}.release(), std::move(input));
  qep::conditional_inner_join_probe_task task_probe{2,
                                                    2,
                                                    predicate(),
                                                    false,
                                                    {0, 2, 0},
                                                    {cudf::data_type{cudf::type_id::STRING},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::STRING}}};
  cudf::get_default_stream().sync();
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor resource{cudf::get_current_device_resource_ref()};
  auto output = probe(task_probe, build, input, stream, resource);
  input.clear();
  build.clear();
  stream.ref().sync();
  auto selected = std::get<qep::state_kind::indirect_column>(*output[0]).state()->gather_source();
  EXPECT_EQ(selected->relation_source(), source);
  EXPECT_EQ(selected,
            std::get<qep::state_kind::indirect_column>(*output[2]).state()->gather_source());
  EXPECT_TRUE(resource.get_outstanding_allocations().contains(
    const_cast<void*>(selected->position_list()->column().view().head())));
  auto gathered = gather(output);
  auto sorted   = cudf::sort(*qep::to_table_view(gathered));
  string_column_wrapper expected_left{"café", "café", "東京"};
  int64_column_wrapper expected_right{10, 20, 20};
  cudf::table_view expected{{expected_left, expected_right, expected_left}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
}

TEST_F(QepConditionalInnerJoinTaskTest, InvalidMetadataIsRejected)
{
  EXPECT_THROW((qep::conditional_inner_join_build_task{0, {}, {}}), std::invalid_argument);
  EXPECT_THROW((qep::conditional_inner_join_build_task{2, {2}, {}}), std::invalid_argument);
  EXPECT_THROW((qep::conditional_inner_join_build_task{2, {0}, {2}}), std::invalid_argument);
  EXPECT_THROW((qep::conditional_inner_join_probe_task{2, 2, nullptr, false, {}, {}}),
               std::invalid_argument);
  EXPECT_THROW((qep::conditional_inner_join_probe_task{
                 2, 2, predicate(), false, {4}, {cudf::data_type{cudf::type_id::INT64}}}),
               std::invalid_argument);
}

TEST_F(QepConditionalInnerJoinTaskTest, IndirectBuildChunksKeepPayloadAndPredicateAlignment)
{
  qep::conditional_inner_join_build_task task{2, {1}, {0}};
  // Payload differs from the predicate operand. The masked-out row (110, 60) would match 25.
  auto first  = indirect_columns(two_int64_columns({110, 120, 130}, {60, 20, 30}), {2, 0, 1});
  first       = with_valid_mask(bool_column_wrapper{true, false, true}.release(), std::move(first));
  auto second = indirect_columns(two_int64_columns({140, 150}, {40, 50}), {1, 0});
  auto build  = fold(task, {first, second});
  auto output = probe(make_probe(false), build, two_int64_columns({100}, {25}));
  first.clear();
  second.clear();
  build.clear();
  auto gathered = gather(output);
  auto sorted   = cudf::sort(*qep::to_table_view(gathered));
  int64_column_wrapper expected_right{130, 140, 150};
  int64_column_wrapper expected_left{100, 100, 100};
  cudf::table_view expected{{expected_right, expected_left, expected_right}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
}

TEST_F(QepConditionalInnerJoinTaskTest, PredicateDoesNotMaterializeUnusedNestedPayload)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::test::lists_column_wrapper<int64_t>{{1, 2}, {3}, {4, 5}}.release());
    columns.push_back(int64_column_wrapper{10, 10, 10}.release());
    auto left = indirect_columns(table_state(std::move(columns)), {2, 0, 1});
    left      = with_valid_mask(bool_column_wrapper{true, true, false}.release(), std::move(left));
    columns.push_back(int64_column_wrapper{20, 5, 30}.release());
    auto right = table_state(std::move(columns));
    qep::conditional_inner_join_build_task task{is_build_left ? 2 : 1, {is_build_left ? 1 : 0}, {}};
    auto build = fold(task, {is_build_left ? left : right});
    qep::conditional_inner_join_probe_task task_probe{
      2,
      1,
      std::make_unique<gqe::greater_expression>(
        std::make_shared<gqe::column_reference_expression>(2),
        std::make_shared<gqe::column_reference_expression>(1)),
      is_build_left,
      {},
      {}};
    EXPECT_EQ(qep::try_row_count(probe(task_probe, build, is_build_left ? right : left)), 4);
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, SharedBuildAndProjectedCloneSurviveTwoProbeStreams)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    gqe::query_context first_stream_context{gqe::make_optimization_parameters(true)};
    auto const first_stream = first_stream_context.create_stream("test/first_stream");
    gqe::query_context second_stream_context{gqe::make_optimization_parameters(true)};
    auto const second_stream = second_stream_context.create_stream("test/second_stream");
    rmm::mr::tracking_resource_adaptor resource{cudf::get_current_device_resource_ref()};
    auto left  = two_int64_columns({100, 200}, {1, 5});
    auto right = two_int64_columns({10, 20}, {10, 20});
    cudf::get_default_stream().sync();
    qep::conditional_inner_join_build_task task{2, {1}, {0}};
    auto build = fold(task, {is_build_left ? left : right}, first_stream, resource);
    // The executor establishes build readiness before independent consumer streams run.
    first_stream.ref().sync();
    auto clone               = make_probe(is_build_left).clone();
    auto const& cloned_probe = static_cast<qep::conditional_inner_join_probe_task const&>(*clone);
    auto first  = probe(cloned_probe, build, is_build_left ? right : left, first_stream, resource);
    auto second = probe(cloned_probe, build, is_build_left ? right : left, second_stream, resource);
    left.clear();
    right.clear();
    build.clear();
    clone.reset();
    first_stream.ref().sync();
    second_stream.ref().sync();
    auto first_values  = gather(first);
    auto second_values = gather(second);
    auto first_sorted  = cudf::sort(*qep::to_table_view(first_values));
    auto second_sorted = cudf::sort(*qep::to_table_view(second_values));
    int64_column_wrapper expected_right{10, 10, 20, 20};
    int64_column_wrapper expected_left{100, 200, 100, 200};
    cudf::table_view expected{{expected_right, expected_left, expected_right}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(first_sorted->view(), expected);
    CUDF_TEST_EXPECT_TABLES_EQUAL(second_sorted->view(), expected);
  }
}

TEST_F(QepConditionalInnerJoinTaskTest, UnequalInputWidthsPreserveLogicalCoordinates)
{
  for (bool is_build_left : {false, true}) {
    SCOPED_TRACE(is_build_left ? "Build left" : "Build right");
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{1, 3}.release());
    auto left  = table_state(std::move(columns));
    auto right = two_int64_columns({20, 40}, {2, 4});
    qep::conditional_inner_join_build_task task{
      is_build_left ? 1 : 2, {is_build_left ? 0 : 1}, {0}};
    auto build = fold(task, {is_build_left ? left : right});
    qep::conditional_inner_join_probe_task task_probe{
      1,
      2,
      std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                             std::make_shared<gqe::column_reference_expression>(2)),
      is_build_left,
      {1, 0, 1},
      std::vector<cudf::data_type>(3, cudf::data_type{cudf::type_id::INT64})};
    auto clone    = task_probe.clone();
    auto output   = probe(static_cast<qep::conditional_inner_join_probe_task const&>(*clone),
                        build,
                        is_build_left ? right : left);
    auto gathered = gather(output);
    auto sorted   = cudf::sort(*qep::to_table_view(gathered));
    int64_column_wrapper expected_left{1, 1, 3};
    int64_column_wrapper expected_right{20, 40, 40};
    cudf::table_view expected{{expected_right, expected_left, expected_right}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
  }
}
