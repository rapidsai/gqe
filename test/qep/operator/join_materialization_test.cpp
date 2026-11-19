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

#include <gqe/qep/operator/join_materialization.hpp>

#include "../lifetime_probe.hpp"

#include <gqe/qep/indirect_gather.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <limits>
#include <stdexcept>

namespace qep      = gqe::qep;
using int32_column = cudf::test::fixed_width_column_wrapper<int32_t>;

/** @brief Builds join outputs from owned columns and position lists, and runs the gather. */
class QepJoinMaterializationTest : public gqe::test::BaseFixture {
 protected:
  static qep::shared_state owned(std::unique_ptr<cudf::column> column)
  {
    return qep::make_shared_state(qep::state_kind::cudf_column{std::move(column)},
                                  gqe::query_stream::default_stream());
  }
  static qep::position_list_handle positions(std::initializer_list<int32_t> rows)
  {
    return qep::make_position_list(int32_column(rows).release(),
                                   gqe::query_stream::default_stream());
  }
  /** @brief Make a gather source over `columns`; bounds are unchecked unless requested. */
  static qep::indirect_gather_source source(
    qep::state_container columns,
    qep::position_list_handle selection,
    cudf::out_of_bounds_policy bounds = cudf::out_of_bounds_policy::DONT_CHECK)
  {
    return qep::indirect_gather_source_state::make(
      qep::indirect_table_source_state::make(std::move(columns)), std::move(selection), bounds);
  }
  /** @brief Run the gather on the default stream and current device resource. */
  static qep::state_container run(qep::state_container_view input, double threshold)
  {
    return qep::materialize_selective_join_output(input,
                                                  threshold,
                                                  gqe::query_stream::default_stream(),
                                                  cudf::get_current_device_resource_ref());
  }
};

TEST_F(QepJoinMaterializationTest, StrictThresholdAndDisabledPolicyPreserveOwnership)
{
  auto backing =
    source({owned(int32_column{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.release())}, positions({7}));
  qep::state_container input{qep::indirect_column_state::make(backing, 0)};
  for (double threshold : {0.0, 0.05, 0.1}) {
    auto result = run(input, threshold);
    EXPECT_EQ(result[0], input[0]);
  }
  auto result = run(input, 0.11);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int32_column{7});
}

TEST_F(QepJoinMaterializationTest, SelectsSourcesIndependentlyAndPreservesProjectionOrder)
{
  auto large = source({owned(int32_column{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.release()),
                       owned(int32_column{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}.release())},
                      positions({7}));
  auto small = source({owned(int32_column{100, 101}.release())}, positions({1}));
  qep::state_container input{qep::indirect_column_state::make(large, 1),
                             qep::indirect_column_state::make(small, 0),
                             owned(int32_column{42}.release()),
                             qep::indirect_column_state::make(large, 0)};
  auto result = run(input, 0.2);
  EXPECT_EQ(result[1], input[1]);
  EXPECT_EQ(result[2], input[2]);
  qep::state_container selected{result[0], result[3]};
  auto table = qep::to_table_view(selected);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int32_column{17});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), int32_column{7});
}

TEST_F(QepJoinMaterializationTest, ComposedPositionsUseOriginalSourceRows)
{
  auto original  = positions({7, 2});
  auto selection = positions({1});
  auto composed  = qep::compose_position_lists(original,
                                              selection,
                                              gqe::query_stream::default_stream(),
                                              cudf::get_current_device_resource_ref());
  auto backing   = source({owned(int32_column{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.release())}, composed);
  qep::state_container input{qep::indirect_column_state::make(backing, 0)};
  // 1/10 qualifies, although the most recent selection ratio is 1/2.
  auto result = run(input, 0.2);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int32_column{2});
}

TEST_F(QepJoinMaterializationTest, DuplicateMatchesCountTowardOutputSize)
{
  auto backing = source({owned(int32_column{0, 1, 2, 3}.release())}, positions({1, 1, 1, 1, 1}));
  qep::state_container input{qep::indirect_column_state::make(backing, 0)};
  EXPECT_EQ(run(input, 1.0)[0], input[0]);
}

TEST_F(QepJoinMaterializationTest, EmptySelectionsReleaseSourcesIncludingEmptyBacking)
{
  for (auto values : {std::initializer_list<int32_t>{}, {1, 2, 3}}) {
    auto backing = source({owned(int32_column(values).release())}, positions({}));
    qep::state_container input{qep::indirect_column_state::make(backing, 0)};
    auto result = run(input, 0.1);
    auto table  = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    EXPECT_EQ(table->num_rows(), 0);
  }
}

TEST_F(QepJoinMaterializationTest, PreservesStringNullsAndOuterJoinNullification)
{
  auto strings =
    cudf::test::strings_column_wrapper{{"zéro", "二", "three", "four"}, {true, false, true, true}};
  auto backing =
    source({owned(strings.release())}, positions({1, 8, 0}), cudf::out_of_bounds_policy::NULLIFY);
  qep::state_container input{qep::indirect_column_state::make(backing, 0)};
  auto result = run(input, 1.0);
  auto table  = qep::to_table_view(result);
  ASSERT_TRUE(table.has_value());
  auto expected = cudf::test::strings_column_wrapper{{"", "", "zéro"}, {false, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected);
}

TEST_F(QepJoinMaterializationTest, ReleasesSourceWhileConsumersShareCompactOutput)
{
  bool destroyed = false;
  auto state     = gqe::test::track_lifetime(
    owned(int32_column{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.release()), destroyed);
  auto backing = source({std::move(state)}, positions({7}));
  qep::state_container input{qep::indirect_column_state::make(std::move(backing), 0)};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  cudf::get_default_stream().sync();
  rmm::mr::tracking_resource_adaptor tracking{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};
  auto result   = qep::materialize_selective_join_output(input, 0.2, stream_handle, tracking);
  auto consumer = result;
  input.clear();
  EXPECT_TRUE(destroyed);
  EXPECT_GT(tracking.get_allocated_bytes(), 0);
  result.clear();
  stream_handle.ref().sync();
  auto table = qep::to_table_view(consumer);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int32_column{7});
  consumer.clear();
  EXPECT_EQ(tracking.get_allocated_bytes(), 0);
}

TEST_F(QepJoinMaterializationTest, PassesDirectAndCountOnlyStatesAndRejectsInvalidInputs)
{
  qep::state_container input{owned(int32_column{1, 2}.release())};
  EXPECT_EQ(run(input, 0.1)[0], input[0]);
  auto count = run(qep::make_row_count_container(42), 0.1);
  EXPECT_EQ(qep::try_row_count(count), 42);
  EXPECT_THROW(std::ignore = run({}, 0.1), std::logic_error);
  for (double threshold : {-0.1,
                           1.1,
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(std::ignore = run(input, threshold), std::invalid_argument);
    EXPECT_THROW(qep::join_materialization_task task(threshold), std::invalid_argument);
  }
  auto mask = cudf::test::fixed_width_column_wrapper<bool>{true, false};
  input.insert(input.begin(),
               qep::make_shared_state(qep::state_kind::valid_mask{mask.release()},
                                      gqe::query_stream::default_stream()));
  EXPECT_THROW(std::ignore = run(input, 0.1), std::logic_error);
}

TEST_F(QepJoinMaterializationTest, ClonePreservesPolicy)
{
  auto backing = source({owned(int32_column{0, 1, 2, 3}.release())}, positions({3}));
  qep::state_container input{qep::indirect_column_state::make(backing, 0)};
  auto task       = qep::join_materialization_task{0.5}.clone();
  auto* transform = dynamic_cast<qep::stateless_transform_task*>(task.get());
  ASSERT_NE(transform, nullptr);
  EXPECT_EQ(dynamic_cast<qep::join_materialization_task&>(*task).threshold(), 0.5);
  auto result = transform->next(input,
                                {get_task_manager_ctx(), get_query_ctx()},
                                gqe::query_stream::default_stream(),
                                cudf::get_current_device_resource_ref());
  ASSERT_TRUE(result.has_value());
  auto table = qep::to_table_view(*result);
  ASSERT_TRUE(table.has_value());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), int32_column{3});
}
