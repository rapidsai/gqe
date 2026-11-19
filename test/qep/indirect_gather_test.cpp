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

#include <gqe/qep/indirect_gather.hpp>

#include <gqe/qep/state.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>

using bool_column_wrapper  = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

namespace {

gqe::qep::shared_state owned_int64(std::initializer_list<int64_t> values)
{
  return gqe::qep::make_shared_state(
    gqe::qep::state_kind::cudf_column{int64_column_wrapper(values).release()},
    gqe::query_stream::default_stream());
}

gqe::qep::position_list_handle positions(std::initializer_list<int32_t> values)
{
  return gqe::qep::make_position_list(int32_column_wrapper(values).release(),
                                      gqe::query_stream::default_stream());
}

gqe::qep::indirect_gather_source gather_source(
  gqe::qep::state_container sources,
  gqe::qep::position_list_handle map,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto relation = gqe::qep::indirect_table_source_state::make(std::move(sources));
  return gqe::qep::indirect_gather_source_state::make(std::move(relation), std::move(map), policy);
}

}  // namespace

TEST(QepIndirectGather, MaterializesBatchInRequestedOrder)
{
  auto source =
    gather_source({owned_int64({10, 11, 12}), owned_int64({20, 21, 22})}, positions({2, 0}));
  std::array<cudf::size_type, 3> const indices{1, 0, 1};

  auto result = gqe::qep::materialize_indirect_columns(source, indices, source->position_list());
  ASSERT_EQ(result.num_columns(), 3);
  int64_column_wrapper expected_first{22, 20};
  int64_column_wrapper expected_second{12, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view().column(0), expected_first);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view().column(1), expected_second);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view().column(2), expected_first);
}

TEST(QepIndirectGather, MaterializesEmptyAndAllFalseMaskedColumns)
{
  auto source = gather_source({owned_int64({10, 20})}, positions({1, 0}));
  bool_column_wrapper all_false{false, false};
  std::array<cudf::size_type, 1> const indices{0};

  auto filtered_positions = gqe::qep::compact_position_list(source->position_list(), all_false);
  auto filtered = gqe::qep::materialize_indirect_columns(source, indices, filtered_positions);
  EXPECT_EQ(filtered.num_rows(), 0);
  EXPECT_EQ(filtered.view().column(0).type().id(), cudf::type_id::INT64);

  auto empty_source = gather_source({owned_int64({10, 20})}, positions({}));
  auto empty_mask   = cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8});
  auto empty_positions =
    gqe::qep::compact_position_list(empty_source->position_list(), empty_mask->view());
  auto empty = gqe::qep::materialize_indirect_columns(empty_source, indices, empty_positions);
  EXPECT_EQ(empty.num_rows(), 0);
  EXPECT_EQ(empty.view().column(0).type().id(), cudf::type_id::INT64);
}

TEST(QepIndirectGather, UsesProvidedStreamAndResourceForCompactedPositionsAndResults)
{
  auto source = gather_source({owned_int64({10, 20, 30})}, positions({2, 1, 0}));
  bool_column_wrapper mask{true, false, true};
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("test-explicit-stream");
  rmm::mr::tracking_resource_adaptor mr{cudf::get_current_device_resource_ref()};

  auto compacted =
    gqe::qep::compact_position_list(source->position_list(), mask, stream_handle, mr);
  std::array<cudf::size_type, 1> const indices{0};
  auto result =
    gqe::qep::materialize_indirect_columns(source, indices, compacted, stream_handle, mr);
  stream_handle.ref().sync();

  EXPECT_TRUE(mr.get_outstanding_allocations().contains(
    const_cast<void*>(compacted->column().view().head())));
  EXPECT_TRUE(
    mr.get_outstanding_allocations().contains(const_cast<void*>(result.view().column(0).head())));
  int32_column_wrapper expected_positions{2, 0};
  int64_column_wrapper expected_values{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(compacted->column().view(), expected_positions);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.view().column(0), expected_values);
}

TEST(QepIndirectGather, ComposedPositionsReleaseInputsBeforeTheIndependentResult)
{
  auto original  = positions({2, 0, 1});
  auto selecting = positions({1, 2});
  std::weak_ptr<gqe::qep::position_list_state const> original_lifetime{original};
  std::weak_ptr<gqe::qep::position_list_state const> selecting_lifetime{selecting};
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");

  auto composed = gqe::qep::compose_position_lists(original, selecting, stream);
  original.reset();
  selecting.reset();

  EXPECT_TRUE(original_lifetime.expired());
  EXPECT_TRUE(selecting_lifetime.expired());
  stream.ref().sync();
  int32_column_wrapper expected{0, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(composed->column().view(), expected);
}

TEST(QepIndirectGather, RejectsInvalidArguments)
{
  auto source = gather_source({owned_int64({10, 20})}, positions({1, 0}));
  std::array<cudf::size_type, 1> const valid_index{0};
  std::array<cudf::size_type, 1> const invalid_index{1};
  std::array<cudf::size_type, 0> const no_indices{};
  int32_column_wrapper non_bool_mask{1, 0};
  bool_column_wrapper valid_mask{true, true};
  bool_column_wrapper wrong_size_mask{true};

  EXPECT_THROW(std::ignore = gqe::qep::materialize_indirect_columns(
                 nullptr, valid_index, source->position_list()),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::materialize_indirect_columns(
                 source, no_indices, source->position_list()),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::materialize_indirect_columns(
                 source, invalid_index, source->position_list()),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::materialize_indirect_columns(
                 source, valid_index, gqe::qep::position_list_handle{}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = gqe::qep::compact_position_list(nullptr, valid_mask),
               std::invalid_argument);
  EXPECT_THROW(
    std::ignore = gqe::qep::compact_position_list(source->position_list(), non_bool_mask),
    std::logic_error);
  EXPECT_THROW(
    std::ignore = gqe::qep::compact_position_list(source->position_list(), wrong_size_mask),
    std::logic_error);
}
