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

#include <gqe/qep/shapes/deferred_column.hpp>

#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <rmm/cuda_stream.hpp>

#include <memory>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

namespace {

std::vector<cudf::data_type> int32_types(std::size_t count)
{
  return std::vector<cudf::data_type>(count, cudf::data_type{cudf::type_id::INT32});
}

/**
 * @brief Fake deferred source recording every decode request.
 */
class counting_source final : public gqe::qep::deferred_table_source_state {
 public:
  counting_source(cudf::size_type num_rows, std::vector<cudf::data_type> column_types)
    : deferred_table_source_state{num_rows, std::move(column_types)}
  {
  }

  mutable std::vector<std::vector<cudf::size_type>> requests;

 protected:
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> materialize_columns_impl(
    std::span<cudf::size_type const> source_column_indices,
    gqe::query_stream const& stream_handle,
    rmm::device_async_resource_ref mr) const override
  {
    auto const stream = stream_handle.ref();
    requests.emplace_back(source_column_indices.begin(), source_column_indices.end());
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (auto const index : source_column_indices) {
      // Allocate through the supplied stream and resource so the recorded deallocation stream is
      // the buffers' actual stream, as the contract requires.
      cudf::numeric_scalar<int32_t> const value{index, true, stream, mr};
      columns.push_back(cudf::make_column_from_scalar(value, num_rows(), stream, mr));
    }
    return columns;
  }
};

}  // namespace

TEST(DeferredColumnTest, ContainerHoldsOneSlotPerSourceColumn)
{
  auto source = std::make_shared<counting_source>(3, int32_types(3));
  auto chunk  = gqe::qep::make_deferred_column_container(source);

  ASSERT_EQ(chunk.size(), 3);
  for (std::size_t i = 0; i < chunk.size(); ++i) {
    auto const* slot = std::get_if<gqe::qep::state_kind::deferred_column>(chunk[i].get());
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->state()->source().get(), source.get());
    EXPECT_EQ(slot->state()->source_column_index(), static_cast<cudf::size_type>(i));
  }
  EXPECT_EQ(gqe::qep::get_row_count(chunk), 3);
  EXPECT_TRUE(source->requests.empty()) << "recognition must not decode";
}

TEST(DeferredColumnTest, MaterializeColumnsDecodesDistinctRequestInOrder)
{
  auto source = std::make_shared<counting_source>(2, int32_types(3));

  auto const first = source->materialize_columns(std::vector<cudf::size_type>{2, 0});
  ASSERT_EQ(first.size(), 2);
  ASSERT_EQ(source->requests.size(), 1);
  EXPECT_EQ(source->requests[0], (std::vector<cudf::size_type>{0, 2}))
    << "the decode request is the distinct set in increasing order";

  // The fake fills each column with its own index, so the results must follow the request order,
  // not the decode order.
  auto const& column_of = [](gqe::qep::shared_state const& state) -> cudf::column_view {
    return std::get<gqe::qep::state_kind::cudf_column>(*state).column->view();
  };
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_of(first[0]),
                                 cudf::test::fixed_width_column_wrapper<int32_t>({2, 2}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_of(first[1]),
                                 cudf::test::fixed_width_column_wrapper<int32_t>({0, 0}));

  // The source retains nothing: a repeated request decodes again and yields independent columns.
  auto const second = source->materialize_columns(std::vector<cudf::size_type>{0, 1});
  ASSERT_EQ(source->requests.size(), 2);
  EXPECT_EQ(source->requests[1], (std::vector<cudf::size_type>{0, 1}));
  EXPECT_NE(second[0].get(), first[1].get());
}

TEST(DeferredColumnTest, RepeatedIndicesShareOneState)
{
  auto source = std::make_shared<counting_source>(2, int32_types(2));

  auto const states = source->materialize_columns(std::vector<cudf::size_type>{1, 1});
  ASSERT_EQ(states.size(), 2);
  EXPECT_EQ(states[0].get(), states[1].get());
  ASSERT_EQ(source->requests.size(), 1);
  EXPECT_EQ(source->requests[0], (std::vector<cudf::size_type>{1}));
}

TEST(DeferredColumnTest, MaterializeAllPreservesNonDeferredSlots)
{
  auto source = std::make_shared<counting_source>(2, int32_types(2));

  cudf::test::fixed_width_column_wrapper<int32_t> direct_wrapper({7, 8});
  auto chunk = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::deferred_column{
                   std::make_shared<gqe::qep::deferred_column_state const>(source, 0)})
                 .add_state(gqe::qep::state_kind::cudf_column{direct_wrapper.release()},
                            gqe::query_stream::default_stream())
                 .add_state(gqe::qep::state_kind::deferred_column{
                   std::make_shared<gqe::qep::deferred_column_state const>(source, 1)})
                 .build();

  auto const materialized = gqe::qep::materialize_deferred_columns(chunk);

  ASSERT_EQ(materialized.size(), 3);
  ASSERT_EQ(source->requests.size(), 1);
  EXPECT_EQ(source->requests[0], (std::vector<cudf::size_type>{0, 1}))
    << "deferred columns must decode in one batch";
  EXPECT_EQ(materialized[1].get(), chunk[1].get()) << "non-deferred slots pass through";
  for (auto const i : {0, 2}) {
    EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*materialized[i]));
  }
  EXPECT_EQ(gqe::qep::get_row_count(materialized), 2);
}

TEST(DeferredColumnTest, RejectsDeferredColumnsFromDifferentSources)
{
  auto source_a = std::make_shared<counting_source>(2, int32_types(1));
  auto source_b = std::make_shared<counting_source>(2, int32_types(1));
  auto chunk    = gqe::qep::state_container_builder()
                 .add_state(gqe::qep::state_kind::deferred_column{
                   std::make_shared<gqe::qep::deferred_column_state const>(source_a, 0)})
                 .add_state(gqe::qep::state_kind::deferred_column{
                   std::make_shared<gqe::qep::deferred_column_state const>(source_b, 0)})
                 .build();

  EXPECT_THROW(std::ignore = gqe::qep::materialize_deferred_columns(chunk), std::logic_error);
}

TEST(DeferredColumnTest, MaterializeDeferredColumnsLeavesUnlistedSlotsDeferred)
{
  auto source = std::make_shared<counting_source>(2, int32_types(4));
  auto chunk  = gqe::qep::make_deferred_column_container(source);

  auto const partial =
    gqe::qep::materialize_deferred_columns(chunk, std::vector<std::size_t>{3, 1, 3});

  ASSERT_EQ(partial.size(), 4);
  ASSERT_EQ(source->requests.size(), 1);
  EXPECT_EQ(source->requests[0], (std::vector<cudf::size_type>{1, 3}));
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*partial[1]));
  EXPECT_TRUE(std::holds_alternative<gqe::qep::state_kind::cudf_column>(*partial[3]));
  EXPECT_EQ(partial[0].get(), chunk[0].get()) << "unlisted slots stay deferred and shared";
  EXPECT_EQ(partial[2].get(), chunk[2].get());

  // A later consumer materializes the remainder; the forwarded direct slots pass through untouched.
  auto const full = gqe::qep::materialize_deferred_columns(partial);
  ASSERT_EQ(source->requests.size(), 2);
  EXPECT_EQ(source->requests[1], (std::vector<cudf::size_type>{0, 2}));
  EXPECT_EQ(full[1].get(), partial[1].get());
}

TEST(DeferredColumnTest, MaterializedColumnsRecordTheMaterializationStream)
{
  auto source = std::make_shared<counting_source>(2, int32_types(2));
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const stream_handle = query_context.create_stream("deferred-materialization-test");

  auto const states =
    source->materialize_columns(std::vector<cudf::size_type>{0, 1}, stream_handle);
  ASSERT_EQ(states.size(), 2);
  for (auto const& state : states) {
    auto const streams = state.deallocation_streams();
    ASSERT_EQ(streams.size(), 1);
    EXPECT_EQ(streams[0].ref().get(), stream_handle.ref().get())
      << "owned columns must record the materialization stream for stream-ordered release";
  }
  stream_handle.ref().sync();
}

TEST(DeferredColumnTest, EnsureMaterializedPassesOtherShapesThrough)
{
  auto const count_only = gqe::qep::make_row_count_container(5);
  auto const passed     = gqe::qep::materialize_deferred_columns(count_only);
  ASSERT_EQ(passed.size(), count_only.size());
  EXPECT_EQ(passed[0].get(), count_only[0].get());
}

TEST(DeferredColumnTest, RejectsInvalidArguments)
{
  auto source = std::make_shared<counting_source>(2, int32_types(2));

  EXPECT_THROW(std::ignore = source->materialize_columns(std::vector<cudf::size_type>{2}),
               std::out_of_range);
  EXPECT_THROW(std::ignore = source->column_type(2), std::out_of_range);
  EXPECT_THROW(gqe::qep::deferred_column_state(nullptr, 0), std::invalid_argument);
  EXPECT_THROW(gqe::qep::deferred_column_state(source, 2), std::out_of_range);
  auto const chunk = gqe::qep::make_deferred_column_container(source);
  EXPECT_THROW(
    std::ignore = gqe::qep::materialize_deferred_columns(chunk, std::vector<std::size_t>{2}),
    std::out_of_range);
  EXPECT_THROW(std::ignore = gqe::qep::make_deferred_column_container(nullptr),
               std::invalid_argument);
  EXPECT_THROW(counting_source(-1, int32_types(1)), std::invalid_argument);
  EXPECT_THROW(counting_source(1, {}), std::invalid_argument);
}
