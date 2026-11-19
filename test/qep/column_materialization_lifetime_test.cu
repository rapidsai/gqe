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

#include "lifetime_probe.hpp"

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/error.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <span>
#include <utility>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

template <typename T>
[[nodiscard]] std::unique_ptr<cudf::column> make_numeric_column(cudf::data_type type,
                                                                std::span<T const> values,
                                                                cuda::stream_ref stream)
{
  auto column = cudf::make_numeric_column(type,
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          cudf::get_current_device_resource_ref());
  GQE_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<T>(),
                               values.data(),
                               values.size_bytes(),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  return column;
}

TEST(QepColumnMaterialization, IndependentOutputDoesNotAnchorCrossStreamInputs)
{
  gqe::query_context query_context{gqe::make_optimization_parameters(true)};
  auto const producer_stream = query_context.create_stream("materialization-producer");
  auto const consumer_stream = query_context.create_stream("materialization-consumer");

  std::array<std::int64_t, 3> const source_values{10, 20, 30};
  bool is_source_destroyed   = false;
  bool is_indirect_destroyed = false;
  auto source_state          = qep::make_shared_state(
    state_kind::cudf_column{make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                std::span<std::int64_t const>{source_values},
                                                producer_stream.ref())},
    producer_stream);
  source_state = gqe::test::track_lifetime(std::move(source_state), is_source_destroyed);

  std::array<cudf::size_type, 2> const position_values{2, 0};
  auto positions =
    qep::make_position_list(make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                std::span<cudf::size_type const>{position_values},
                                                producer_stream.ref()),
                            producer_stream);
  producer_stream.ref().sync();

  qep::state_container source_columns;
  source_columns.push_back(std::move(source_state));
  auto relation = qep::indirect_table_source_state::make(std::move(source_columns));
  auto source = qep::indirect_gather_source_state::make(std::move(relation), std::move(positions));
  auto indirect = qep::indirect_column_state::make(std::move(source), 0);
  indirect      = gqe::test::track_lifetime(std::move(indirect), is_indirect_destroyed);
  qep::state_container input;
  input.push_back(std::move(indirect));

  auto logical = qep::logical_table_view::try_from(input);
  ASSERT_TRUE(logical.has_value());
  auto materialized = qep::materialize_all_columns(*logical, consumer_stream);

  input.clear();

  EXPECT_TRUE(is_indirect_destroyed);
  EXPECT_TRUE(is_source_destroyed);

  consumer_stream.ref().sync();
  producer_stream.ref().sync();

  cudf::test::fixed_width_column_wrapper<std::int64_t> expected{30, 10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(materialized.table_view().column(0), expected);
}

}  // namespace
