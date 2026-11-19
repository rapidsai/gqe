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

#pragma once

#include <gqe/qep/operator/gather.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <cuda/stream>
#include <cudf/table/table.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace gqe_test {

using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using string_column_wrapper = cudf::test::strings_column_wrapper;

inline gqe::qep::state_container table_state(std::vector<std::unique_ptr<cudf::column>> columns)
{
  return gqe::qep::state_container_builder()
    .add_state(cudf::table{std::move(columns)}, gqe::query_stream::default_stream())
    .build();
}

inline gqe::qep::state_container two_int64_columns(std::initializer_list<int64_t> keys,
                                                   std::initializer_list<int64_t> values)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper(keys).release());
  columns.push_back(int64_column_wrapper(values).release());
  return table_state(std::move(columns));
}

inline gqe::qep::state_container two_int64_columns(std::vector<int64_t> const& keys,
                                                   std::vector<int64_t> const& values)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper(keys.begin(), keys.end()).release());
  columns.push_back(int64_column_wrapper(values.begin(), values.end()).release());
  return table_state(std::move(columns));
}

inline gqe::qep::state_container masked_two_int64_columns(std::unique_ptr<cudf::column> mask,
                                                          std::initializer_list<int64_t> keys,
                                                          std::initializer_list<int64_t> values)
{
  auto table = two_int64_columns(keys, values);
  gqe::qep::state_container result;
  result.push_back(gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask{std::move(mask)},
                                               gqe::query_stream::default_stream()));
  result.insert(result.end(), table.begin(), table.end());
  return result;
}

/** @brief Re-express direct source columns at the supplied logical row positions. */
inline gqe::qep::state_container indirect_columns(
  gqe::qep::state_container source_columns,
  std::initializer_list<cudf::size_type> positions,
  cudf::out_of_bounds_policy policy = cudf::out_of_bounds_policy::DONT_CHECK)
{
  auto const num_columns = static_cast<cudf::size_type>(source_columns.size());
  auto relation          = gqe::qep::indirect_table_source_state::make(std::move(source_columns));
  auto position_list =
    gqe::qep::make_position_list(int32_column_wrapper(positions.begin(), positions.end()).release(),
                                 gqe::query_stream::default_stream());
  auto gather_source = gqe::qep::indirect_gather_source_state::make(
    std::move(relation), std::move(position_list), policy);

  gqe::qep::state_container result;
  result.reserve(static_cast<std::size_t>(num_columns));
  for (cudf::size_type index = 0; index < num_columns; ++index) {
    result.push_back(gqe::qep::indirect_column_state::make(gather_source, index));
  }
  return result;
}

/** @brief Prepend an owned relation-level mask to logical columns. */
inline gqe::qep::state_container with_valid_mask(std::unique_ptr<cudf::column> mask,
                                                 gqe::qep::state_container columns)
{
  gqe::qep::state_container result;
  result.reserve(columns.size() + 1);
  result.push_back(gqe::qep::make_shared_state(gqe::qep::state_kind::valid_mask{std::move(mask)},
                                               gqe::query_stream::default_stream()));
  result.insert(result.end(), columns.begin(), columns.end());
  return result;
}

/** @brief Provides fold, probe, and gather helpers shared by QEP join task tests. */
class QepJoinTaskTestBase : public gqe::test::BaseFixture {
 protected:
  QepJoinTaskTestBase() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  /** @tparam BuildTask Concrete QEP join build fold task. */
  template <typename BuildTask>
  gqe::qep::state_container fold(
    BuildTask const& task,
    std::vector<gqe::qep::state_container> const& chunks,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
  {
    auto accumulator = task.initialize(ctx_ref, stream_handle, mr);
    for (auto const& chunk : chunks) {
      task.next(chunk, accumulator, ctx_ref, stream_handle, mr);
    }
    return task.finalize(std::move(accumulator), ctx_ref, stream_handle, mr);
  }

  /** @tparam ProbeTask Concrete QEP join probe transform task. */
  template <typename ProbeTask>
  gqe::qep::state_container probe(
    ProbeTask const& task,
    gqe::qep::state_container const& build,
    gqe::qep::state_container const& input,
    gqe::query_stream const& stream_handle = gqe::query_stream::default_stream(),
    rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref())
  {
    auto accumulator = task.initialize(build, ctx_ref, stream_handle, mr);
    auto result      = task.next(input, accumulator, ctx_ref, stream_handle, mr);
    if (!result.has_value()) {
      ADD_FAILURE() << "join probe task returned no result";
      return {};
    }
    task.finalize(std::move(accumulator), ctx_ref, stream_handle, mr);
    return std::move(*result);
  }

  gqe::qep::state_container gather(gqe::qep::state_container const& input)
  {
    auto result = gather_task.next(input, ctx_ref);
    if (!result.has_value()) {
      ADD_FAILURE() << "gather task returned no result";
      return {};
    }
    return std::move(*result);
  }

  gqe::context_reference ctx_ref;
  gqe::qep::gather_task gather_task;
};

}  // namespace gqe_test
