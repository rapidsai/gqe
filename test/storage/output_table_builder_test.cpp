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

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/pruning.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

/// Records the stream retained by every non-empty device buffer owned recursively by `column`.
void collect_buffer_streams(std::unique_ptr<cudf::column> column,
                            std::vector<cuda::stream_ref>& streams)
{
  auto contents = column->release();
  if (contents.data && contents.data->size() > 0) { streams.push_back(contents.data->stream()); }
  if (contents.null_mask && contents.null_mask->size() > 0) {
    streams.push_back(contents.null_mask->stream());
  }
  for (auto& child : contents.children) {
    collect_buffer_streams(std::move(child), streams);
  }
}

}  // namespace

class OutputTableBuilderTest : public gqe::test::BaseFixtureWithParam<bool> {
 public:
  OutputTableBuilderTest()
  {
    auto opms                  = gqe::make_optimization_parameters(false);
    opms.compression_format    = gqe::compression_format::none;
    opms.use_overlap_mtx       = GetParam();
    opms.read_use_zero_copy    = false;
    opms.use_partition_pruning = false;
    // Give the pool a fixed ceiling so the test does not inherit the CI-wide task manager budget.
    opms.initial_task_manager_memory = std::size_t{256} * 1024 * 1024;
    opms.max_task_manager_memory     = std::size_t{4} * 1024 * 1024 * 1024;
    initialize_contexts(opms);
  }

  /// Append a row group holding one INT32 and one nullable STRING column.
  void add_row_group(std::vector<int32_t> const& ints,
                     std::vector<std::string> const& strings,
                     std::vector<bool> const& validity)
  {
    cudf::test::fixed_width_column_wrapper<int32_t> int_column(ints.begin(), ints.end());
    cudf::test::strings_column_wrapper string_column(
      strings.begin(), strings.end(), validity.begin());

    _expected_int_columns.push_back(int_column.release());
    _expected_string_columns.push_back(string_column.release());

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    columns.push_back(std::make_unique<gqe::storage::contiguous_column>(
      cudf::column(*_expected_int_columns.back())));
    columns.push_back(std::make_unique<gqe::storage::contiguous_column>(
      cudf::column(*_expected_string_columns.back())));
    _row_groups.push_back(std::make_unique<gqe::storage::row_group>(std::move(columns)));
  }

  std::unique_ptr<gqe::storage::pruning_results_t> make_pruning_results(
    std::vector<cudf::size_type> const& column_indexes)
  {
    std::vector<gqe::storage::row_group_interface const*> row_groups;
    for (auto const& row_group : _row_groups) {
      row_groups.push_back(row_group.get());
    }
    return gqe::storage::detail::evaluate_in_memory_pruning(
      row_groups, column_indexes, std::nullopt, false, 0);
  }

  std::vector<std::unique_ptr<cudf::column>> _expected_int_columns;
  std::vector<std::unique_ptr<cudf::column>> _expected_string_columns;
  std::vector<std::unique_ptr<gqe::storage::row_group>> _row_groups;
};

// Columns returned by `output_table_builder::build` must free their device memory on the stream
// that the caller passed in. The builder may run `cudf::concatenate` on the shared copy-engine
// stream, and a column left holding that stream is freed there without being ordered after the
// consumer's reads on `stream`.
TEST_P(OutputTableBuilderTest, BuiltColumnsDeallocateOnProducerStream)
{
  // Non-ASCII strings and nulls exercise the chars, offsets, and null mask buffers.
  add_row_group({1, 2, 3, 4}, {"alpha", "b\xC3\xA9ta", "", "delta"}, {true, true, false, true});
  add_row_group({5, 6, 7}, {"\xCE\xB5psilon", "zeta", "eta"}, {true, true, true});

  std::vector<cudf::size_type> const column_indexes = {0, 1};
  std::vector<cudf::data_type> const data_types     = {cudf::data_type{cudf::type_id::INT32},
                                                       cudf::data_type{cudf::type_id::STRING}};

  gqe::context_reference ctx_ref{get_task_manager_ctx(), get_query_ctx()};
  gqe::storage::output_table_builder builder{
    ctx_ref, column_indexes, data_types, make_pruning_results(column_indexes)};

  // An explicit stream distinct from both the default stream and the shared copy-engine stream.
  rmm::cuda_stream producer;
  auto table = builder.build(producer, rmm::mr::get_current_device_resource_ref());
  producer.synchronize();

  // The data must be correct regardless of which stream built it.
  std::vector<cudf::column_view> expected_ints;
  std::vector<cudf::column_view> expected_strings;
  for (auto const& column : _expected_int_columns) {
    expected_ints.push_back(column->view());
  }
  for (auto const& column : _expected_string_columns) {
    expected_strings.push_back(column->view());
  }
  ASSERT_EQ(table->num_columns(), 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->get_column(0), *cudf::concatenate(expected_ints));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->get_column(1), *cudf::concatenate(expected_strings));

  auto columns = table->release();
  for (auto& column : columns) {
    std::vector<cuda::stream_ref> streams;
    collect_buffer_streams(std::move(column), streams);
    ASSERT_FALSE(streams.empty());
    for (auto const stream : streams) {
      EXPECT_EQ(stream.get(), producer.value())
        << "A built column retains a stream other than the producer stream";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(OverlapMutex,
                         OutputTableBuilderTest,
                         ::testing::Bool(),
                         [](::testing::TestParamInfo<bool> const& info) {
                           return info.param ? "UseOverlapMtx" : "NoOverlapMtx";
                         });
