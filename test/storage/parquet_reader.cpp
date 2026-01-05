/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/storage/parquet_reader.hpp>

#include "../utility.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

gqe::storage::table_with_metadata write_file_and_load_back(cudf::table_view table,
                                                           bool use_dictionary_encoding,
                                                           bool load_columns = true)
{
  auto column_names   = get_column_names(table.num_columns());
  auto table_filepath = temp_env->get_temp_filepath("table.parquet");

  write_table_to_file(table, column_names, table_filepath, use_dictionary_encoding);

  gqe::optimization_parameters opms{};
  opms.max_num_workers   = 1;
  opms.use_customized_io = true;
  gqe::query_context query_ctx(opms);

  auto const bounce_buffer_size = query_ctx.io_bounce_buffer_mr->get_block_size();
  rmm::device_buffer bounce_buffer(
    bounce_buffer_size, cudf::get_default_stream(), query_ctx.io_bounce_buffer_mr.get());

  gqe::storage::table_with_metadata result_table;

  if (load_columns) {
    result_table = gqe::storage::read_parquet_custom({table_filepath},
                                                     column_names,
                                                     bounce_buffer.data(),
                                                     bounce_buffer_size,
                                                     query_ctx.parameters.io_auxiliary_threads,
                                                     query_ctx.parameters.io_block_size,
                                                     query_ctx.parameters.io_engine,
                                                     query_ctx.parameters.io_pipelining,
                                                     query_ctx.parameters.io_alignment,
                                                     query_ctx.disk_timer,
                                                     query_ctx.h2d_timer,
                                                     query_ctx.decomp_timer,
                                                     query_ctx.decode_timer);
  } else {
    result_table = gqe::storage::read_parquet_custom({table_filepath},
                                                     std::vector<std::string>(),
                                                     bounce_buffer.data(),
                                                     bounce_buffer_size,
                                                     query_ctx.parameters.io_auxiliary_threads,
                                                     query_ctx.parameters.io_block_size,
                                                     query_ctx.parameters.io_engine,
                                                     query_ctx.parameters.io_pipelining,
                                                     query_ctx.parameters.io_alignment,
                                                     query_ctx.disk_timer,
                                                     query_ctx.h2d_timer,
                                                     query_ctx.decomp_timer,
                                                     query_ctx.decode_timer);
  }

  return result_table;
}

TEST(ParquetReader, TestRandomColumns)
{
  constexpr cudf::size_type num_rows = 3000000;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.1, -30, 30));
  columns.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.01, -30, 30));
  columns.push_back(generate_fixed_width_column<int32_t>(num_rows, 0.5, -30, 30));
  columns.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.9, -30, 30));
  columns.push_back(generate_fixed_width_column<double>(num_rows, 0.1, -30.0, 30.0));
  columns.push_back(generate_fixed_width_column<float>(num_rows, 0.1, -30.0, 30.0));
  // Currently, we cannot test fixed point column because cuDF's Parquet writer uses the deprecated
  // converted type for decimals, which the customized Parquet reader does not support.
  // columns.push_back(
  //  generate_fixed_point_column<int32_t>(num_rows, 0.1, 0, 30, numeric::scale_type{-2}));
  auto ref_table = std::make_unique<cudf::table>(std::move(columns));

  auto loaded_table = write_file_and_load_back(ref_table->view(), false);
  CUDF_TEST_EXPECT_TABLES_EQUAL(loaded_table.table->view(), ref_table->view());
  ASSERT_EQ(loaded_table.rows_per_file.size(), 1);
  ASSERT_EQ(loaded_table.rows_per_file[0], num_rows);
}

TEST(ParquetReader, TestDictionaryEncoding)
{
  constexpr cudf::size_type num_rows = 30000;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.1, -30, 30));

  auto ref_table = std::make_unique<cudf::table>(std::move(columns));

  // Currently, the customized Parquet reader does not support dictionary encoding. This test checks
  // the correct exception is thrown so that the read task can fallback to cuDF's Parquet reader.
  ASSERT_THROW(write_file_and_load_back(ref_table->view(), true), gqe::storage::unsupported_error);
}

TEST(ParquetReader, TestEmptyColumnMetadata)
{
  constexpr cudf::size_type num_rows = 30000;

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(generate_fixed_width_column<int64_t>(num_rows, 0.1, -30, 30));
  auto ref_table = std::make_unique<cudf::table>(std::move(columns));

  auto loaded_table = write_file_and_load_back(ref_table->view(), false, false);
  ASSERT_EQ(loaded_table.table->num_columns(), 0);
  ASSERT_EQ(loaded_table.rows_per_file.size(), 1);
  ASSERT_EQ(loaded_table.rows_per_file[0], num_rows);
}
