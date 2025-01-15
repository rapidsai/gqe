/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "../utility.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/parquet_reader.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

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
  gqe::query_context qctx(opms);

  auto const bounce_buffer_size = qctx.io_bounce_buffer_mr->get_block_size();
  rmm::device_buffer bounce_buffer(
    bounce_buffer_size, rmm::cuda_stream_default, qctx.io_bounce_buffer_mr.get());

  gqe::storage::table_with_metadata result_table;

  if (load_columns) {
    result_table = gqe::storage::read_parquet_custom({table_filepath},
                                                     column_names,
                                                     bounce_buffer.data(),
                                                     bounce_buffer_size,
                                                     qctx.parameters.io_auxiliary_threads,
                                                     qctx.parameters.io_block_size,
                                                     qctx.parameters.io_engine,
                                                     qctx.parameters.io_pipelining,
                                                     qctx.parameters.io_alignment,
                                                     qctx.disk_timer,
                                                     qctx.h2d_timer,
                                                     qctx.decomp_timer,
                                                     qctx.decode_timer);
  } else {
    result_table = gqe::storage::read_parquet_custom({table_filepath},
                                                     std::vector<std::string>(),
                                                     bounce_buffer.data(),
                                                     bounce_buffer_size,
                                                     qctx.parameters.io_auxiliary_threads,
                                                     qctx.parameters.io_block_size,
                                                     qctx.parameters.io_engine,
                                                     qctx.parameters.io_pipelining,
                                                     qctx.parameters.io_alignment,
                                                     qctx.disk_timer,
                                                     qctx.h2d_timer,
                                                     qctx.decomp_timer,
                                                     qctx.decode_timer);
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
