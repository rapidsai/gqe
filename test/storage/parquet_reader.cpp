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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/query_context.hpp>
#include <gqe/storage/parquet_reader.hpp>

#include <cudf/column/column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename data_type>
std::vector<data_type> generate_random_data(cudf::size_type length,
                                            data_type min_value,
                                            data_type max_value)
{
  std::vector<data_type> values(length);
  std::mt19937 generator(std::random_device{}());

  if constexpr (std::is_integral_v<data_type>) {
    std::uniform_int_distribution<data_type> dist(min_value, max_value);
    for (auto& element : values) {
      element = dist(generator);
    }
  } else {
    std::uniform_real_distribution<data_type> dist(min_value, max_value);
    for (auto& element : values) {
      element = dist(generator);
    }
  }

  return values;
}

std::vector<uint8_t> generate_random_bitmask(cudf::size_type length, double null_rate)
{
  std::vector<uint8_t> validity(length);
  std::mt19937 generator(std::random_device{}());

  std::bernoulli_distribution dist(1 - null_rate);
  for (auto& element : validity) {
    element = dist(generator);
  }

  return validity;
}

// Generate a column with type `data_type`. Each row is filled with a random number in the interval
// [min_value, max_value]. Each row has a probability `null_rate` to be NULL.
template <typename data_type>
std::unique_ptr<cudf::column> generate_fixed_width_column(cudf::size_type num_rows,
                                                          double null_rate,
                                                          data_type min_value,
                                                          data_type max_value)
{
  auto values   = generate_random_data<data_type>(num_rows, min_value, max_value);
  auto validity = generate_random_bitmask(num_rows, null_rate);

  cudf::test::fixed_width_column_wrapper<data_type> column(
    values.begin(), values.end(), validity.begin());
  return column.release();
}

template <typename rep_type>
std::unique_ptr<cudf::column> generate_fixed_point_column(cudf::size_type num_rows,
                                                          double null_rate,
                                                          rep_type min_value,
                                                          rep_type max_value,
                                                          numeric::scale_type scale)
{
  auto values   = generate_random_data<rep_type>(num_rows, min_value, max_value);
  auto validity = generate_random_bitmask(num_rows, null_rate);

  cudf::test::fixed_point_column_wrapper<rep_type> column(
    values.begin(), values.end(), validity.begin(), scale);
  return column.release();
}

std::unique_ptr<cudf::table> write_file_and_load_back(cudf::table_view table,
                                                      bool use_dictionary_encoding)
{
  std::vector<std::string> column_names;
  column_names.reserve(table.num_columns());

  cudf::io::table_input_metadata table_metadata(table);
  for (cudf::size_type column_idx = 0; column_idx < table.num_columns(); column_idx++) {
    auto const column_name = "column_" + std::to_string(column_idx);
    column_names.push_back(column_name);
    table_metadata.column_metadata[column_idx].set_name(column_name);
  }

  auto table_filepath = temp_env->get_temp_filepath("table.parquet");
  auto table_options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(table_filepath), table);
  table_options.metadata(table_metadata);
  table_options.compression(cudf::io::compression_type::SNAPPY);
  table_options.dictionary_policy(use_dictionary_encoding ? cudf::io::dictionary_policy::ALWAYS
                                                          : cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(table_options);

  gqe::optimization_parameters opms{};
  opms.max_num_workers   = 1;
  opms.use_customized_io = true;
  gqe::query_context qctx(&opms);

  auto const bounce_buffer_size = qctx.io_bounce_buffer_mr->get_block_size();
  rmm::device_buffer bounce_buffer(
    bounce_buffer_size, rmm::cuda_stream_default, qctx.io_bounce_buffer_mr.get());

  return gqe::storage::read_parquet({table_filepath},
                                    column_names,
                                    bounce_buffer.data(),
                                    bounce_buffer_size,
                                    opms.io_auxiliary_threads);
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

  auto load_table = write_file_and_load_back(ref_table->view(), false);
  CUDF_TEST_EXPECT_TABLES_EQUAL(load_table->view(), ref_table->view());
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
