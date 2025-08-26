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

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <gqe/utility/helpers.hpp>

#ifdef GQE_ENABLE_QUERY_COMPILER
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Format.h>
#endif

#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gqe_test {

std::string get_tpch_data_path()
{
  constexpr auto env_name = "TPCH_DATA_DIR";

  char const* const env_path = std::getenv(env_name);

  if (env_path) {
    return env_path;
  } else {
    throw std::invalid_argument(std::string() + "expected the environment variable " + env_name +
                                " to be set.");
  }
}

/**
 * @brief Generate test vectors based on row count
 */
inline std::pair<std::vector<int32_t>, std::vector<double>> generateTestVectors(size_t row_num)
{
  std::vector<int32_t> id_data;
  std::vector<double> value_data;

  id_data.reserve(row_num);
  value_data.reserve(row_num);

  for (size_t i = 0; i < row_num; ++i) {
    id_data.push_back(static_cast<int32_t>(i + 1));
    value_data.push_back((i + 1) + 0.0001 * (i + 1));
  }

  return std::make_pair(std::move(id_data), std::move(value_data));
}

#ifdef GQE_ENABLE_QUERY_COMPILER
/**
 * @brief Read cuDF table from parquet files
 */
std::unique_ptr<cudf::table> readTableFromParquet(llvm::StringRef dataPath,
                                                  std::vector<std::string>& columns)
{
  auto filePaths = gqe::utility::get_parquet_files(dataPath.str());

  auto readSource = cudf::io::source_info(std::move(filePaths));
  cudf::io::parquet_reader_options_builder builder(readSource);
  builder.columns(columns);
  auto table = cudf::io::read_parquet(builder);

  return std::move(table.tbl);
}
#endif

/**
 * @brief Convert the read columns into the specified types if necessary
 *
 * Note: Copied from the private namespace in src/storage/parquet.cpp.
 */
std::unique_ptr<cudf::table> enforceDataTypes(std::unique_ptr<cudf::table> input,
                                              std::vector<cudf::data_type> const& data_type)
{
  auto input_columns     = input->release();
  auto const num_columns = input_columns.size();

  std::vector<std::unique_ptr<cudf::column>> converted_columns;
  converted_columns.reserve(num_columns);
  for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++) {
    auto const column_view   = input_columns[column_idx]->view();
    auto const expected_type = data_type[column_idx];

    if (column_view.type() == expected_type) {
      converted_columns.push_back(std::move(input_columns[column_idx]));
    } else {
      converted_columns.push_back(cudf::cast(column_view, expected_type));
    }
  }
  assert(converted_columns.size() == num_columns);

  return std::make_unique<cudf::table>(std::move(converted_columns));
}

}  // namespace gqe_test

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

inline std::vector<uint8_t> generate_random_bitmask(cudf::size_type length, double null_rate)
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

// Column names used in Parquet files generated by test cases
inline std::vector<std::string> get_column_names(cudf::size_type num_columns)
{
  std::vector<std::string> column_names;
  column_names.reserve(num_columns);

  for (cudf::size_type column_idx = 0; column_idx < num_columns; column_idx++) {
    auto const column_name = "column_" + std::to_string(column_idx);
    column_names.push_back(column_name);
  }

  return column_names;
}

inline void write_table_to_file(cudf::table_view table,
                                std::vector<std::string> const& column_names,
                                std::string const& file_path,
                                bool use_dictionary_encoding = false)
{
  assert(table.num_columns() == static_cast<cudf::size_type>(column_names.size()));

  cudf::io::table_input_metadata table_metadata(table);
  for (cudf::size_type column_idx = 0; column_idx < table.num_columns(); column_idx++) {
    table_metadata.column_metadata[column_idx].set_name(column_names[column_idx]);
  }

  auto table_options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(file_path), table);
  table_options.metadata(table_metadata);
  table_options.compression(cudf::io::compression_type::SNAPPY);
  table_options.dictionary_policy(use_dictionary_encoding ? cudf::io::dictionary_policy::ALWAYS
                                                          : cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(table_options);
}
