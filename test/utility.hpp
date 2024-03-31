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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <filesystem>
#include <random>
#include <string>

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

inline void write_table_to_files(cudf::table_view table,
                                 std::string dir,
                                 bool use_dictionary_encoding,
                                 std::string file_name_suffix = "")
{
  std::vector<std::string> column_names;
  column_names.reserve(table.num_columns());

  cudf::io::table_input_metadata table_metadata(table);
  for (cudf::size_type column_idx = 0; column_idx < table.num_columns(); column_idx++) {
    auto const column_name = "column_" + std::to_string(column_idx);
    column_names.push_back(column_name);
    table_metadata.column_metadata[column_idx].set_name(column_name);
  }
  std::filesystem::create_directory(dir);
  auto table_filepath = dir + "/table" + file_name_suffix + ".parquet";
  std::cout << "writing to " + table_filepath << std::endl;
  auto table_options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(table_filepath), table);
  table_options.metadata(table_metadata);
  table_options.compression(cudf::io::compression_type::SNAPPY);
  table_options.dictionary_policy(use_dictionary_encoding ? cudf::io::dictionary_policy::ALWAYS
                                                          : cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(table_options);
}
