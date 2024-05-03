/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/storage/parquet_reader.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>

namespace gqe::storage {

table_with_metadata read_parquet_cudf(std::vector<std::string> const& file_paths,
                                      std::vector<std::string> const& column_names)
{
  table_with_metadata result;

  // Read the table from Parquet files
  auto source  = cudf::io::source_info(file_paths);
  auto options = cudf::io::parquet_reader_options::builder(source);
  options.columns(column_names);
  result.table = std::move(cudf::io::read_parquet(options).tbl);

  // Get the number of rows per file
  std::vector<cudf::size_type> rows_per_file;
  rows_per_file.reserve(file_paths.size());

  for (auto const& file_path : file_paths) {
    auto file_source = cudf::io::source_info(file_path);
    auto metadata    = cudf::io::read_parquet_metadata(file_source);
    rows_per_file.push_back(metadata.num_rows());
  }
  result.rows_per_file = std::move(rows_per_file);

  return result;
}

}  // namespace gqe::storage
