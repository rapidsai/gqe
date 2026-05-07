/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
