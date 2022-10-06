/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/read.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>

namespace gqe {

read_task::read_task(int32_t task_id,
                     int32_t stage_id,
                     std::string file_location,
                     file_format_type file_format,
                     std::vector<std::string> column_names,
                     std::vector<cudf::data_type> data_types,
                     std::unique_ptr<gqe::expression> predicate)
  : task(task_id, stage_id, {}),
    _file_location(std::move(file_location)),
    _file_format(file_format),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types)),
    _predicate(std::move(predicate))
{
}

void read_task::execute()
{
  if (_file_format != file_format_type::parquet)
    throw std::logic_error("Read task can only load Parquet files");

  auto const num_columns = _column_names.size();

  auto source  = cudf::io::source_info(_file_location);
  auto options = cudf::io::parquet_reader_options::builder(source);
  options.columns(_column_names);

  // FIXME: Support predicate pushdown

  auto table_with_metadata = cudf::io::read_parquet(options);
  auto read_columns        = table_with_metadata.tbl->release();
  assert(read_columns.size() == num_columns);

  // Convert the read columns into the specified types if necessary

  std::vector<std::unique_ptr<cudf::column>> converted_columns;

  if (_data_types.empty()) {
    converted_columns = std::move(read_columns);
  } else {
    if (_data_types.size() != num_columns)
      throw std::length_error("data_types must have the same length as the number of columns");

    converted_columns.reserve(num_columns);
    for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++) {
      auto const column_view   = read_columns[column_idx]->view();
      auto const expected_type = _data_types[column_idx];

      if (column_view.type() == expected_type) {
        converted_columns.push_back(std::move(read_columns[column_idx]));
      } else {
        converted_columns.push_back(cudf::cast(column_view, expected_type));
      }
    }
  }

  assert(converted_columns.size() == num_columns);
  update_result_cache(std::make_unique<cudf::table>(std::move(converted_columns)));
  remove_dependencies();
}

}  // namespace gqe
