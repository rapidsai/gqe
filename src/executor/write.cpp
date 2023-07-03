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

#include <gqe/executor/write.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <cstdint>
#include <stdexcept>

namespace gqe {

write_task_base::write_task_base(int32_t task_id, int32_t stage_id, std::shared_ptr<task> input)
  : task(task_id, stage_id, {std::move(input)}, {})
{
}

parquet_write_task::parquet_write_task(int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<task> input,
                                       std::vector<std::string> file_paths,
                                       std::vector<std::string> column_names,
                                       std::vector<cudf::data_type> data_types)
  : write_task_base(task_id, stage_id, input),
    _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types))
{
}

void parquet_write_task::execute()
{
  prepare_dependencies();
  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  // check if input schema matches output schema
  auto num_columns = _column_names.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data type list must have the same length as the number of columns");
  }
  if (static_cast<std::size_t>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  // set column names
  auto metadata = cudf::io::table_input_metadata(input_table);
  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    metadata.column_metadata[column_idx].set_name(_column_names[column_idx]);
  }

  // write to Parquet file
  auto sink    = cudf::io::sink_info(std::move(_file_paths));
  auto builder = cudf::io::chunked_parquet_writer_options::builder(sink).metadata(&metadata);

  cudf::io::parquet_chunked_writer(builder).write(input_table).close();

  remove_dependencies();
}

}  // namespace gqe
