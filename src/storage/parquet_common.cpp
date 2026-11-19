/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "parquet_common.hpp"

#include <gqe/query_context.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/parquet_reader.hpp>
#include <gqe/utility/logger.hpp>

#include <cuda/stream>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

#include <cassert>
#include <utility>

namespace gqe {
namespace storage {
namespace detail {

namespace {

/**
 * @brief Read parquet files via the customized or cuDF reader, depending on configuration.
 *
 */
table_with_metadata table_from_parquet(context_reference ctx_ref,
                                       std::vector<std::string> const& file_paths,
                                       std::vector<std::string> const& column_names,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  assert(!file_paths.empty());

  table_with_metadata result;

#ifdef ENABLE_CUSTOMIZED_PARQUET
  if (ctx_ref._query_context->parameters.use_customized_io) {
    try {
      auto const bounce_buffer_size = ctx_ref._query_context->io_bounce_buffer_mr->get_block_size();
      // `device_buffer` is used as a RAII wrapper. `bounce_buffer` lives in pinned host
      // memory, not device memory.
      rmm::device_buffer bounce_buffer(
        bounce_buffer_size, stream, *ctx_ref._query_context->io_bounce_buffer_mr);

      result = read_parquet_custom(file_paths,
                                   column_names,
                                   bounce_buffer.data(),
                                   bounce_buffer_size,
                                   ctx_ref._query_context->parameters.io_auxiliary_threads,
                                   ctx_ref._query_context->parameters.io_block_size,
                                   ctx_ref._query_context->parameters.io_engine,
                                   ctx_ref._query_context->parameters.io_pipelining,
                                   ctx_ref._query_context->parameters.io_alignment,
                                   ctx_ref._query_context->disk_timer,
                                   ctx_ref._query_context->h2d_timer,
                                   ctx_ref._query_context->decomp_timer,
                                   ctx_ref._query_context->decode_timer,
                                   stream,
                                   mr);
    } catch (unsupported_error const& error) {
      GQE_LOG_TRACE(error.what());
      GQE_LOG_TRACE("Fallback to cuDF's Parquet reader when loading: " + file_paths[0]);
      result = read_parquet_cudf(file_paths, column_names, stream, mr);
    }
  } else {
    result = read_parquet_cudf(file_paths, column_names, stream, mr);
  }
#else
  result = read_parquet_cudf(file_paths, column_names, stream, mr);
#endif

  assert(result.table->num_columns() == static_cast<cudf::size_type>(column_names.size()));
  return result;
}

}  // namespace

std::unique_ptr<cudf::table> enforce_data_types(std::unique_ptr<cudf::table> input,
                                                std::vector<cudf::data_type> const& data_type,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto input_columns     = input->release();
  auto const num_columns = input_columns.size();
  assert(num_columns == data_type.size());

  std::vector<std::unique_ptr<cudf::column>> converted_columns;
  converted_columns.reserve(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    auto const column_view   = input_columns[i]->view();
    auto const expected_type = data_type[i];
    if (column_view.type() == expected_type) {
      converted_columns.push_back(std::move(input_columns[i]));
    } else {
      converted_columns.push_back(cast(column_view, expected_type, stream, mr));
    }
  }

  return std::make_unique<cudf::table>(std::move(converted_columns));
}

std::unique_ptr<cudf::table> read_pruned_parquet(context_reference ctx_ref,
                                                 hive_file_filter_info const& info,
                                                 std::vector<std::string> const& file_paths,
                                                 std::vector<std::string> const& column_names,
                                                 std::vector<cudf::data_type> const& data_types,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  assert(!column_names.empty());
  assert(column_names.size() == data_types.size());

  auto const has_non_partitioned_files = !info.non_partitioned_files.empty();
  auto const has_partitioned_files     = !info.partitioned_files.empty();

  std::unique_ptr<cudf::table> non_partitioned_table;
  if (has_non_partitioned_files) {
    std::vector<std::string> file_paths_to_load;
    file_paths_to_load.reserve(info.non_partitioned_files.size());
    for (auto const& file_idx : info.non_partitioned_files) {
      file_paths_to_load.push_back(file_paths[file_idx]);
    }

    non_partitioned_table =
      table_from_parquet(ctx_ref, file_paths_to_load, column_names, stream, mr).table;
    non_partitioned_table =
      enforce_data_types(std::move(non_partitioned_table), data_types, stream, mr);
  }

  std::unique_ptr<cudf::table> partitioned_table;
  if (has_partitioned_files) {
    std::vector<std::string> file_paths_to_load;
    std::vector<int64_t> partition_keys;
    file_paths_to_load.reserve(info.partitioned_files.size());
    partition_keys.reserve(info.partitioned_files.size());
    for (auto const& [file_idx, partition_key] : info.partitioned_files) {
      file_paths_to_load.push_back(file_paths[file_idx]);
      partition_keys.push_back(*std::static_pointer_cast<int64_t>(partition_key));
    }

    std::vector<std::string> columns_to_load(column_names);
    auto const partition_column_idx = static_cast<std::size_t>(info.partition_column_idx);

    GQE_EXPECTS(partition_column_idx < columns_to_load.size(),
                "Partition column index is out of range",
                std::invalid_argument);
    columns_to_load.erase(columns_to_load.begin() + partition_column_idx);

    auto loaded_table =
      table_from_parquet(ctx_ref, file_paths_to_load, std::move(columns_to_load), stream, mr);

    auto loaded_columns = loaded_table.table->release();
    loaded_columns.insert(loaded_columns.begin() + partition_column_idx,
                          construct_partition_key_column(data_types[partition_column_idx],
                                                         std::move(partition_keys),
                                                         loaded_table.rows_per_file,
                                                         stream,
                                                         mr));

    partitioned_table = std::make_unique<cudf::table>(std::move(loaded_columns));
    partitioned_table = enforce_data_types(std::move(partitioned_table), data_types, stream, mr);
  }

  if (!has_non_partitioned_files && !has_partitioned_files) {
    // No surviving files: emit a schema-correct empty table.
    std::vector<std::unique_ptr<cudf::column>> empty_columns;
    empty_columns.reserve(data_types.size());
    for (auto const& column_type : data_types) {
      empty_columns.push_back(cudf::make_empty_column(column_type));
    }
    return std::make_unique<cudf::table>(std::move(empty_columns));
  }

  if (has_non_partitioned_files && !has_partitioned_files) { return non_partitioned_table; }
  if (!has_non_partitioned_files && has_partitioned_files) { return partitioned_table; }

  GQE_LOG_WARN(
    "Both the partitioned and unpartitioned files are present. This usually means the dataset "
    "is not partitioned properly.");
  std::vector<cudf::table_view> tables_to_concat{non_partitioned_table->view(),
                                                 partitioned_table->view()};
  return cudf::concatenate(tables_to_concat, stream, mr);
}

}  // namespace detail
}  // namespace storage
}  // namespace gqe
