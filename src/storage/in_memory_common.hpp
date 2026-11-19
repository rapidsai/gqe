/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/storage/in_memory.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace gqe::storage::detail {

/**
 * @brief Construct the in-memory compression configuration from query parameters.
 */
[[nodiscard]] compression::configuration make_compression_configuration(
  query_context const& query_ctx);

/**
 * @brief Select the physical storage layout for a column.
 */
[[nodiscard]] column_layout pick_column_layout(cudf::column_view const& column,
                                               compression_format configured_format,
                                               cuda::stream_ref stream);

/**
 * @brief Materialize and append one process-local row group.
 */
[[nodiscard]] std::optional<table_statistics> write_row_group(
  in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  cuda::stream_ref stream);

/**
 * @brief Materialize and append one shared-memory row group.
 *
 * @return Statistics for a published row group, or `std::nullopt` when this rank does not publish.
 */
[[nodiscard]] std::optional<table_statistics> write_row_group(
  shared_in_memory_table::row_group_appender& row_group_appender,
  cudf::table_view input_table,
  rmm::device_async_resource_ref memory_resource,
  memory_kind::type memory_kind,
  std::vector<cudf::size_type> const& column_indexes,
  std::vector<cudf::data_type> const& data_types,
  compression::configuration const& compression_config,
  cudf::size_type partition_size,
  cuda::stream_ref stream);

/**
 * @brief Return the logical cuDF type stored by an in-memory column.
 */
[[nodiscard]] cudf::data_type cudf_type_of(column_base const& column);

/**
 * @brief Return whether a layout stores nvCOMP-compressed strings, with 32- or 64-bit offsets.
 *
 * Writers pick the offset width per row group, so one column can use both layouts, and readers
 * must accept that mix.
 */
[[nodiscard]] bool is_compressed_string_layout(column_layout layout);

/**
 * @brief Return the aggregate selected row count after validating that it fits in cuDF.
 */
[[nodiscard]] cudf::size_type checked_row_count(pruning_results_t const& pruning_results);

/**
 * @brief Create a column view slice.
 */
[[nodiscard]] cudf::column_view slice_column(const cudf::column_view& full_column,
                                             const cudf::size_type size,
                                             const cudf::size_type offset,
                                             const cudf::size_type null_counts);

}  // namespace gqe::storage::detail
