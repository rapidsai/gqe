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

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/storage/hive_partitioning.hpp>

#include <cuda/stream>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string>
#include <vector>

namespace gqe {
namespace storage {
namespace detail {

/**
 * @brief Read parquet files into a single owned table, with Hive-partition column synthesis
 *        and type enforcement.
 *
 * Mirrors the read flow of `parquet_read_task::read()`: reads non-partitioned and
 * partitioned file groups separately, synthesizes the partition column for the partitioned
 * group, and concatenates if both are present. Returns a schema-correct empty table when no
 * files survive.
 *
 * @param[in] ctx_ref Context used by the underlying parquet reader.
 * @param[in] info File selection.
 * @param[in] file_paths The full file list, indexed by `info`.
 * @param[in] column_names The output column names. Must be non-empty.
 * @param[in] data_types Expected output column types.
 *
 * @return The owned result table.
 */
[[nodiscard]] std::unique_ptr<cudf::table> read_pruned_parquet(
  context_reference ctx_ref,
  hive_file_filter_info const& info,
  std::vector<std::string> const& file_paths,
  std::vector<std::string> const& column_names,
  std::vector<cudf::data_type> const& data_types,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Cast each column of `input` to its expected type if necessary.
 *
 * Columns whose type already matches `data_types[i]` are moved through unchanged; the rest are
 * cast via `gqe::storage::cast`.
 *
 * @param[in] input The table to convert. Consumed.
 * @param[in] data_types Expected type per column. Must have the same length as `input`.
 *
 * @return The type-enforced table.
 */
[[nodiscard]] std::unique_ptr<cudf::table> enforce_data_types(
  std::unique_ptr<cudf::table> input,
  std::vector<cudf::data_type> const& data_types,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace detail
}  // namespace storage
}  // namespace gqe
