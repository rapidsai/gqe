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

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gqe {
namespace storage {

/**
 * @brief Hive partition pruning specification.
 *
 * Configures a reader for a Hive-partitioned dataset where the partition column's value is
 * encoded in the directory name (for example, `lineitem/l_shipmode=AIR/part-0001.parquet`)
 * and is absent from the on-disk schema.
 */
struct hive_pruner_options {
  cudf::size_type partition_column_idx{};  ///< Index of the partition column in the output schema.
};

/**
 * @brief Construct a Hive partition key column.
 *
 * Helpful when the dataset is Hive-partitioned: the partition-key value lives in the
 * directory name and is absent from the file's on-disk schema. This function synthesizes
 * the column in-memory from the per-segment keys and row counts.
 *
 * @param[in] dtype The numeric data type of the output column.
 * @param[in] keys One key value per segment.
 * @param[in] num_rows Row count per segment. Must have the same length as `keys`.
 *
 * @return An owned column whose i-th segment of `num_rows[i]` rows holds `keys[i]`.
 */
[[nodiscard]] std::unique_ptr<cudf::column> construct_partition_key_column(
  cudf::data_type dtype,
  std::vector<int64_t> keys,
  std::vector<cudf::size_type> num_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

namespace detail {

/**
 * @brief File selection produced by Hive-partition pruning.
 *
 * Hive partitioning encodes a partition-column value in the directory path, e.g.
 * `.../part_col=42/file.parquet`. The filtering step separates files whose parsed key
 * survives a haystack filter from files that do not match the Hive path pattern.
 */
struct hive_file_filter_info {
  std::vector<std::pair<std::size_t, std::shared_ptr<void>>>
    partitioned_files;  ///< (file_idx, partition_key) pairs surviving the haystack filter.
  cudf::size_type partition_column_idx{};  ///< Index of the Hive partition column in the output
                                           ///< schema. Unused when there are no partitioned files.
  std::vector<std::size_t>
    non_partitioned_files;  ///< File indices whose paths did not match the Hive partition pattern.
};

/**
 * @brief Build a `hive_file_filter_info` for the no-pruner case: every file is read normally.
 */
[[nodiscard]] hive_file_filter_info all_files_non_partitioned(std::size_t file_count);

/**
 * @brief Parse Hive keys from each file path, then intersect with the haystack.
 *
 * Files whose paths match the `<column_name>=<value>` pattern contribute their parsed key to
 * the haystack filter; files whose paths do not match pass through to `non_partitioned_files`
 * with a warning. The Hive default-partition sentinel is treated as null and omitted from the
 * partitioned-file output.
 *
 * @param[in] file_paths The candidate files.
 * @param[in] partition_column_idx Index of the partition column in the output schema.
 * @param[in] partition_column_name Name encoded in the file path (`<name>=<value>`).
 * @param[in] haystack Single integral column of acceptable values, wrapped in a one-column
 *   table view (nulls are dropped internally).
 *
 * @param[in] stream CUDA stream used for all device work.
 * @param[in] mr Memory resource used for device allocations, including temporary storage.
 *
 * @throws std::runtime_error On haystack shape errors or unparseable partition keys.
 */
[[nodiscard]] hive_file_filter_info filter_hive_files_by_haystack(
  std::vector<std::string> const& file_paths,
  cudf::size_type partition_column_idx,
  std::string const& partition_column_name,
  cudf::table_view const& haystack,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Per-iterator state for chunked Hive-partition column synthesis.
 *
 * Tracks the surviving file list, per-file row counts, and a cursor that maps streamed chunks
 * back to the Hive partition keys they span. This state is file-format agnostic; callers supply
 * the per-file row counts from their underlying file reader.
 */
struct hive_partitioning_state {
  cudf::size_type partition_column_idx{};  ///< Index of the partition column in the output schema.
  std::vector<int64_t> partition_keys;     ///< One key per surviving file.
  std::vector<cudf::size_type> rows_per_file;        ///< Row count per surviving file.
  std::size_t cursor_file_idx                  = 0;  ///< File whose data is being consumed next.
  cudf::size_type cursor_rows_consumed_in_file = 0;  ///< Rows already consumed in current file.
};

/**
 * @brief Hive partition-key segments covered by one streamed output chunk.
 */
struct hive_chunk_segments {
  std::vector<int64_t> keys;
  std::vector<cudf::size_type> rows;
};

/**
 * @brief Convert one chunk's row count into per-file `(partition_key, rows)` segments.
 *
 * Updates the cursor in `state` so repeated calls can process a stream of chunks.
 */
[[nodiscard]] hive_chunk_segments compute_hive_chunk_segments(hive_partitioning_state& state,
                                                              cudf::size_type chunk_rows);

}  // namespace detail
}  // namespace storage
}  // namespace gqe
