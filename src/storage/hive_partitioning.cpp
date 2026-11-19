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

#include <gqe/storage/hive_partitioning.hpp>

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace gqe {
namespace storage {
namespace detail {

namespace {

/**
 * @brief Filter Hive needles against the haystack column.
 *
 * For each `(file_idx, key)` needle, retains the file if `key` appears in the haystack.
 * Dispatched on the haystack's integral element type.
 */
struct hive_file_filter_functor {
  template <typename T>
  std::vector<std::pair<std::size_t, std::shared_ptr<void>>> operator()(
    std::vector<std::pair<std::size_t, int64_t>> const& needles,
    cudf::column_view const& acceptable_value_col,
    cuda::stream_ref stream)
  {
    std::vector<std::pair<std::size_t, std::shared_ptr<void>>> filtered_needles;

    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      filtered_needles.reserve(needles.size());

      rmm::device_uvector<T> haystack(
        acceptable_value_col.size(), stream, cudf::get_pinned_memory_resource());
      GQE_CUDA_TRY(cudaMemcpyAsync(haystack.data(),
                                   acceptable_value_col.data<T>(),
                                   acceptable_value_col.size() * sizeof(T),
                                   cudaMemcpyDefault,
                                   stream.get()));
      // The host hash set below consumes the copied keys.
      stream.sync();

      // Oversize the hash set to reduce construction time.
      std::unordered_set<int64_t> haystack_set(
        haystack.data(), haystack.data() + haystack.size(), 2 * haystack.size());

      for (auto const& needle : needles) {
        if (haystack_set.find(needle.second) != haystack_set.end()) {
          filtered_needles.emplace_back(needle.first, std::make_shared<int64_t>(needle.second));
        }
      }
    } else {
      GQE_LOG_WARN("Hive partition pruning does not support non-integral haystack of type {}",
                   cudf::type_to_name(acceptable_value_col.type()));
    }

    return filtered_needles;
  }
};

}  // namespace

hive_file_filter_info all_files_non_partitioned(std::size_t file_count)
{
  hive_file_filter_info info;
  info.non_partitioned_files.resize(file_count);
  std::iota(info.non_partitioned_files.begin(), info.non_partitioned_files.end(), std::size_t{0});
  return info;
}

hive_file_filter_info filter_hive_files_by_haystack(std::vector<std::string> const& file_paths,
                                                    cudf::size_type partition_column_idx,
                                                    std::string const& partition_column_name,
                                                    cudf::table_view const& haystack,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (haystack.num_columns() != 1) {
    throw std::runtime_error("Incorrect number of columns in the partial filter haystack");
  }
  if (!cudf::is_integral(haystack.column(0).type())) {
    throw std::runtime_error("Hive partition pruning does not support non-integral haystack");
  }

  hive_file_filter_info info;
  info.partition_column_idx = partition_column_idx;

  // Parse Hive keys from each file path; file paths that don't match fall through to
  // `non_partitioned_files`.
  std::vector<std::pair<std::size_t, int64_t>> needles;
  needles.reserve(file_paths.size());

  std::regex const valid_filename("(^|/)" + partition_column_name +
                                  "=([+-]?[0-9]+|__HIVE_DEFAULT_PARTITION__)(/|$)");

  for (std::size_t file_idx = 0; file_idx < file_paths.size(); ++file_idx) {
    std::smatch matches;
    std::regex_search(file_paths[file_idx], matches, valid_filename);
    if (matches.size() == 4) {
      auto const partition_value = matches.str(2);
      if (partition_value == "__HIVE_DEFAULT_PARTITION__") { continue; }
      needles.emplace_back(file_idx, std::stoll(partition_value));
    } else {
      GQE_LOG_WARN("Cannot parse Hive partition key from file path");
      info.non_partitioned_files.push_back(file_idx);
    }
  }

  auto acceptable_value_table     = cudf::drop_nulls(haystack, {0}, 1, stream, mr);
  auto const acceptable_value_col = acceptable_value_table->view().column(0);

  info.partitioned_files = cudf::type_dispatcher(
    acceptable_value_col.type(), hive_file_filter_functor{}, needles, acceptable_value_col, stream);

  return info;
}

hive_chunk_segments compute_hive_chunk_segments(hive_partitioning_state& state,
                                                cudf::size_type chunk_rows)
{
  hive_chunk_segments out;
  cudf::size_type remaining = chunk_rows;
  while (remaining > 0) {
    GQE_EXPECTS(state.cursor_file_idx < state.rows_per_file.size(),
                "Hive partitioning state cannot account for all rows in the current chunk");
    auto const rows_left =
      state.rows_per_file[state.cursor_file_idx] - state.cursor_rows_consumed_in_file;
    auto const take = std::min(remaining, rows_left);
    out.keys.push_back(state.partition_keys[state.cursor_file_idx]);
    out.rows.push_back(take);
    state.cursor_rows_consumed_in_file += take;
    remaining -= take;
    if (state.cursor_rows_consumed_in_file == state.rows_per_file[state.cursor_file_idx]) {
      ++state.cursor_file_idx;
      state.cursor_rows_consumed_in_file = 0;
    }
  }
  return out;
}

}  // namespace detail
}  // namespace storage
}  // namespace gqe
