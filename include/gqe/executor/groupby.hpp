/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe {
//! `groupby` APIs
namespace groupby {

/**
 * @brief Groups values by keys and computes aggregations on those groups.
 */
class groupby {
 public:
  groupby()               = delete;
  ~groupby()              = default;
  groupby(groupby const&) = delete;
  groupby(groupby&&)      = delete;
  groupby& operator=(groupby const&) = delete;
  groupby& operator=(groupby&&) = delete;

  /**
   * @brief Construct a groupby object with the specified `keys`
   *
   * @note This object does *not* maintain the lifetime of `keys`. It is the
   * user's responsibility to ensure the `groupby` object does not outlive the
   * data viewed by the `keys` `table_view`.
   *
   * Rows in `keys` that contain NULL values are always included.
   * Assumed that `keys` donot have nested columns.
   *
   * @param keys Table whose rows act as the groupby keys
   *
   */
  explicit groupby(cudf::table_view const& keys);

  /**
   * @brief Performs grouped aggregations on the specified values.
   *
   * This is a optimized version of cudf's hash based groupby implementation
   * for low cardinality. It achieves higher performance by doing
   * pre-aggregations using per-block shared hash tables
   *
   * @throws cudf::logic_error If `requests[i].values.size() !=
   * keys.num_rows()`.
   *
   * @param requests The set of columns to aggregate and the aggregations to
   * perform
   * @param mr Device memory resource used to allocate the returned table and columns' device memory
   * @return Pair containing the table with each group's unique key and
   * a vector of aggregation_results for each request in the same order as
   * specified in `requests`.
   */
  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> aggregate(
    cudf::host_span<cudf::groupby::aggregation_request const> requests,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

  /**
   * @copydoc aggregate(cudf::host_span<cudf::groupby::aggregation_request const>,
   * rmm::mr::device_memory_resource*)
   *
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> aggregate(
    cudf::host_span<cudf::groupby::aggregation_request const> requests,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

 private:
  cudf::table_view _keys;  ///< Keys that determine grouping
};
}  // namespace groupby
}  // namespace gqe
