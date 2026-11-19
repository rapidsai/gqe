/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe::groupby::hash {

/**
 * @brief Performs hash-based grouped aggregations.
 *
 * @param keys Table whose rows determine the groups
 * @param requests Columns to aggregate and the aggregations to perform
 * @param active_mask Boolean mask indicating which rows participate in the aggregation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns
 * @return The unique keys and aggregation results for each request, in request order
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> groupby(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::column_view const& active_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace gqe::groupby::hash
