/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace gqe::perfect_hash {

/**
 * @brief Aggregate `values` into the groups given by `indices`.
 *
 * Row `i` of `values` is combined into output row `indices[i]`, or `output_map[indices[i]]` when
 * `output_map` is given. An output row that receives no value holds the identity of the
 * aggregation.
 *
 * @pre `indices` has one entry per row of `values`, and every output row is less than `max_index`.
 *
 * @throw std::invalid_argument if a type or `aggregation_kind` is not supported.
 *
 * @tparam indices_type Element type of `indices`.
 *
 * @param[in] values Values to aggregate.
 * @param[in] indices Group index of each row of `values`.
 * @param[in] mask Rows of `values` to aggregate, or an empty view to aggregate every row.
 * @param[in] output_map Optional mapping from a group index to an output row.
 * @param[in] aggregation_kind Aggregation to apply.
 * @param[in] max_index Number of output rows.
 * @param[in] output_type_id Element type of the result.
 * @param[in] stream CUDA stream used for all device work.
 *
 * @return A column of `max_index` aggregated values, or an empty column if `values` is empty or
 *         `max_index` is 0.
 */
template <typename indices_type>
cudf::column scatter_aggregate(cudf::column_view const& values,
                               rmm::device_uvector<indices_type> const& indices,
                               cudf::column_view const& mask,
                               const std::optional<cudf::column_view> output_map,
                               const cudf::aggregation::Kind aggregation_kind,
                               int64_t max_index,
                               const cudf::type_id output_type_id,
                               cuda::stream_ref stream = cudf::get_default_stream());

}  // namespace gqe::perfect_hash
