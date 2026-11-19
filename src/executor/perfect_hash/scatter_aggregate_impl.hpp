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

#include <gqe/executor/perfect_hash/detail/query_common.hpp>

#include <cuda/stream>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>

namespace gqe::perfect_hash {

/**
 * @brief Aggregates values by index into an output buffer of the requested element type.
 *
 * @tparam output_type Element type of the aggregate results.
 * @param values Values to aggregate.
 * @param indices Output position for each input value.
 * @param mask Mask selecting input rows, or an empty view to select every row.
 * @param output_map Optional remapping of output positions.
 * @param aggregation_kind Aggregation to apply.
 * @param max_index Number of elements in the output buffer.
 * @param stream CUDA stream used for the operation.
 * @return Buffer containing the aggregate results.
 */
template <typename output_type>
CudaGpuBuffer scatter_aggregate_typed(cudf::column_view const& values,
                                      rmm::device_uvector<int> const& indices,
                                      cudf::column_view const& mask,
                                      std::optional<cudf::column_view> output_map,
                                      cudf::aggregation::Kind aggregation_kind,
                                      int64_t max_index,
                                      cuda::stream_ref stream);

}  // namespace gqe::perfect_hash
