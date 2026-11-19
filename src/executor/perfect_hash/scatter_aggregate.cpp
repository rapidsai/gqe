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

#include <gqe/executor/perfect_hash/scatter_aggregate.hpp>

#include "scatter_aggregate_impl.hpp"

#include <cudf/column/column_factories.hpp>

#include <cstdint>
#include <optional>

namespace gqe::perfect_hash {

namespace {

CudaGpuBuffer scatter_aggregate_dispatch(cudf::column_view const& values,
                                         rmm::device_uvector<int> const& indices,
                                         cudf::column_view const& mask,
                                         std::optional<cudf::column_view> output_map,
                                         cudf::aggregation::Kind aggregation_kind,
                                         int64_t max_index,
                                         cudf::type_id const output_type_id,
                                         cuda::stream_ref stream)
{
  switch (output_type_id) {
    case cudf::type_id::INT32:
      return perfect_hash::scatter_aggregate_typed<int32_t>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream);
    case cudf::type_id::INT64:
      return perfect_hash::scatter_aggregate_typed<int64_t>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream);
    case cudf::type_id::FLOAT32:
      return perfect_hash::scatter_aggregate_typed<float>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream);
    case cudf::type_id::FLOAT64:
      return perfect_hash::scatter_aggregate_typed<double>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream);
    default:
      std::stringstream what;
      what << "Error: Yet unsupported output type: " << static_cast<int32_t>(output_type_id)
           << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
  }
}

}  // namespace

template <typename indices_type>
cudf::column scatter_aggregate(cudf::column_view const& values,
                               rmm::device_uvector<indices_type> const& indices,
                               cudf::column_view const& mask,
                               const std::optional<cudf::column_view> output_map,
                               const cudf::aggregation::Kind aggregation_kind,
                               int64_t max_index,
                               const cudf::type_id output_type_id,
                               cuda::stream_ref stream)
{
  if (values.size() == 0 || max_index == 0) {
    return std::move(*cudf::make_empty_column(cudf::data_type(output_type_id)));
  }

  auto ret = scatter_aggregate_dispatch(
    values, indices, mask, output_map, aggregation_kind, max_index, output_type_id, stream);

  return cudf::column(cudf::data_type(output_type_id),
                      ret.numel(),
                      std::move(ret.get_buffer()),
                      rmm::device_buffer(0, stream),
                      0);
}

template cudf::column scatter_aggregate(cudf::column_view const& values,
                                        rmm::device_uvector<int> const& indices,
                                        cudf::column_view const& mask,
                                        const std::optional<cudf::column_view> output_map,
                                        const cudf::aggregation::Kind aggregation_kind,
                                        int64_t max_index,
                                        const cudf::type_id output_type_id,
                                        cuda::stream_ref stream);

}  // namespace gqe::perfect_hash
