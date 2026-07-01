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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <rmm/device_uvector.hpp>

namespace gqe::groupby::detail {

inline std::unique_ptr<cudf::table> gather(cudf::table_view const& source_table,
                                           rmm::device_uvector<cudf::size_type> const& gather_map,
                                           cudf::out_of_bounds_policy bounds_policy,
                                           cudf::negative_index_policy negative_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  cudf::column_view gather_map_view(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                    static_cast<cudf::size_type>(gather_map.size()),
                                    gather_map.data(),
                                    nullptr,
                                    0);
  return cudf::gather(source_table, gather_map_view, bounds_policy, negative_policy, stream, mr);
}

inline std::unique_ptr<cudf::table> gather(cudf::table_view const& source_table,
                                           cudf::device_span<cudf::size_type const> gather_map,
                                           cudf::out_of_bounds_policy bounds_policy,
                                           cudf::negative_index_policy negative_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  cudf::column_view gather_map_view(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                    static_cast<cudf::size_type>(gather_map.size()),
                                    gather_map.data(),
                                    nullptr,
                                    0);
  return cudf::gather(source_table, gather_map_view, bounds_policy, negative_policy, stream, mr);
}

inline std::unique_ptr<cudf::table> gather(cudf::table_view const& source_table,
                                           cudf::column_view const& gather_map,
                                           cudf::out_of_bounds_policy bounds_policy,
                                           cudf::negative_index_policy negative_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return cudf::gather(source_table, gather_map, bounds_policy, negative_policy, stream, mr);
}

}  // namespace gqe::groupby::detail
