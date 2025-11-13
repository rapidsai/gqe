/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

namespace libperfect {

template <typename indices_type>
cudf::column scatter_aggregate(cudf::column_view const& values,
                               rmm::device_uvector<indices_type> const& indices,
                               cudf::column_view const& mask,
                               const std::optional<cudf::column_view> output_map,
                               const cudf::aggregation::Kind aggregation_kind,
                               int64_t max_index,
                               const cudf::type_id output_type_id);

}  // namespace libperfect
