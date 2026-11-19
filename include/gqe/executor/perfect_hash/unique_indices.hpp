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
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace gqe::perfect_hash {

/**
 * @brief Group rows by their keys using a perfect hash.
 *
 * Rows whose values are equal in every column of `keys` form one group. The order of the groups
 * and the representative row chosen for each group are unspecified.
 *
 * @pre The key columns are fixed-width, have no nulls, and have the same number of rows.
 *
 * @throw std::invalid_argument if `keys` is empty or `mask` has an unsupported type.
 *
 * @param[in] keys Key columns.
 * @param[in] mask Rows to group, or an empty view to group every row. The group index of a row
 *                 outside the mask is unspecified.
 * @param[in] stream CUDA stream used for all device work.
 *
 * @return The row index of one representative per group, and the group index of each row.
 */
std::tuple<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
unique_indices(std::vector<cudf::column_view> const& keys,
               cudf::column_view const& mask,
               cuda::stream_ref stream = cudf::get_default_stream());

}  // namespace gqe::perfect_hash
