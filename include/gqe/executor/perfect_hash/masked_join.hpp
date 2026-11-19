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
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace gqe::perfect_hash {

/**
 * @brief Inner-join two tables on equal keys using a perfect hash.
 *
 * Each selected row of `right_keys` is matched with the selected row of `left_keys` that has the
 * same keys. The order of the matched pairs is unspecified.
 *
 * @pre The keys of the selected rows of `left_keys` are unique.
 * @pre `left_mask` and `right_mask` are empty or `BOOL8` columns with one row per key row.
 *
 * @throw std::invalid_argument if a key column contains nulls or is not fixed-width.
 *
 * @param[in] left_keys Key columns of the side with unique keys.
 * @param[in] right_keys Key columns of the probe side.
 * @param[in] left_mask Rows of `left_keys` to join, or an empty view to join every row.
 * @param[in] right_mask Rows of `right_keys` to join, or an empty view to join every row.
 * @param[in] stream CUDA stream used for all device work.
 *
 * @return The row indices of the matched pairs in `left_keys` and in `right_keys`.
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys,
             const cudf::table_view& right_keys,
             const cudf::column_view& left_mask  = cudf::column_view(),
             const cudf::column_view& right_mask = cudf::column_view(),
             cuda::stream_ref stream             = cudf::get_default_stream());

/**
 * @brief Inner-join two tables on equal keys using a perfect hash.
 *
 * Same as the overload without `output_mr`, except that the result is allocated from `output_mr`.
 *
 * @param[in] left_keys Key columns of the side with unique keys.
 * @param[in] right_keys Key columns of the probe side.
 * @param[in] left_mask Rows of `left_keys` to join, or an empty view to join every row.
 * @param[in] right_mask Rows of `right_keys` to join, or an empty view to join every row.
 * @param[in] stream CUDA stream used for all device work.
 * @param[in] output_mr Device memory resource used to allocate the result.
 *
 * @return The row indices of the matched pairs in `left_keys` and in `right_keys`.
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys,
             const cudf::table_view& right_keys,
             const cudf::column_view& left_mask,
             const cudf::column_view& right_mask,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref output_mr);

}  // namespace gqe::perfect_hash
