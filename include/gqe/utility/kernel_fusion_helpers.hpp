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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/expression.hpp>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <rmm/resource_ref.hpp>

namespace gqe {
namespace utility {

/**
 * @brief Get the mask column from filter expression.
 *
 * @param parameters optimization parameters for expression evaluation
 * @param input Physical columns, logical binding, and row geometry.
 * @param filter_condition filter expression
 */
[[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>> get_mask_from_filter(
  gqe::optimization_parameters const& parameters,
  expression_input const& input,
  std::unique_ptr<gqe::expression> filter_condition);

/**
 * @brief Identity predicate.
 */
struct identity_pred {
  __device__ bool operator()(bool x) const { return x; }
};

/**
 * @brief Count active rows from a Boolean mask.
 *
 * When `active_mask` is non-empty, this function reduces it on `stream` and synchronizes that
 * stream to return the exact count to the host. An empty mask includes all `num_keys` rows and does
 * not synchronize.
 *
 * @param[in] num_keys Number of rows represented by an empty mask.
 * @param[in] active_mask Boolean mask whose true entries are active.
 * @param[in] stream CUDA stream used for the reduction and host read.
 * @param[in] mr Memory resource used for the reduction result.
 * @return Number of active rows.
 */
[[nodiscard]] cudf::size_type get_num_active_keys(cudf::size_type num_keys,
                                                  cudf::column_view const& active_mask,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

/**
 * @brief Count active rows using the default stream and current device memory resource.
 *
 * @copydetails get_num_active_keys(cudf::size_type, cudf::column_view const&,
 *                                  cuda::stream_ref, rmm::device_async_resource_ref)
 */
[[nodiscard]] cudf::size_type get_num_active_keys(cudf::size_type num_keys,
                                                  cudf::column_view const& active_mask);

}  // namespace utility
}  // namespace gqe
