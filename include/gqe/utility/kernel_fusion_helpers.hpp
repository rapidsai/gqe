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

#include <gqe/executor/eval.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/expression.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>

namespace gqe {
namespace utility {

/**
 * @brief Get the mask column from filter expression.
 *
 * @param parameters optimization parameters for expression evaluation
 * @param table_view table to be applied the filter expression
 * @param filter_condition filter expression
 * @param column_reference_offset column reference offset for the filter expression
 */
std::pair<cudf::column_view, std::unique_ptr<cudf::column>> get_mask_from_filter(
  gqe::optimization_parameters const& parameters,
  cudf::table_view const& table_view,
  std::unique_ptr<gqe::expression> filter_condition,
  cudf::size_type column_reference_offset);

/**
 * @brief Identity predicate.
 */
struct identity_pred {
  __device__ bool operator()(bool x) const { return x; }
};

/**
 * @brief Get the num active keys object from the active mask column by counting number of 1s.
 *
 * @param num_keys
 * @param active_mask
 */
cudf::size_type get_num_active_keys(cudf::size_type num_keys, cudf::column_view const& active_mask);

}  // namespace utility
}  // namespace gqe