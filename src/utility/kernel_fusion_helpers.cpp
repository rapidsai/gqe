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

#include <gqe/utility/kernel_fusion_helpers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>

namespace gqe {
namespace utility {

std::pair<cudf::column_view, std::unique_ptr<cudf::column>> get_mask_from_filter(
  gqe::optimization_parameters const& parameters,
  cudf::table_view const& table_view,
  std::unique_ptr<gqe::expression> filter_condition,
  cudf::size_type column_reference_offset)
{
  std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>> eval_output;
  cudf::column_view active_mask;
  std::unique_ptr<cudf::column> active_mask_column;
  if (filter_condition != nullptr) {
    std::vector<expression const*> condition_expr{filter_condition.get()};
    eval_output =
      evaluate_expressions(parameters, table_view, condition_expr, column_reference_offset);
    active_mask        = (eval_output.first)[0];
    active_mask_column = std::move((eval_output.second)[0]);
  }
  return {active_mask, std::move(active_mask_column)};
}

cudf::size_type get_num_active_keys(cudf::size_type num_keys, cudf::column_view const& active_mask)
{
  if (!active_mask.is_empty()) {
    auto total_active_keys = cudf::reduce(active_mask,
                                          *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                          cudf::data_type{cudf::type_id::INT32});
    return static_cast<cudf::numeric_scalar<cudf::size_type>*>(total_active_keys.get())->value();
  } else {
    return num_keys;
  }
}

}  // namespace utility
}  // namespace gqe
