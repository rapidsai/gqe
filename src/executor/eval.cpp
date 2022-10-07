/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/eval.hpp>
#include <gqe/expression/column_reference.hpp>

#include <stdexcept>

namespace gqe {

std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset)
{
  // FIXME: Extend this function to support scalar results.
  // The current implementation always evaluates to a cudf::column_view. This is okay for
  // expressions like `col_ref(1) + 10`. However, other expressions like `3 + 5` should be evaluated
  // to a cudf::scalar instead.

  std::vector<cudf::column_view> evaluated_results;
  std::vector<std::unique_ptr<cudf::column>> column_cache;

  for (auto expr : exprs) {
    // FIXME: Right now, only column reference expressions are implemented
    if (expr->type() == expression::expression_type::column_reference) {
      auto const column_idx =
        dynamic_cast<gqe::column_reference_expression const*>(expr)->column_idx();

      if (column_idx < column_reference_offset)
        throw std::out_of_range("Invalid column index and offset combination in expression: " +
                                expr->to_string());

      evaluated_results.push_back(table.column(column_idx - column_reference_offset));
    } else {
      throw std::logic_error("Cannot evaluate expression: " + expr->to_string());
    }
  }

  return std::make_pair(std::move(evaluated_results), std::move(column_cache));
}

}  // namespace gqe
