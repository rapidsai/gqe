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

#pragma once

#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {

/**
 * @brief Evaluate a batch of expressions on a table.
 *
 * @param[in] table Table on which to evaluate the expressions.
 * @param[in] exprs Expressions to be evaluated.
 * @param[in] column_reference_offset Offset for column reference expressions. For example,
 * if this argument is 2, col_ref(3) refers to column 1 (3 - 2).
 *
 * @return A pair of [evaluated_results, column_cache] where `evaluated_results` are the results of
 * evaluating `exprs` on `table`, and `column_cache` must be kept alive for `evaluated_results` to
 * be valid.
 */
std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>>
evaluate_expressions(cudf::table_view const& table,
                     std::vector<expression const*> const& exprs,
                     cudf::size_type column_reference_offset = 0)
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
