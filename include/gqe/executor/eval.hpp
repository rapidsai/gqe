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

#include <gqe/expression/expression.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
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
                     cudf::size_type column_reference_offset = 0);

}  // namespace gqe
