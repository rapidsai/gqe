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
#include <cuda/stream>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
namespace gqe::detail {
/**
 * @brief Evaluate a Boolean AST into a nonnullable BOOL8 keep mask.
 *
 * Selection must be a row-aligned BOOL8 mask. False and null entries reject rows.
 * The result is an owning keep mask. Only selected rows with valid, true expression results
 * survive. `expected_type` must be BOOL8, which this function checks before evaluation.
 *
 * Returns null if the parser output type differs from `expected_type` or the plan has no
 * operators. Blocks use at most 128 threads and shrink to fit
 * the device shared-memory limit. Inputs, selection, and literal owners must survive stream work.
 * Work is O(n * k) and scratch is O(k) per thread
 * for n rows and k AST intermediates.
 *
 * @throws std::logic_error If `expected_type` or selection is not BOOL8, or selection row count
 * differs from the table.
 * @throws std::runtime_error If nonempty input requires more shared memory for one thread
 * than the device permits per block.
 */
[[nodiscard]] std::unique_ptr<cudf::column> masked_filter_ast(
  cudf::table_view table,
  cudf::ast::expression const& expression,
  cudf::column_view selection,
  cudf::data_type expected_type,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
/**
 * @brief Evaluate a fixed-width AST into a column with rejected rows marked null.
 *
 * Selection must be a row-aligned BOOL8 mask. False and null entries reject rows. Selected
 * rows retain expression null semantics. The result owns its data; callers must retain the
 * upstream selection for subsequent operators.
 *
 * Returns null if the parser output type differs from `expected_type`, the output is not
 * fixed-width, or the plan has no operators. Blocks use at most 128 threads and shrink to fit
 * the device shared-memory limit. Inputs, selection, and literal owners must survive stream work.
 * Work is O(n * k) and scratch is O(k) per thread
 * for n rows and k AST intermediates.
 *
 * @throws std::logic_error If selection is not BOOL8 or its row count differs from the table.
 * @throws std::runtime_error If nonempty input requires more shared memory for one thread
 * than the device permits per block.
 */
[[nodiscard]] std::unique_ptr<cudf::column> masked_projection_ast(
  cudf::table_view table,
  cudf::ast::expression const& expression,
  cudf::column_view selection,
  cudf::data_type expected_type,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
}  // namespace gqe::detail
