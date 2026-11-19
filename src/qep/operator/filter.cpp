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

#include <gqe/qep/operator/filter.hpp>

#include "filter_mask.hpp"

#include <gqe/executor/eval.hpp>
#include <gqe/expression/utility.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {

filter_task::filter_task(std::unique_ptr<gqe::expression> condition)
  : stateless_transform_task(task_ordering::preserve), _condition(std::move(condition))
{
  GQE_EXPECTS(
    _condition != nullptr, "qep::filter_task: condition must not be null", std::invalid_argument);
  std::vector<gqe::expression const*> const roots{_condition.get()};
  _condition_column_indices = collect_column_references(roots);
  _expression_binding       = expression_binding(_condition_column_indices);
  GQE_EXPECTS(!_condition_column_indices.empty(),
              "qep::filter_task: condition must reference at least one logical column",
              std::invalid_argument);
}

std::unique_ptr<task> filter_task::clone() const
{
  return std::make_unique<filter_task>(_condition->clone());
}

std::optional<state_container> filter_task::next(state_container_view inputs,
                                                 context_reference ctx_ref,
                                                 query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::filter_task::next"};
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "qep::filter_task::next: query context must not be null");

  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "qep::filter_task::next: input is not a column-bearing logical relation");

  column_materialization predicate_columns =
    materialize_columns(*logical, _condition_column_indices, stream_handle, mr);
  auto const input_table = predicate_columns.table_view();
  expression_input const expression_columns(input_table, _expression_binding, logical->row_count());
  auto const input_mask = logical->valid_mask();
  std::unique_ptr<cudf::column> output_mask;
  expression_evaluator evaluator(expression_columns,
                                 _condition.get(),
                                 ctx_ref._query_context->parameters.use_ast_jit,
                                 stream,
                                 mr);
  if (input_mask.has_value()) { output_mask = evaluator.try_evaluate_masked_filter(*input_mask); }
  if (!output_mask) {
    auto [predicate, predicate_owner] =
      evaluator.evaluate(ctx_ref._query_context->parameters.filter_use_like_shift_and);
    GQE_EXPECTS(predicate.type().id() == cudf::type_id::BOOL8,
                "qep::filter_task::next: condition must evaluate to BOOL8");
    GQE_EXPECTS(predicate.size() == input_table.num_rows(),
                "qep::filter_task::next: condition row count must match input table");

    if (input_mask.has_value()) {
      output_mask = detail::compose_filter_masks(*input_mask, predicate, stream_handle, mr);
    } else if (_condition->type() == expression::expression_type::column_reference) {
      // The evaluator returns a borrowed view for a column reference.
      // Copy it because the filter output must own the valid-mask column.
      output_mask = std::make_unique<cudf::column>(predicate, stream, mr);
    } else {
      output_mask = std::move(predicate_owner);
    }
  }

  GQE_LOG_TRACE(
    "Execute QEP filter task: input_rows={}, input_cols={}, input_masked={}, "
    "output_valid_mask_nulls={}",
    input_table.num_rows(),
    logical->num_columns(),
    input_mask.has_value(),
    output_mask->null_count());

  // Pass through each column's original `shared_state` so the output co-owns upstream storage and
  // preserves its owned or borrowed representation after the caller releases the input. The
  // logical view excludes the upstream valid mask because `output_mask` supersedes it.
  state_container_builder builder;
  builder.add_state(state_kind::valid_mask{std::move(output_mask)}, stream_handle);
  for (auto const& column_state : logical->column_states()) {
    builder.add_state(column_state);
  }
  auto output = builder.build();
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace qep
}  // namespace gqe
