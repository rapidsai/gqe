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

#include <gqe/qep/operator/project.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/utility.hpp>
#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/query_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe {
namespace qep {
namespace {

/** @brief Return borrowed views of the computed expressions in output order. */
[[nodiscard]] std::vector<gqe::expression const*> computed_expression_views(
  std::span<std::unique_ptr<gqe::expression> const> output_expressions)
{
  std::vector<gqe::expression const*> result;
  result.reserve(output_expressions.size());
  for (auto const& output_expression : output_expressions) {
    if (output_expression->type() != expression::expression_type::column_reference) {
      result.push_back(output_expression.get());
    }
  }
  return result;
}

}  // namespace

project_task::project_task(std::vector<std::unique_ptr<gqe::expression>> output_expressions)
  : stateless_transform_task(task_ordering::preserve),
    _output_expressions(std::move(output_expressions))
{
  GQE_EXPECTS(!_output_expressions.empty(),
              "qep::project_task: output expressions must not be empty",
              std::invalid_argument);
  GQE_EXPECTS(std::ranges::all_of(_output_expressions,
                                  [](auto const& expression) { return expression != nullptr; }),
              "qep::project_task: output expression must not be null",
              std::invalid_argument);

  auto const roots          = utility::to_const_raw_ptrs(_output_expressions);
  _all_column_indices       = collect_column_references(roots);
  auto const computed_roots = computed_expression_views(_output_expressions);
  _computed_column_indices  = collect_column_references(computed_roots);
  _expression_binding       = expression_binding(_computed_column_indices);
}

std::unique_ptr<task> project_task::clone() const
{
  return std::make_unique<project_task>(clone_expressions(_output_expressions));
}

std::optional<state_container> project_task::next(state_container_view inputs,
                                                  context_reference ctx_ref,
                                                  query_stream const& stream_handle,
                                                  rmm::device_async_resource_ref mr) const
{
  auto const stream = stream_handle.ref();
  utility::nvtx_scoped_range range{"qep::project_task::next"};
  GQE_EXPECTS(ctx_ref._query_context != nullptr,
              "qep::project_task::next: query context must not be null");
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "qep::project_task::next: input is not a column-bearing logical relation");
  validate_column_references(_all_column_indices, logical->num_columns());

  column_materialization computed_operands =
    materialize_columns(*logical, _computed_column_indices, stream_handle, mr);

  auto const input_mask           = logical->valid_mask();
  auto const evaluation_table     = computed_operands.table_view();
  auto const computed_expressions = computed_expression_views(_output_expressions);
  expression_input const expression_columns(
    evaluation_table, _expression_binding, logical->row_count());
  std::vector<std::unique_ptr<cudf::column>> computed_results;
  computed_results.reserve(computed_expressions.size());
  std::optional<null_mask_injection> injection;

  for (auto const* expression : computed_expressions) {
    std::unique_ptr<cudf::column> output;
    expression_evaluator evaluator(
      expression_columns, expression, ctx_ref._query_context->parameters.use_ast_jit, stream, mr);
    if (input_mask) { output = evaluator.try_evaluate_masked_projection(*input_mask); }
    if (!output) {
      if (!injection && input_mask && !_computed_column_indices.empty()) {
        injection = inject_valid_mask_as_nulls(*input_mask, evaluation_table, stream_handle, mr);
      }
      auto const table = injection ? std::make_optional(injection->as_table_view()) : std::nullopt;
      auto [view, owner] =
        evaluator.evaluate(ctx_ref._query_context->parameters.filter_use_like_shift_and, table);
      GQE_EXPECTS(owner != nullptr,
                  "qep::project_task::next: computed expression must own evaluation result");
      output = std::move(owner);
    }
    computed_results.push_back(std::move(output));
  }
  state_container_builder builder;
  if (auto const* mask_state = logical->valid_mask_state(); mask_state != nullptr) {
    builder.add_state(*mask_state);
  }

  std::size_t computed_index = 0;
  for (std::size_t i = 0; i < _output_expressions.size(); ++i) {
    auto const& expression = _output_expressions[i];
    if (expression->type() == expression::expression_type::column_reference) {
      auto const index = static_cast<column_reference_expression const&>(*expression).column_idx();
      builder.add_state(logical->column_state(index));
    } else {
      builder.add_state(state_kind::cudf_column{std::move(computed_results[computed_index++])},
                        stream_handle);
    }
  }
  auto output = builder.build();
  order_deallocation_after_stream_use(inputs, stream_handle);
  return output;
}

}  // namespace qep
}  // namespace gqe
