/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/context_reference.hpp>
#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/eval.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <gqe/executor/groupby.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>

#include "../libperfect/scatter_aggregate.hpp"
#include "../libperfect/unique_indices.hpp"

namespace gqe {

namespace {

// Note that some operations are not available in reduction or groupby. For example, "COUNT_VALID"
// is not available in reduction. "ANY" is not available in groupby. So we have two separate
// functions `get_reduce_aggregation` and `get_groupby_aggregation` instead of a single templated
// function.

std::unique_ptr<cudf::reduce_aggregation> get_reduce_aggregation(
  cudf::aggregation::Kind aggregation_kind)
{
  switch (aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::ANY: return cudf::make_any_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::ALL: return cudf::make_all_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::SUM_OF_SQUARES:
      return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MEAN: return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::STD: return cudf::make_std_aggregation<cudf::reduce_aggregation>();
    case cudf::aggregation::MEDIAN:
      return cudf::make_median_aggregation<cudf::reduce_aggregation>();
    default:
      throw std::runtime_error("Unknown reduce aggregation: " + std::to_string(aggregation_kind));
  }
}

std::unique_ptr<cudf::groupby_aggregation> get_groupby_aggregation(
  cudf::aggregation::Kind aggregation_kind)
{
  switch (aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::SUM_OF_SQUARES:
      return cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MEAN: return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::STD: return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MEDIAN:
      return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::runtime_error("Unknown groupby aggregation: " + std::to_string(aggregation_kind));
  }
}

}  // namespace

aggregate_task::aggregate_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> input,
  std::vector<std::unique_ptr<expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values,
  std::unique_ptr<expression> condition,
  bool perfect_hashing)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}),
    _keys(std::move(keys)),
    _values(std::move(values)),
    _condition(std::move(condition)),
    _perfect_hashing(perfect_hashing)
{
}

static bool fixed_width_columns(std::vector<cudf::column_view> const& columns)
{
  for (uint column_index = 0; column_index < columns.size(); column_index++) {
    const auto& current_column = columns[column_index];
    if (!cudf::is_fixed_width(current_column.type())) { return false; }
  }
  return true;
}

void aggregate_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range aggregate_task_range("aggregate_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto const input_table = dependent_tasks[0]->result().value();

  std::vector<expression const*> value_exprs;
  std::vector<cudf::aggregation::Kind> operations;
  value_exprs.reserve(_values.size());
  operations.reserve(_values.size());
  for (auto const& [kind, expr] : _values) {
    value_exprs.push_back(expr.get());
    operations.push_back(kind);
  }
  bool use_like_shift_and = get_query_context()->parameters.filter_use_like_shift_and;
  auto [value_columns, value_columns_cache] = evaluate_expressions(
    input_table, value_exprs, /*column_reference_offset=*/0, use_like_shift_and);

  std::pair<std::vector<cudf::column_view>, std::vector<std::unique_ptr<cudf::column>>> eval_output;

  cudf::column_view active_mask;
  if (_condition != nullptr) {
    std::vector<expression const*> condition_expr{_condition.get()};
    eval_output = evaluate_expressions(
      input_table, condition_expr, /*column_reference_offset=*/0, use_like_shift_and);
    active_mask = (eval_output.first)[0];
  }

  assert(value_columns.size() == operations.size());

  std::vector<std::unique_ptr<cudf::column>> result_columns;

  if (_keys.size() == 0) {
    if (!active_mask.is_empty()) {
      throw std::logic_error("Using mask is not supported by cudf::reduce");
    }
    std::transform(value_columns.begin(),
                   value_columns.end(),
                   operations.begin(),
                   std::back_inserter(result_columns),
                   [](cudf::column_view const& value_column, cudf::aggregation::Kind const& kind) {
                     std::unique_ptr<cudf::scalar> result;
                     if (kind == cudf::aggregation::COUNT_VALID) {
                       result = cudf::make_fixed_width_scalar<cudf::size_type>(
                         value_column.size() - value_column.null_count());
                     } else if (kind == cudf::aggregation::COUNT_ALL) {
                       result = cudf::make_fixed_width_scalar<cudf::size_type>(value_column.size());
                     } else {
                       auto reduce_op = get_reduce_aggregation(kind);
                       result         = cudf::reduce(value_column,
                                             *reduce_op,
                                             cudf::detail::target_type(value_column.type(), kind));
                     }
                     return cudf::make_column_from_scalar(*result, 1);
                   });
  } else {
    auto [key_columns, key_columns_cache] = evaluate_expressions(input_table,
                                                                 utility::to_const_raw_ptrs(_keys),
                                                                 /*column_reference_offset=*/0,
                                                                 use_like_shift_and);

    auto const can_perfect = fixed_width_columns(key_columns);
    if (perfect_hashing() && !can_perfect) {
      throw std::logic_error(
        "Perfect hashing is not supported for groupby when the key columns are not fixed width.");
    }
    if (perfect_hashing()) {
      auto [key_indices, group_indices] = libperfect::unique_indices(key_columns, active_mask);
      auto key_outputs =
        cudf::gather(cudf::table_view(key_columns),
                     cudf::column_view(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                                       key_indices.size(),
                                       key_indices.data(),
                                       nullptr,
                                       0));  // no nulls
      result_columns = key_outputs->release();

      for (uint i = 0; i < operations.size(); i++) {
        auto const& current_value_column = value_columns[i];
        cudf::type_id output_type_id =
          cudf::detail::target_type(current_value_column.type(), operations[i]).id();
        auto aggregate_results = libperfect::scatter_aggregate(current_value_column,
                                                               group_indices,
                                                               active_mask,
                                                               std::nullopt,
                                                               operations[i],
                                                               key_indices.size(),
                                                               output_type_id);
        result_columns.push_back(std::make_unique<cudf::column>(std::move(aggregate_results)));
      }
    } else {
      std::vector<cudf::groupby::aggregation_request> agg_requests;
      std::transform(
        value_columns.begin(),
        value_columns.end(),
        operations.begin(),
        std::back_inserter(agg_requests),
        [](cudf::column_view const& value_column, cudf::aggregation::Kind const& kind) {
          std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
          aggs.push_back(get_groupby_aggregation(kind));
          return cudf::groupby::aggregation_request({value_column, std::move(aggs)});
        });

      auto const& device_properties =
        get_context_reference()._task_manager_context->get_device_properties();

      // In SQL standard, two NULL values are not equal, but for the purpose of grouping, two or
      // more values with NULL should be grouped together.
      gqe::groupby::groupby groupby_obj{cudf::table_view(key_columns)};
      auto [key_outputs, agg_results] =
        groupby_obj.aggregate(agg_requests, active_mask, device_properties);
      result_columns = key_outputs->release();

      for (auto& agg_result : agg_results) {
        assert(agg_result.results.size() == 1);
        result_columns.push_back(std::move(agg_result.results[0]));
      }
    }
  }

  auto result = std::make_unique<cudf::table>(std::move(result_columns));

  GQE_LOG_TRACE("Execute aggregate task: task_id={}, stage_id={}, input_size={}, output_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows(),
                result->num_rows());
  emit_result(std::move(result));
  remove_dependencies();
}

}  // namespace gqe
