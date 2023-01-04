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

#include <gqe/executor/aggregate.hpp>
#include <gqe/executor/eval.hpp>
#include <gqe/utility.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>

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
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> input,
  std::vector<std::unique_ptr<expression>> keys,
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values)
  : task(task_id, stage_id, {std::move(input)}, {}),
    _keys(std::move(keys)),
    _values(std::move(values))
{
}

void aggregate_task::execute()
{
  prepare_dependencies();
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
  auto [value_columns, value_columns_cache] = evaluate_expressions(input_table, value_exprs);
  assert(value_columns.size() == operations.size());

  std::vector<std::unique_ptr<cudf::column>> result_columns;

  if (_keys.size() == 0) {
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
                       result = cudf::reduce(value_column,
                                             get_reduce_aggregation(kind),
                                             cudf::detail::target_type(value_column.type(), kind));
                     }
                     return cudf::make_column_from_scalar(*result, 1);
                   });
  } else {
    std::vector<cudf::groupby::aggregation_request> agg_requests;
    std::transform(value_columns.begin(),
                   value_columns.end(),
                   operations.begin(),
                   std::back_inserter(agg_requests),
                   [](cudf::column_view const& value_column, cudf::aggregation::Kind const& kind) {
                     std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
                     aggs.push_back(get_groupby_aggregation(kind));
                     return cudf::groupby::aggregation_request({value_column, std::move(aggs)});
                   });

    auto [key_columns, key_columns_cache] =
      evaluate_expressions(input_table, utility::to_const_raw_ptrs(_keys));

    // In SQL standard, two NULL values are not equal, but for the purpose of grouping, two or more
    // values with NULL should be grouped together.
    cudf::groupby::groupby groupby_obj(cudf::table_view(key_columns), cudf::null_policy::INCLUDE);

    auto [key_outputs, agg_results] = groupby_obj.aggregate(agg_requests);

    result_columns = key_outputs->release();

    for (auto& agg_result : agg_results) {
      assert(agg_result.results.size() == 1);
      result_columns.push_back(std::move(agg_result.results[0]));
    }
  }

  update_result_cache(std::make_unique<cudf::table>(std::move(result_columns)));
  remove_dependencies();
}

}  // namespace gqe
