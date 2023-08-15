/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gqe/executor/window.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>

namespace gqe {

window_task::window_task(query_context* query_context,
                         int32_t task_id,
                         int32_t stage_id,
                         std::shared_ptr<task> input,
                         cudf::aggregation::Kind aggr_func,
                         std::vector<std::unique_ptr<expression>> ident_cols,
                         std::vector<std::unique_ptr<expression>> arguments,
                         std::vector<std::unique_ptr<expression>> partition_by,
                         std::vector<std::unique_ptr<expression>> order_by,
                         std::vector<cudf::order> order_dirs,
                         window_frame_bound::type window_lower_bound,
                         window_frame_bound::type window_upper_bound)
  : task(query_context, task_id, stage_id, {std::move(input)}, {}),
    _aggr_func{aggr_func},
    _ident_cols{std::move(ident_cols)},
    _arguments{std::move(arguments)},
    _partition_by{std::move(partition_by)},
    _order_by{std::move(order_by)},
    _order_dirs{std::move(order_dirs)},
    _window_lower_bound{window_lower_bound},
    _window_upper_bound{window_upper_bound}
{
}

namespace {

std::unique_ptr<cudf::rolling_aggregation> get_rolling_aggregation(
  cudf::aggregation::Kind aggregation_kind)
{
  switch (aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::MEAN: return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::STD: return cudf::make_std_aggregation<cudf::rolling_aggregation>();
    case cudf::aggregation::ROW_NUMBER:
      return cudf::make_row_number_aggregation<cudf::rolling_aggregation>();
    default:
      throw std::runtime_error("Unknown rolling aggregation: " + std::to_string(aggregation_kind));
  }
}
}  // namespace

struct constant_scalar_functor {
  template <typename T>
  std::unique_ptr<cudf::scalar> operator()(int val)
  {
    if constexpr (cudf::is_integral<T>()) {
      return cudf::make_fixed_width_scalar<T>(val);
    } else {
      throw std::runtime_error("Non-integral order-by columns not supported!\n");
    }
  }
};

std::unique_ptr<cudf::table> window_partition_and_order(
  cudf::table_view input_table,
  cudf::aggregation::Kind aggr_func,
  std::vector<std::unique_ptr<expression>> ident_cols,
  std::vector<std::unique_ptr<expression>> arguments,
  std::vector<std::unique_ptr<expression>> partition_by,
  std::vector<std::unique_ptr<expression>> order_by,
  std::vector<cudf::order> order_dirs,
  window_frame_bound::type window_lower_bound,
  window_frame_bound::type window_upper_bound)
{
  if (order_by.size() > 1) {
    throw std::runtime_error("Window function cannot have more than one order_by column");
  }
  if (arguments.size() > 1) {
    throw std::runtime_error("Window function cannot have more than one argument column");
  }

  std::unique_ptr<cudf::table> sorted_table;
  std::vector<std::unique_ptr<cudf::column>> grouped_cols_cache;
  std::vector<cudf::column_view> grouped_cols;

  if (partition_by.size() > 0) {
    auto [partition_cols, partition_cols_cache] =
      evaluate_expressions(input_table, utility::to_const_raw_ptrs(partition_by));
    auto partition_by_table = cudf::table_view(partition_cols);

    // group entire table using the partition_by columns as keys
    cudf::groupby::groupby grpby(partition_by_table);
    auto groups             = grpby.get_groups(input_table);
    auto const host_offsets = groups.offsets;

    rmm::device_uvector<cudf::size_type> device_offsets{host_offsets.size(),
                                                        rmm::cuda_stream_default};
    GQE_CUDA_TRY(cudaMemcpy(device_offsets.data(),
                            host_offsets.data(),
                            host_offsets.size() * sizeof(cudf::size_type),
                            cudaMemcpyDefault));
    auto const offsets_col = cudf::column{std::move(device_offsets), rmm::device_buffer{}, 0};

    auto [order_table_cols, order_table_cols_cache] =
      evaluate_expressions(groups.values.get()->view(), utility::to_const_raw_ptrs(order_by));
    auto order_by_table = cudf::table_view(order_table_cols);

    sorted_table = cudf::segmented_sort_by_key(
      groups.values.get()->view(), order_by_table, offsets_col.view(), std::move(order_dirs));

    // TODO: partition_by and order_by columns are currently evaluated twice: once before
    // sorting/grouping and once after (for the cuDF rolling window)
    auto [grouped_partition_keys_cols, grouped_partition_keys_cols_cache] =
      evaluate_expressions(sorted_table.get()->view(), utility::to_const_raw_ptrs(partition_by));
    grouped_cols_cache = std::move(grouped_partition_keys_cols_cache);
    grouped_cols       = std::move(grouped_partition_keys_cols);
  } else {
    auto [order_table_cols, order_table_cols_cache] =
      evaluate_expressions(input_table, utility::to_const_raw_ptrs(order_by));
    auto order_by_table = cudf::table_view(order_table_cols);

    sorted_table = cudf::sort_by_key(input_table, order_by_table, std::move(order_dirs));

    // To execute order-by, we need to have a sort-aware window function. Unfortunately, cuDF only
    // provides this functionality for grouped data. To get around this, we initialize a dummy
    // column of all zeros to place all rows in the same group before calling the appropriate cuDF
    // rolling window function.
    size_t num_rows = sorted_table.get()->num_rows();
    auto dummy_col  = cudf::make_column_from_scalar(cudf::numeric_scalar<uint32_t>{0}, num_rows);
    grouped_cols.push_back(dummy_col.get()->view());
    grouped_cols_cache.push_back(std::move(dummy_col));
  }

  auto grouped_table = cudf::table_view(grouped_cols);

  auto const [sort_cols, sort_cols_cache] =
    evaluate_expressions(sorted_table.get()->view(), utility::to_const_raw_ptrs(order_by));
  auto const sort_col_view = sort_cols[0];

  std::unique_ptr<cudf::column> window_col;

  if (aggr_func == cudf::aggregation::RANK) {
    if (std::holds_alternative<window_frame_bound::bounded>(window_lower_bound)) {
      throw std::runtime_error(
        "RANK-aggregated window function does not support a custom lower window bound.");
    }

    cudf::groupby::groupby grpby_scan(grouped_table);
    cudf::groupby::scan_request request;
    request.values = sort_col_view;

    std::vector<std::unique_ptr<cudf::groupby_scan_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(cudf::rank_method::MIN));
    request.aggregations = std::move(aggregations);

    std::vector<cudf::groupby::scan_request> scan_requests;
    scan_requests.push_back(std::move(request));

    auto [result_table, agg_results] =
      grpby_scan.scan(cudf::host_span<const cudf::groupby::scan_request>{std::move(scan_requests)});

    window_col = std::move(agg_results[0].results[0]);
  } else {
    auto const [aggr_cols, aggr_cols_cache] =
      evaluate_expressions(sorted_table.get()->view(), utility::to_const_raw_ptrs(arguments));
    auto aggr_col_view = aggr_cols[0];

    auto get_window_bound = [&sort_col_view](window_frame_bound::type bound) {
      if (std::holds_alternative<window_frame_bound::unbounded>(bound)) {
        return cudf::range_window_bounds::unbounded(sort_col_view.type());
      } else {
        auto const_scalar =
          cudf::type_dispatcher(sort_col_view.type(),
                                constant_scalar_functor{},
                                std::get<window_frame_bound::bounded>(bound).get_bound());
        return cudf::range_window_bounds::get(*const_scalar.get());
      }
    };

    auto rolling_window_order = order_dirs[0];

    auto aggr  = get_rolling_aggregation(aggr_func);
    window_col = cudf::grouped_range_rolling_window(grouped_table,
                                                    sort_col_view,
                                                    rolling_window_order,
                                                    aggr_col_view,
                                                    get_window_bound(window_lower_bound),
                                                    get_window_bound(window_upper_bound),
                                                    1 /*= min_periods*/,
                                                    *aggr.get());
  }

  auto [final_col_views, final_col_views_cache] =
    evaluate_expressions(sorted_table.get()->view(), utility::to_const_raw_ptrs(ident_cols));

  std::vector<std::unique_ptr<cudf::column>> final_cols;
  for (auto final_col_view : final_col_views) {
    final_cols.push_back(std::make_unique<cudf::column>(final_col_view));
  }
  final_cols.push_back(std::move(window_col));

  return std::make_unique<cudf::table>(std::move(final_cols));
}

void window_task::execute()
{
  prepare_dependencies();
  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto const input_table = dependent_tasks[0]->result().value();

  std::unique_ptr<cudf::table> window_col_table;

  if (_order_by.size() > 0) {
    window_col_table = window_partition_and_order(std::move(input_table),
                                                  _aggr_func,
                                                  std::move(_ident_cols),
                                                  std::move(_arguments),
                                                  std::move(_partition_by),
                                                  std::move(_order_by),
                                                  std::move(_order_dirs),
                                                  _window_lower_bound,
                                                  _window_upper_bound);
  } else {
    throw std::runtime_error("Window task needs an order-by expression\n");
  }

  update_result_cache(std::move(window_col_table));
}

}  // namespace gqe
