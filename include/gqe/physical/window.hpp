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

#pragma once

#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/aggregation.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for scanning a rolling window accross a given column
 */
class window_relation : public relation {
 public:
  /**
   * @brief Construct a physical window relation. Currently only works on a single partition.
   * The output of this task is the window column prepended by the appropriately ordered and sorted
   * `ident_cols`.
   *
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input Input table containing columns necessary for the window function
   * @param[in] subquery_relations Subquery relations that are referenced within the given
   * expressions.
   * @param[in] aggr_func The mathematical function used to aggregate rows inside the current
   * window.
   * @param[in] ident_cols Columns that are processed during the window function before being
   * prepended to the output column. Used to merge the result back into the result table.
   * @param[in] arguments Columns on which the window function is to be performed.
   * @param[in] partition_by Columns which are used to group the input table before windowing.
   * @param[in] order_by Columns which are used to order the grouped input table before windowing.
   * @param[in] lower_window_bound Number of rows by which window frame extends beyond the current
   * row index. Has type window_frame_bound::unbounded if the window extends to the boundary of the
   * partition and window_frame_bound::bounded otherwise.
   * @param[in] upper_window_bound Number of rows by which window frame extends above the current
   * row index. Has type window_frame_bound::unbounded if the window extends to the boundary of the
   * partition and window_frame_bound::bounded otherwise.

   */
  window_relation(std::shared_ptr<relation> input,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  cudf::aggregation::Kind aggr_func,
                  std::vector<std::unique_ptr<expression>> ident_cols,
                  std::vector<std::unique_ptr<expression>> arguments,
                  std::vector<std::unique_ptr<expression>> partition_by,
                  std::vector<std::unique_ptr<expression>> order_by,
                  std::vector<cudf::order> order_dirs,
                  window_frame_bound::type window_lower_bound,
                  window_frame_bound::type window_upper_bound)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _aggr_func(aggr_func),
      _ident_cols(std::move(ident_cols)),
      _arguments(std::move(arguments)),
      _partition_by(std::move(partition_by)),
      _order_by(std::move(order_by)),
      _order_dirs(std::move(order_dirs)),
      _window_lower_bound(window_lower_bound),
      _window_upper_bound(window_upper_bound)
  {
  }

 private:
  cudf::aggregation::Kind _aggr_func;
  std::vector<std::unique_ptr<expression>> _ident_cols;
  std::vector<std::unique_ptr<expression>> _arguments;
  std::vector<std::unique_ptr<expression>> _partition_by;
  std::vector<std::unique_ptr<expression>> _order_by;
  std::vector<cudf::order> _order_dirs;
  window_frame_bound::type _window_lower_bound;
  window_frame_bound::type _window_upper_bound;

 public:
  cudf::aggregation::Kind aggr_func() { return _aggr_func; }
  std::vector<expression*> ident_cols_unsafe() { return utility::to_raw_ptrs(_ident_cols); }
  std::vector<expression*> arguments_unsafe() { return utility::to_raw_ptrs(_arguments); }
  std::vector<expression*> partition_by_unsafe() { return utility::to_raw_ptrs(_partition_by); }
  std::vector<expression*> order_by_unsafe() { return utility::to_raw_ptrs(_order_by); }
  std::vector<cudf::order> order_dirs() { return _order_dirs; }
  window_frame_bound::type window_lower_bound() { return _window_lower_bound; }
  window_frame_bound::type window_upper_bound() { return _window_upper_bound; }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }
};

}  // namespace physical
}  // namespace gqe
