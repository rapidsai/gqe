/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  cudf::aggregation::Kind aggr_func() const { return _aggr_func; }
  std::vector<expression*> ident_cols_unsafe() const { return utility::to_raw_ptrs(_ident_cols); }
  std::vector<expression*> arguments_unsafe() const { return utility::to_raw_ptrs(_arguments); }
  std::vector<expression*> partition_by_unsafe() const
  {
    return utility::to_raw_ptrs(_partition_by);
  }
  std::vector<expression*> order_by_unsafe() const { return utility::to_raw_ptrs(_order_by); }
  std::vector<cudf::order> order_dirs() const { return _order_dirs; }
  window_frame_bound::type window_lower_bound() const { return _window_lower_bound; }
  window_frame_bound::type window_upper_bound() const { return _window_upper_bound; }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;
};

}  // namespace physical
}  // namespace gqe
