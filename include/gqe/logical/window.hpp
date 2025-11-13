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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace optimizer {
class optimization_rule;
}  // namespace optimizer
namespace logical {

class window_relation : public relation {
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   *
   * @brief Construct a new window relation object. The logical window relation appends a column
   * containing the window function result to the end of the input relation.
   *
   * @param input_relation Input relation to apply window function on
   * @param aggr_func Window aggregation function
   * @param arguments Column on which to execute window function
   * @param order_by Expression used to order arguments column before applying aggregation function
   * @param partition_by Expression used to group arguments column before applying affregation
   * function
   * @param order_dirs Direction of sorting applied via `order_by`
   * @param[in] lower_window_bound Number of rows by which window frame extends beyond the current
   * row index. Has type window_frame_bound::unbounded if the window extends to the boundary of the
   * partition and window_frame_bound::bounded otherwise.
   * @param[in] upper_window_bound Number of rows by which window frame extends above the current
   * row index. Has type window_frame_bound::unbounded if the window extends to the boundary of the
   * partition and window_frame_bound::bounded otherwise.
   */
  window_relation(std::shared_ptr<relation> input_relation,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  cudf::aggregation::Kind aggr_func,
                  std::vector<std::unique_ptr<expression>> arguments,
                  std::vector<std::unique_ptr<expression>> order_by,
                  std::vector<std::unique_ptr<expression>> partition_by,
                  std::vector<cudf::order> order_dirs,
                  window_frame_bound::type window_lower_bound,
                  window_frame_bound::type window_upper_bound);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::window; }

  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] cudf::aggregation::Kind aggr_func() const noexcept;
  [[nodiscard]] std::vector<expression*> arguments_unsafe() const noexcept;
  [[nodiscard]] std::vector<expression*> order_by_unsafe() const noexcept;
  [[nodiscard]] std::vector<expression*> partition_by_unsafe() const noexcept;
  [[nodiscard]] std::vector<cudf::order> order_dirs() const noexcept;
  [[nodiscard]] window_frame_bound::type window_lower_bound() const noexcept;
  [[nodiscard]] window_frame_bound::type window_upper_bound() const noexcept;

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  cudf::aggregation::Kind _aggr_func;
  std::vector<std::unique_ptr<expression>> _arguments;
  std::vector<std::unique_ptr<expression>> _order_by;
  std::vector<std::unique_ptr<expression>> _partition_by;
  std::vector<cudf::order> _order_dirs;
  window_frame_bound::type _window_lower_bound;
  window_frame_bound::type _window_upper_bound;
};

}  // namespace logical
}  // namespace gqe