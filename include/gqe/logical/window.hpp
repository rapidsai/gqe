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

#include <gqe/logical/relation.hpp>

namespace gqe {

namespace logical {

class window_relation : public relation {
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

  [[nodiscard]] std::vector<cudf::data_type> data_types() const noexcept override
  {
    return _data_types;
  }

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

 private:
  std::vector<cudf::data_type> _data_types;
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