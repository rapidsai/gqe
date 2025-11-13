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

class sort_relation : public relation {
  friend class gqe::optimizer::optimization_rule;

 public:
  /**
   * @brief Construct a new sort relation object
   *
   * @param input_relation Input table to be sorted
   * @param subquery_relations Subquery relations that are referenced within the `expressions`
   * @param column_orders Desired order for each column in `expressions`. The size of this argument
   * must be the same as the size of `expressions`
   * @param null_precedences Whether a null element is smaller or larger than other elements.
   * The size of this argument must be the same as the size of `expressions`
   * @param expressions Expressions evaluated on `input` to determine the ordering
   */
  sort_relation(std::shared_ptr<relation> input_relation,
                std::vector<std::shared_ptr<relation>> subquery_relations,
                std::vector<cudf::order> column_orders,
                std::vector<cudf::null_order> null_precedences,
                std::vector<std::unique_ptr<expression>> expressions);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::sort; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the list of expressions.
   *
   * @note The returned expressions do not share ownership. This object must be kept alive for the
   * returned expressions to be valid.
   *
   * @return List of output expressions
   */
  [[nodiscard]] std::vector<expression*> expressions_unsafe() const noexcept
  {
    return gqe::utility::to_raw_ptrs(_expressions);
  }

  /**
   * @brief Accessor for column orders. Indicates the direction of the sort for each column
   */
  [[nodiscard]] std::vector<cudf::order> column_orders() const noexcept { return _column_orders; }

  /**
   * @brief Accessor for null orders. Indicates whether to return NULLs first or last for each
   * column
   */
  [[nodiscard]] std::vector<cudf::null_order> null_orders() const noexcept { return _null_orders; }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::vector<std::unique_ptr<expression>> _expressions;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_orders;
};

}  // namespace logical
}  // namespace gqe