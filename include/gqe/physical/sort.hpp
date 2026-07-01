/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/expression/expression.hpp>
#include <gqe/physical/relation.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Abstract base class for all physical sort relations.
 */
class sort_relation_base : public relation {
 public:
  /**
   * @brief Construct a physical sort relation.
   *
   * Create a new table that reorders the rows of `input` according to the lexicographic ordering of
   * the rows of `keys`.
   *
   * @param[in] input Input table to be sorted.
   * @param[in] subquery_relations Subquery relations that are referenced within the `keys`
   * expressions.
   * @param[in] keys Expressions evaluated on `input` to determine the ordering.
   * @param[in] column_orders Desired order for each column in `keys`. The size of this argument
   * must be the same as the size of `keys`.
   * @param[in] null_precedences Whether a null element is smaller or larger than other elements.
   * The size of this argument must be the same as the size of `keys`.
   */
  sort_relation_base(std::shared_ptr<relation> input,
                     std::vector<std::shared_ptr<relation>> subquery_relations,
                     std::vector<std::unique_ptr<expression>> keys,
                     std::vector<cudf::order> column_orders,
                     std::vector<cudf::null_order> null_precedences,
                     bool use_like_shift_and = false)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _keys(std::move(keys)),
      _column_orders(std::move(column_orders)),
      _null_precedences(std::move(null_precedences)),
      _use_like_shift_and(use_like_shift_and)
  {
  }

  /**
   * @brief Return the keys which determine the ordering of the sort.
   */
  std::vector<expression*> keys_unsafe() const { return utility::to_raw_ptrs(_keys); }

  /**
   * @brief Return the desired order for each column.
   */
  std::vector<cudf::order> column_orders() const { return _column_orders; }

  /**
   * @brief Return whether a null element is smaller or larger than other elements.
   */
  std::vector<cudf::null_order> null_precedences() const { return _null_precedences; }

  /**
   * @brief Return whether to use the shift-and algorithm for LIKE middle-pattern matching when
   * evaluating the sort keys.
   */
  [[nodiscard]] bool use_like_shift_and() const noexcept { return _use_like_shift_and; }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @brief Print all members as well as output data types
   */
  [[nodiscard]] std::string print() const;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_precedences;
  bool _use_like_shift_and;
};

/**
 * @brief Sort the input partitions by concatenating into a single one.
 */
class concatenate_sort_relation : public sort_relation_base {
  using sort_relation_base::sort_relation_base;

 public:
  [[nodiscard]] relation_type type() const noexcept override
  {
    return relation_type::concatenate_sort;
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;
};

}  // namespace physical
}  // namespace gqe
