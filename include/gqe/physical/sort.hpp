/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
                     std::vector<cudf::null_order> null_precedences)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _keys(std::move(keys)),
      _column_orders(std::move(column_orders)),
      _null_precedences(std::move(null_precedences))
  {
  }

  /**
   * @brief Return the keys which determine the ordering of the sort.
   */
  std::vector<expression*> keys_unsafe() { return utility::to_raw_ptrs(_keys); }

  /**
   * @brief Return the desired order for each column.
   */
  std::vector<cudf::order> column_orders() { return _column_orders; }

  /**
   * @brief Return whether a null element is smaller or larger than other elements.
   */
  std::vector<cudf::null_order> null_precedences() { return _null_precedences; }

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_precedences;
};

/**
 * @brief Sort the input partitions by concatenating into a single one.
 */
class concatenate_sort_relation : public sort_relation_base {
  using sort_relation_base::sort_relation_base;

 public:
  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }
};

}  // namespace physical
}  // namespace gqe
