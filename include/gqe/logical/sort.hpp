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

#pragma once

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

class sort_relation : public relation {
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
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

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
    return utility::to_raw_ptrs(_expressions);
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

 private:
  std::vector<std::unique_ptr<expression>> _expressions;
  std::vector<cudf::order> _column_orders;
  std::vector<cudf::null_order> _null_orders;
  std::vector<cudf::data_type> _data_types;
};

}  // namespace logical
}  // namespace gqe