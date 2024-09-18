/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe {
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;

template <typename T>
constexpr bool optimizable_child_relation()
{
  return std::is_same_v<T, gqe::logical::filter_relation> ||
         std::is_same_v<T, gqe::logical::join_relation>;
}

/**
 * @brief This rule optimizes the logical plan by attempting to push projection to its child
 * relation. Projection indices of the child relation are used to materialize only necessary
 * columns, and in the correct order, if required. Currently, this is only supported by filter and
 * join relations
 *
 * When the project relation does only dropping/reordering of columns, we can accomplish
 * that directly in the filter/join relation using its projection indices. Hence, optimizer
 * removes the project relation from the query plan
 *
 * Whereas, if the project relation has column operations (ex. unary op, binary op, etc)
 * optimizer keeps the project relation to do any operations and reordering,
 * but uses the filter/join relation to drop any non-necessary columns.
 * The column references in the project relation are modified accordingly.
 */
class projection_pushdown : public optimization_rule {
 public:
  projection_pushdown(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::UP)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::projection_pushdown;
  }

 private:
  void rewrite_project_relation(std::shared_ptr<gqe::logical::project_relation> project,
                                std::vector<cudf::size_type> const& required_cr_indices) const;

  template <typename T, typename = std::enable_if_t<optimizable_child_relation<T>()>>
  void rewrite_child_relation(std::shared_ptr<T> child_relation,
                              std::vector<cudf::size_type> const& required_cr_indices) const;

  template <typename T, typename = std::enable_if_t<optimizable_child_relation<T>()>>
  std::shared_ptr<logical::relation> try_pushdown(
    std::shared_ptr<gqe::logical::project_relation> project,
    std::shared_ptr<gqe::logical::relation> child,
    bool& rule_applied) const;
};

}  // namespace optimizer
}  // namespace gqe
