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
/**
 * @brief This rule pushes projection to filter by using the projection indices of filter relation
 * to materialize only necessary columns, and in the correct order, if required.
 *
 * When the project relation does only dropping/reordering of columns, we can accomplish
 * that directly in the filter relation using its projection indices. Hence, optimizer
 * removes the project relation from the query plan
 *
 * Whereas, if the project relation has column operations (ex. unary op, binary op, etc)
 * optimizer keeps the project relation to do any operations and reordering,
 * but uses the filter relation to drop any non-necessary columns.
 * The column references in the project relation are modified accordingly.
 */
class push_projection_to_filter : public optimization_rule {
 public:
  push_projection_to_filter(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::UP)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::push_projection_to_filter;
  }

 private:
  void rewrite_project_relation(std::shared_ptr<gqe::logical::relation> project,
                                std::vector<cudf::size_type> const& required_cr_indices) const;

  void rewrite_filter_relation(std::shared_ptr<gqe::logical::filter_relation> filter,
                               std::vector<cudf::size_type> const& required_cr_indices) const;
};

}  // namespace optimizer
}  // namespace gqe
