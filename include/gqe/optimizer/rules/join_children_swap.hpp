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

#include <gqe/optimizer/logical_optimization.hpp>
#include <gqe/physical/join.hpp>

namespace gqe {
namespace optimizer {
/**
 * @brief This rule inspect each logical `join_relation` and swap the children based on the
 * estimated size of each child. The `default_broadcast_policy` lets the optimizer knows which side
 * should be smaller. This rule is current only used for testing the optimizer and not in actual
 * execution. Note that the semantic of this rule is currently implemented manually in the physical
 * translation phase.
 */
class join_children_swap : public optimization_rule {
 public:
  join_children_swap(catalog const* cat, gqe::physical::broadcast_policy default_broadcast_policy)
    : optimization_rule(cat, optimization_rule::transform_direction::UP),
      _default_broadcast_policy(default_broadcast_policy)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::join_children_swap;
  }

  gqe::physical::broadcast_policy default_broadcast_policy() const noexcept
  {
    return _default_broadcast_policy;
  };

 private:
  gqe::physical::broadcast_policy _default_broadcast_policy;
  void swap_join_keys_inplace(gqe::logical::join_relation* join,
                              cudf::size_type n_cols_left,
                              cudf::size_type n_cols_right) const;
};

}  // namespace optimizer
}  // namespace gqe
