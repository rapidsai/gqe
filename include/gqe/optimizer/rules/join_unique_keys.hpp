/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/catalog.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <memory>

namespace gqe {
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;
/**
 * @brief This rule inspects each logical `join_relation` and determines whether each inner join can
 * be executed using the unique keys optimization. This rule should be applied after rule
 * `uniqueness_propagation`. Currently this ordering must be manually enforced when instantiating
 * `optimization_configuration`.
 */
class join_unique_keys : public optimization_rule {
 public:
  join_unique_keys(catalog const* cat)
    : optimization_rule(cat,
                        optimization_rule::transform_direction::UP /* can be either up or down */)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::join_unique_keys;
  }
};

}  // namespace optimizer
}  // namespace gqe