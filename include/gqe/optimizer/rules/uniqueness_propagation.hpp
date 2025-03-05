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

#include <gqe/catalog.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>
#include <memory>

namespace gqe {
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;
/**
 * @brief This rule determines how uniqueness properties from the relation's children's columns can
 * be conservatively propagated to the columns of the relation itself. This rule is current only
 * used for testing the optimizer and not in actual execution.
 */
class uniqueness_propagation : public optimization_rule {
 public:
  uniqueness_propagation(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::UP)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::uniqueness_propagation;
  }
};

}  // namespace optimizer
}  // namespace gqe