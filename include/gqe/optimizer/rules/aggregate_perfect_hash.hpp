/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/catalog.hpp>
#include <gqe/logical/relation.hpp>
#include <gqe/optimizer/logical_optimization.hpp>

#include <memory>

namespace gqe {
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;

/**
 * @brief This rule inspects each logical `aggregate_relation` and enables perfect hashing when its
 * group-by keys are unique in the input relation and all have fixed-width types.
 *
 * - Uniqueness is derived from the input relation's `unique_keys()` properties, which are
 *   populated by the `uniqueness_propagation` rule. This rule must therefore run after
 *   `uniqueness_propagation`. Currently this ordering must be manually enforced when instantiating
 *   `optimization_configuration`.
 * - Fixed-width check ensures the executor's `libperfect::unique_indices` can handle the keys.
 * - Pure reductions (no group-by keys) are skipped; perfect hashing applies only to grouped aggs.
 */
class aggregate_perfect_hash : public optimization_rule {
 public:
  aggregate_perfect_hash(catalog const* cat) : optimization_rule(cat) {}

  std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                           bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::aggregate_perfect_hash;
  }

 private:
  class apply_visitor;
};

}  // namespace optimizer
}  // namespace gqe
