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
