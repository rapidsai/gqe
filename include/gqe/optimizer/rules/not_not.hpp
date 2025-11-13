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

#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe {
namespace optimizer {
using relation_t = gqe::logical::relation::relation_type;
/**
 * @brief This is a test rule and will not be used in actual execution. DataFusion already
 * applies this optimization to the logical plan. The rule searches for a pattern of
 * `not(not(expression))` and simplify it to `expression` in the plan.
 */
class not_not_rewrite : public optimization_rule {
 public:
  not_not_rewrite(catalog const* cat)
    : optimization_rule(cat, optimization_rule::transform_direction::UP)
  {
  }

  std::shared_ptr<logical::relation> try_optimize(
    std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::not_not_rewrite;
  }
};

}  // namespace optimizer
}  // namespace gqe
