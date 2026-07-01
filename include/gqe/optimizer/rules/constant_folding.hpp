/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Fold trivial constant predicates in the logical plan.
 *
 * Currently handles two cases:
 *  1. `filter_relation` whose condition is `literal_expression<bool>(true)`:
 *     Rewritten to a `project_relation` over the same child whose output expressions are
 *     `column_reference_expression`s built from `projection_indices`. An empty
 *     `projection_indices` produces a zero-expression project (0-column output), preserving
 *     the filter's output schema contract while dropping the dead condition.
 *  2. `read_relation` whose `partial_filter` is `literal_expression<bool>(true)`: the partial
 *     filter is cleared.
 *
 * This rule is intended to be extended to other constant-folding cases (e.g. `literal(false)` →
 * empty relation, `x AND true` → `x`, evaluation of constant subexpressions). It runs first in
 * `make_default_optimizer_rules()` so that downstream rules don't waste work on no-op filters.
 */
class constant_folding : public optimization_rule {
 public:
  constant_folding(catalog const* cat) : optimization_rule(cat) {}

  std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                           bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::constant_folding;
  }

 private:
  class apply_visitor;
};

}  // namespace optimizer
}  // namespace gqe
