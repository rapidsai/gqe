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

#include <gqe/optimizer/logical_optimization.hpp>

namespace gqe {
namespace optimizer {

/**
 * @brief Remove redundant identity projections from the logical plan.
 *
 * DataFusion's Substrait producer wraps a bare column list (e.g. `SELECT a, b, c`) in a
 * `ProjectRel` even when the child relation already produces exactly those columns in order. GQE
 * materializes this as an identity `project_relation`, which is not free at runtime.
 *
 * This rule removes a project whose output is a strict identity over its child (see
 * `project_relation::is_identity()`), regardless of the child type, splicing the child into the
 * project's place. It covers every position a project can occupy — a child of another relation,
 * the root of a subquery, and the root of the plan itself — and collapses chains of stacked
 * identity projects in a single traversal.
 *
 * ### Relationship to `projection_pushdown`
 *
 * The two rules solve different problems, which is why they are kept separate:
 * - `projection_pushdown` is a **column-pruning** transform: it moves a project's column selection
 *   into the child's `_projection_indices` so the child materializes fewer columns. It therefore
 *   **changes what the child produces**, and is limited to `filter`/`join` — the only relations
 *   that own a `_projection_indices` member (enforced by `optimizable_child_relation<T>()`).
 * - This rule is a **redundant-node removal** transform: it deletes a project that is already a
 *   no-op, **changing nothing** about what any relation produces, and so applies to any child type.
 *
 * The distinguishing question is whether the transform changes what a relation produces — for
 * `projection_pushdown` that is the point; here it must never happen.
 *
 * The two overlap only on an identity project over a `filter`/`join`, where `projection_pushdown`
 * already removes it as a degenerate case of its all-column-reference branch; both yield the same
 * plan, and `projection_pushdown` runs first in the default pipeline. This rule is intentionally
 * not restricted to the remaining child types, so that it stays self-contained and independent of
 * rule ordering.
 *
 * ### Safety
 *
 * Removal is schema-name-safe: the query's final output column names come from the Substrait
 * `RelRoot.names` and are applied to the result batch after execution, independent of the plan
 * tree shape.
 *
 * Removing a subquery root is index-safe: the subquery list is rewritten in place and never
 * resized, so any `subquery_expression::relation_index()` pointing into it stays valid.
 */
class identity_project_elimination : public optimization_rule {
 public:
  identity_project_elimination(catalog const* cat) : optimization_rule(cat) {}

  /**
   * @copydoc optimization_rule::apply()
   */
  std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                           bool& rule_applied) const override;

  /**
   * @copydoc optimization_rule::type()
   */
  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::identity_project_elimination;
  }

 private:
  class apply_visitor;
};

}  // namespace optimizer
}  // namespace gqe
