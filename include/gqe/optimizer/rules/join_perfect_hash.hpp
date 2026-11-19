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

/**
 * @brief This rule inspects each logical `join_relation` and enables perfect hashing when the
 * equi-join keys on one (or both) sides cover a primary key or unique key.
 *
 * - Uniqueness information is derived from each side's `unique_keys()` relation properties,
 *   which are populated by the `uniqueness_propagation` rule. This rule must therefore run after
 *   `uniqueness_propagation`.
 * - Only inner joins with a pure equi-join predicate are eligible.
 * - The rule sets `join_relation::_perfect_hash_policy` to `left`, `right`, or `either`
 *   depending on which side's join keys are unique. The physical plan builder (`plan_join.cpp`)
 *   maps this to an actual `perfect_hashing` flag only when the unique side is also the build
 *   (broadcast) side.
 * - NULL_EQUALS predicates (IS NOT DISTINCT FROM) are excluded: a nullable UNIQUE column permits
 *   multiple NULL rows, so perfect hashing on NULL-equal keys could map different NULL rows to
 *   the same slot and produce incorrect results.
 * - LIMITATION: the rule does not verify that the join-key columns are non-nullable.
 *   `perfect_hash::perfect_join` requires *both* sides' keys to contain no nulls and throws
 *   otherwise. A SQL UNIQUE key may itself be nullable, and the probe side is unchecked. This is
 *   tracked as a follow-up; until non-null propagation exists, eligibility relies on the join keys
 *   being non-null in practice. See issue #404.
 */
class join_perfect_hash : public optimization_rule {
 public:
  join_perfect_hash(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::join_perfect_hash;
  }

 private:
  class apply_visitor;
};

}  // namespace optimizer
}  // namespace gqe
