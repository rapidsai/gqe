/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

namespace gqe::optimizer {

/**
 * @brief Extract non-representable join predicate operands into per-side projects.
 *
 * The executor can only translate join predicates where per-side operands map to a representable
 * AST expression type. For other predicates (for example, some decimal cast/arithmetic forms), this
 * rule inserts a project under the affected join input(s) to precompute those operands, then
 * rewrites the join condition to reference the projected columns.
 *
 * - TODO: unify this join extraction path with
 *   `complex_expression_extraction_into_project` once both policies can run as a single pass.
 */
class join_predicate_extraction_into_project : public optimization_rule {
 public:
  join_predicate_extraction_into_project(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::join_predicate_extraction_into_project;
  }

 private:
  class apply_visitor;

  void extract_join(logical::join_relation* join, bool& rule_applied) const;
};

}  // namespace gqe::optimizer
