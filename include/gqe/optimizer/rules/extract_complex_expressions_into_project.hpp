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

namespace gqe {
namespace optimizer {

/**
 * @brief Extract complex expressions out of join and aggregate operators into a project.
 *
 * The QEP join and aggregate operators only carry simple `column_reference` expressions: joins
 * compare `column_reference` operands and aggregates group by `column_reference` keys and
 * aggregate `column_reference` value columns. This rule pulls any complex (non-`column_reference`)
 * expression out into a new `project_relation` inserted below the operator, leaving only
 * `column_reference` expressions behind.
 *
 * - **Aggregate**: a project emitting `[keys..., measure-args...]` is inserted as the child; the
 *   aggregate keys become `column_reference(0..k-1)` and measure-args become
 *   `column_reference(k..k+m-1)`.
 * - **Join** (inner only): complex equi-key operands are materialized as trailing columns by a
 *   project on each side; the condition and projection indices are rewritten to reference them.
 *   The rule is conservative — any condition it cannot safely split (non-equality with complex
 *   operands, operands spanning both inputs, subqueries, non-inner joins) is left untouched.
 */
class complex_expression_extraction_into_project : public optimization_rule {
 public:
  complex_expression_extraction_into_project(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::complex_expression_extraction_into_project;
  }

 private:
  class apply_visitor;

  /**
   * @brief Extract complex keys/measure-args of an aggregate into a child project.
   *
   * @param agg Aggregate relation to rewrite (no-op if all keys/measures are column references)
   * @param rule_applied Set to true if the aggregate was rewritten
   */
  void extract_aggregate(logical::aggregate_relation* agg, bool& rule_applied) const;

  /**
   * @brief Extract complex equi-key operands of an inner join into per-side projects.
   *
   * @param join Join relation to rewrite (no-op / untouched if the condition cannot be split)
   * @param rule_applied Set to true if the join was rewritten
   */
  void extract_join(logical::join_relation* join, bool& rule_applied) const;
};

}  // namespace optimizer
}  // namespace gqe
