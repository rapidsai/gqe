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

#include <memory>

namespace gqe::optimizer {

/**
 * @brief Rewrite join predicates to cast mixed-width decimal operands.
 *
 * For each join, rewrites the join condition so mixed-width decimal binary ops cast the narrower
 * operand to the wider decimal `type_id` (scale preserved). This rule does not insert projection
 * nodes.
 */
class normalize_decimal_widths : public optimization_rule {
 public:
  normalize_decimal_widths(catalog const* cat) : optimization_rule(cat) {}

  [[nodiscard]] std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                                         bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::normalize_decimal_widths;
  }

 private:
  class apply_visitor;

  void normalize_join(gqe::logical::join_relation* join, bool& rule_applied) const;

  void rewrite_join_condition(gqe::logical::join_relation* join, bool& rule_applied) const;
};

}  // namespace gqe::optimizer
