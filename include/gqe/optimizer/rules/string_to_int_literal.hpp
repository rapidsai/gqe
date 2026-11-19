/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
using relation_t = gqe::logical::relation::relation_type;

/**
 * @brief This rule modifies the expressions that have cudf::type_id::INT8 columns being
 * equated to string literals, by changing the string literal to int8 literal
 */
class string_to_int_literal : public optimization_rule {
 public:
  string_to_int_literal(catalog const* cat) : optimization_rule(cat) {}

  std::shared_ptr<logical::relation> apply(std::shared_ptr<logical::relation> root,
                                           bool& rule_applied) const override;

  [[nodiscard]] logical_optimization_rule_type type() const noexcept override
  {
    return logical_optimization_rule_type::string_to_int_literal;
  }

 private:
  class apply_visitor;
};

}  // namespace optimizer
}  // namespace gqe
