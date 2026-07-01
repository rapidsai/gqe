/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/not_not.hpp>

#include <gqe/catalog.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/logical/project.hpp>

#include <cudf/unary.hpp>

#include <iostream>
#include <memory>

namespace gqe::optimizer {

class not_not_rewrite::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(not_not_rewrite const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit_relation(gqe::logical::relation* rel) override
  {
    visit_children(rel);  // post-order: recurse first

    bool applied_here  = false;
    auto expr_modifier = [&applied_here](
                           gqe::expression* expr,
                           [[maybe_unused]] std::vector<cudf::data_type> const& column_types)
      -> std::unique_ptr<gqe::expression> {
      if (expr->type() == gqe::expression::expression_type::unary_op) {
        auto op = dynamic_cast<gqe::unary_op_expression*>(expr);
        assert(op->children().size() == 1);
        auto inner_expr = op->children()[0];
        if (op->unary_operator() == cudf::unary_operator::NOT) {
          if (inner_expr->type() == gqe::expression::expression_type::unary_op) {
            auto inner_op = dynamic_cast<gqe::unary_op_expression*>(inner_expr);
            assert(inner_op->children().size() == 1);
            if (inner_op->unary_operator() == cudf::unary_operator::NOT) {
              applied_here = true;
              return inner_op->children()[0]->clone();
            }
          }
        }
      }
      return nullptr;
    };

    _rule.rewrite_relation_expressions(rel, std::move(expr_modifier), transform_direction::DOWN);
    _rule_applied |= applied_here;
  }

 private:
  not_not_rewrite const& _rule;
  bool& _rule_applied;
};

}  // namespace gqe::optimizer

std::shared_ptr<gqe::logical::relation> gqe::optimizer::not_not_rewrite::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}
