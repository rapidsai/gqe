/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/rules/string_to_int_literal.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/literal.hpp>

#include <cudf/types.hpp>

#include <iostream>
#include <memory>

namespace gqe::optimizer {

class string_to_int_literal::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(string_to_int_literal const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit_relation(gqe::logical::relation* rel) override
  {
    visit_children(rel);  // post-order: recurse first

    bool applied_here = false;
    auto expr_modifier =
      [&applied_here](
        gqe::expression* expr,
        std::vector<cudf::data_type> const& column_types) -> std::unique_ptr<gqe::expression> {
      if (expr->type() == gqe::expression::expression_type::binary_op) {
        auto op = dynamic_cast<gqe::binary_op_expression*>(expr);

        if (op->binary_operator() == cudf::binary_operator::EQUAL) {
          assert(op->children().size() == 2);
          auto left_expr  = op->children()[0];
          auto right_expr = op->children()[1];

          // DataFusion's type coercion wraps an INT8 column in
          // Cast(STRING, …) when it sees a comparison against a STRING
          // literal. Peel that cast so the rule also matches the coerced
          // shape; non-cast inputs fall through unchanged.
          auto col_candidate = left_expr;
          if (col_candidate->type() == gqe::expression::expression_type::cast) {
            auto cast = dynamic_cast<gqe::cast_expression*>(col_candidate);
            if (cast->out_type() == cudf::data_type(cudf::type_id::STRING)) {
              col_candidate = cast->children()[0];
            }
          }

          if (col_candidate->type() == gqe::expression::expression_type::column_reference &&
              col_candidate->data_type(column_types) == cudf::data_type(cudf::type_id::INT8) &&
              right_expr->type() == gqe::expression::expression_type::literal &&
              right_expr->data_type(column_types) == cudf::data_type(cudf::type_id::STRING)) {
            auto string_literal = dynamic_cast<gqe::literal_expression<std::string>*>(right_expr);
            auto string_value   = string_literal->value();

            if (string_value.size() == 1) {
              // Treats the literal as the byte value of its character. Correct
              // for char-as-byte Int8 columns; wrong for numeric Int8. Rule
              // does not distinguish.
              int8_t char_value = static_cast<int8_t>(string_value[0]);
              applied_here      = true;
              std::shared_ptr<gqe::expression> cloned_left_expr(col_candidate->clone());
              auto new_right_expr = std::make_shared<gqe::literal_expression<int8_t>>(char_value);

              auto new_conditon =
                std::make_unique<gqe::equal_expression>(cloned_left_expr, new_right_expr);
              return new_conditon;
            }
          }
        }
      }
      return nullptr;
    };

    _rule.rewrite_relation_expressions(rel, std::move(expr_modifier), transform_direction::UP);
    _rule_applied |= applied_here;
  }

 private:
  string_to_int_literal const& _rule;
  bool& _rule_applied;
};

}  // namespace gqe::optimizer

std::shared_ptr<gqe::logical::relation> gqe::optimizer::string_to_int_literal::apply(
  std::shared_ptr<logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}
