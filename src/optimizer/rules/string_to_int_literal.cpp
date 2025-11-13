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

#include <cudf/types.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/optimizer/rules/string_to_int_literal.hpp>

#include <iostream>
#include <memory>

std::shared_ptr<gqe::logical::relation> gqe::optimizer::string_to_int_literal::try_optimize(
  std::shared_ptr<logical::relation> logical_relation, bool& rule_applied) const
{
  auto expr_modifier =
    [&](expression* expr,
        std::vector<cudf::data_type> const& column_types) -> std::unique_ptr<gqe::expression> {
    if (expr->type() == gqe::expression::expression_type::binary_op) {
      auto op = dynamic_cast<gqe::binary_op_expression*>(expr);

      if (op->binary_operator() == cudf::binary_operator::EQUAL) {
        assert(op->children().size() == 2);
        auto left_expr  = op->children()[0];
        auto right_expr = op->children()[1];

        if (left_expr->type() == gqe::expression::expression_type::column_reference &&
            left_expr->data_type(column_types) == cudf::data_type(cudf::type_id::INT8) &&
            right_expr->type() == gqe::expression::expression_type::literal &&
            right_expr->data_type(column_types) == cudf::data_type(cudf::type_id::STRING)) {
          auto string_literal = dynamic_cast<gqe::literal_expression<std::string>*>(right_expr);
          auto string_value   = string_literal->value();

          if (string_value.size() == 1) {
            int8_t char_value = static_cast<int8_t>(string_value[0]);
            rule_applied      = true;
            std::shared_ptr<gqe::expression> cloned_left_expr(left_expr->clone());
            auto new_right_expr = std::make_shared<gqe::literal_expression<int8_t>>(char_value);

            auto new_conditon =
              std::make_unique<gqe::equal_expression>(cloned_left_expr, new_right_expr);
            return new_conditon;
          }
        }
      }
    }
    // Pattern not found
    return static_cast<std::unique_ptr<gqe::expression>>(nullptr);
  };

  rewrite_relation_expressions(logical_relation.get(), expr_modifier, transform_direction::UP);
  return logical_relation;
}
