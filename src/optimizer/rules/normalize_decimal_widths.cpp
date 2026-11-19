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

#include <gqe/optimizer/rules/normalize_decimal_widths.hpp>

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/logical/join.hpp>

#include <cudf/utilities/traits.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] bool column_index_valid(gqe::expression const* expr,
                                      std::vector<cudf::data_type> const& column_types)
{
  if (expr->type() != gqe::expression::expression_type::column_reference) { return true; }
  auto const idx = static_cast<gqe::column_reference_expression const*>(expr)->column_idx();
  return idx >= 0 && idx < static_cast<cudf::size_type>(column_types.size());
}

[[nodiscard]] std::optional<cudf::data_type> expr_data_type(
  gqe::expression const* expr, std::vector<cudf::data_type> const& column_types)
{
  if (!column_index_valid(expr, column_types)) { return std::nullopt; }
  return expr->data_type(column_types);
}

[[nodiscard]] std::unique_ptr<gqe::expression> rebuild_binary_op(
  cudf::binary_operator op,
  std::shared_ptr<gqe::expression> lhs,
  std::shared_ptr<gqe::expression> rhs)
{
  switch (op) {
    case cudf::binary_operator::ADD:
      return std::make_unique<gqe::add_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::SUB:
      return std::make_unique<gqe::subtract_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::MUL:
      return std::make_unique<gqe::multiply_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::DIV:
      return std::make_unique<gqe::divide_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::EQUAL:
      return std::make_unique<gqe::equal_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::NULL_EQUALS:
      return std::make_unique<gqe::nulls_equal_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::NOT_EQUAL:
      return std::make_unique<gqe::not_equal_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::LESS:
      return std::make_unique<gqe::less_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::GREATER:
      return std::make_unique<gqe::greater_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::LESS_EQUAL:
      return std::make_unique<gqe::less_equal_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::GREATER_EQUAL:
      return std::make_unique<gqe::greater_equal_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::NULL_LOGICAL_AND:
      return std::make_unique<gqe::logical_and_expression>(std::move(lhs), std::move(rhs));
    case cudf::binary_operator::NULL_LOGICAL_OR:
      return std::make_unique<gqe::logical_or_expression>(std::move(lhs), std::move(rhs));
    default:
      throw std::logic_error("normalize_decimal_widths: unsupported binary operator for rebuild");
  }
}

[[nodiscard]] int type_rank(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::DECIMAL32: return 0;
    case cudf::type_id::DECIMAL64: return 1;
    case cudf::type_id::DECIMAL128: return 2;
    default: return -1;
  }
}

[[nodiscard]] std::unique_ptr<gqe::expression> normalize_binary_expression(
  gqe::expression const* expr, std::vector<cudf::data_type> const& column_types)
{
  if (expr->type() != gqe::expression::expression_type::binary_op) { return nullptr; }

  auto const* bop     = static_cast<gqe::binary_op_expression const*>(expr);
  auto const children = expr->children();
  assert(children.size() == 2);

  auto const lt_opt = expr_data_type(children[0], column_types);
  auto const rt_opt = expr_data_type(children[1], column_types);
  if (!lt_opt || !rt_opt) { return nullptr; }

  auto const lt = *lt_opt;
  auto const rt = *rt_opt;
  if (!(cudf::is_fixed_point(lt) && cudf::is_fixed_point(rt) && lt.id() != rt.id())) {
    return nullptr;
  }

  auto new_lhs = std::shared_ptr<gqe::expression>(children[0]->clone().release());
  auto new_rhs = std::shared_ptr<gqe::expression>(children[1]->clone().release());

  bool const wrap_lhs        = type_rank(lt.id()) < type_rank(rt.id());
  auto const wide_id         = wrap_lhs ? rt.id() : lt.id();
  auto const preserved_scale = wrap_lhs ? lt.scale() : rt.scale();
  cudf::data_type const target_type{wide_id, preserved_scale};
  if (wrap_lhs) {
    new_lhs = std::make_shared<gqe::cast_expression>(std::move(new_lhs), target_type);
  } else {
    new_rhs = std::make_shared<gqe::cast_expression>(std::move(new_rhs), target_type);
  }

  return rebuild_binary_op(bop->binary_operator(), std::move(new_lhs), std::move(new_rhs));
}

}  // namespace

namespace gqe::optimizer {

class normalize_decimal_widths::apply_visitor : public gqe::logical::relation_visitor {
 public:
  apply_visitor(normalize_decimal_widths const& rule, bool& rule_applied)
    : _rule{rule}, _rule_applied{rule_applied}
  {
  }

  void visit(gqe::logical::join_relation* join) override
  {
    visit_children(join);
    _rule.normalize_join(join, _rule_applied);
  }

  void visit_relation(gqe::logical::relation* rel) override { visit_children(rel); }

 private:
  normalize_decimal_widths const& _rule;
  bool& _rule_applied;
};

void normalize_decimal_widths::normalize_join(gqe::logical::join_relation* join,
                                              bool& rule_applied) const
{
  rewrite_join_condition(join, rule_applied);
}

void normalize_decimal_widths::rewrite_join_condition(gqe::logical::join_relation* join,
                                                      bool& rule_applied) const
{
  if (join->condition() == nullptr) { return; }
  bool local_applied = false;
  auto modifier      = [&](
                    gqe::expression* expr,
                    std::vector<cudf::data_type> const& types) -> std::unique_ptr<gqe::expression> {
    auto normalized = normalize_binary_expression(expr, types);
    if (normalized) {
      local_applied = true;
      return normalized;
    }
    return nullptr;
  };
  rewrite_relation_expressions(join, modifier, transform_direction::UP);
  rule_applied |= local_applied;
}

std::shared_ptr<gqe::logical::relation> normalize_decimal_widths::apply(
  std::shared_ptr<gqe::logical::relation> root, bool& rule_applied) const
{
  apply_visitor visitor{*this, rule_applied};
  root->accept(visitor);
  return root;
}

}  // namespace gqe::optimizer
