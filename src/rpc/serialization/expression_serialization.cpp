/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/if_then_else.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/scalar_function.hpp>
#include <gqe/expression/subquery.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/rpc/serialization/data_type.hpp>
#include <gqe/rpc/serialization/expression.hpp>

#include <cstring>
#include <string>

namespace gqe::rpc {

namespace {

template <typename T>
std::string value_to_bytes(T value)
{
  return std::string(reinterpret_cast<char const*>(&value), sizeof(value));
}

template <typename T>
void serialize_literal(proto::LiteralExpr* msg,
                       literal_expression<T> const* expr,
                       cudf::type_id type_id)
{
  *msg->mutable_data_type() = serialize_data_type(cudf::data_type{type_id});
  msg->set_value(value_to_bytes(expr->value()));
  msg->set_is_null(expr->is_null());
}

class serialize_expression_visitor : public expression_visitor {
 public:
  proto::Expression result;

  void visit(column_reference_expression const* expr) override
  {
    auto* msg = result.mutable_column_reference();
    msg->set_column_idx(expr->column_idx());
    msg->set_column_name(expr->column_name());
  }

  void visit(literal_expression<int8_t> const* expr) override
  {
    serialize_literal(result.mutable_literal(), expr, cudf::type_id::INT8);
  }

  void visit(literal_expression<int32_t> const* expr) override
  {
    serialize_literal(result.mutable_literal(), expr, cudf::type_id::INT32);
  }

  void visit(literal_expression<int64_t> const* expr) override
  {
    serialize_literal(result.mutable_literal(), expr, cudf::type_id::INT64);
  }

  void visit(literal_expression<float> const* expr) override
  {
    serialize_literal(result.mutable_literal(), expr, cudf::type_id::FLOAT32);
  }

  void visit(literal_expression<double> const* expr) override
  {
    serialize_literal(result.mutable_literal(), expr, cudf::type_id::FLOAT64);
  }

  void visit(literal_expression<std::string> const* expr) override
  {
    auto* msg                 = result.mutable_literal();
    *msg->mutable_data_type() = serialize_data_type(cudf::data_type{cudf::type_id::STRING});
    msg->set_value(expr->value());
    msg->set_is_null(expr->is_null());
  }

  void visit(literal_expression<cudf::timestamp_D> const* expr) override
  {
    auto* msg                 = result.mutable_literal();
    *msg->mutable_data_type() = serialize_data_type(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
    msg->set_value(value_to_bytes(expr->value().time_since_epoch().count()));
    msg->set_is_null(expr->is_null());
  }

  void visit(is_null_expression const* expr) override
  {
    auto* msg             = result.mutable_is_null();
    *msg->mutable_input() = serialize_expression(expr->children()[0]);
  }

  void visit(cast_expression const* expr) override
  {
    auto* msg                   = result.mutable_cast();
    *msg->mutable_input()       = serialize_expression(expr->children()[0]);
    *msg->mutable_target_type() = serialize_data_type(expr->out_type());
  }

  void visit(binary_op_expression const* expr) override
  {
    auto* msg = result.mutable_binary_op();
    msg->set_op(static_cast<int32_t>(expr->binary_operator()));
    auto children         = expr->children();
    *msg->mutable_left()  = serialize_expression(children[0]);
    *msg->mutable_right() = serialize_expression(children[1]);
  }

  void visit(unary_op_expression const* expr) override
  {
    auto* msg = result.mutable_unary_op();
    msg->set_op(static_cast<int32_t>(expr->unary_operator()));
    *msg->mutable_input() = serialize_expression(expr->children()[0]);
  }

  void visit(if_then_else_expression const* expr) override
  {
    auto* msg                 = result.mutable_if_then_else();
    auto children             = expr->children();
    *msg->mutable_condition() = serialize_expression(children[0]);
    *msg->mutable_then_expr() = serialize_expression(children[1]);
    *msg->mutable_else_expr() = serialize_expression(children[2]);
  }

  void visit(scalar_function_expression const* expr) override
  {
    auto* msg = result.mutable_scalar_function();
    msg->set_function_kind(static_cast<int32_t>(expr->fn_kind()));

    for (auto const* child : expr->children()) {
      *msg->add_arguments() = serialize_expression(child);
    }

    switch (expr->fn_kind()) {
      case scalar_function_expression::function_kind::datepart: {
        auto const* dp = static_cast<datepart_expression const*>(expr);
        msg->set_datetime_component(static_cast<int32_t>(dp->component()));
        break;
      }
      case scalar_function_expression::function_kind::like: {
        auto const* lk = static_cast<like_expression const*>(expr);
        msg->set_pattern(lk->pattern());
        msg->set_escape_character(lk->escape_character());
        msg->set_ignore_case(lk->ignore_case());
        break;
      }
      case scalar_function_expression::function_kind::round: {
        auto const* rn = static_cast<round_expression const*>(expr);
        msg->set_decimal_places(rn->decimal_places());
        break;
      }
      case scalar_function_expression::function_kind::substr: {
        auto const* ss = static_cast<substr_expression const*>(expr);
        msg->set_start(ss->start());
        msg->set_length(ss->length());
        break;
      }
    }
  }

  void visit(subquery_expression const* expr) override
  {
    auto const* ip = static_cast<in_predicate_expression const*>(expr);
    auto* msg      = result.mutable_in_predicate();
    msg->set_relation_index(static_cast<int32_t>(ip->relation_index()));
    for (auto const* child : expr->children()) {
      *msg->add_needles() = serialize_expression(child);
    }
  }
};

}  // namespace

proto::Expression serialize_expression(expression const* expr)
{
  if (!expr) return {};
  serialize_expression_visitor visitor;
  expr->accept(visitor);
  return visitor.result;
}

}  // namespace gqe::rpc
