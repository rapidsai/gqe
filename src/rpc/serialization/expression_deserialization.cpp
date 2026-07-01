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
#include <stdexcept>
#include <string>

namespace gqe::rpc {

namespace {

std::shared_ptr<expression> deserialize_expression_shared(proto::Expression const& proto)
{
  return std::shared_ptr<expression>(deserialize_expression(proto));
}

}  // namespace

std::unique_ptr<expression> deserialize_expression(proto::Expression const& proto)
{
  switch (proto.expr_case()) {
    case proto::Expression::kColumnReference: {
      auto const& msg = proto.column_reference();
      if (msg.column_name().empty()) {
        return std::make_unique<column_reference_expression>(msg.column_idx());
      }
      return std::make_unique<column_reference_expression>(msg.column_idx(), msg.column_name());
    }
    case proto::Expression::kLiteral: {
      auto const& msg = proto.literal();
      auto dt         = deserialize_data_type(msg.data_type());
      auto const& val = msg.value();

      switch (dt.id()) {
        case cudf::type_id::INT8: {
          int8_t v;
          std::memcpy(&v, val.data(), sizeof(v));
          return std::make_unique<literal_expression<int8_t>>(v, msg.is_null());
        }
        case cudf::type_id::INT32: {
          int32_t v;
          std::memcpy(&v, val.data(), sizeof(v));
          return std::make_unique<literal_expression<int32_t>>(v, msg.is_null());
        }
        case cudf::type_id::INT64: {
          int64_t v;
          std::memcpy(&v, val.data(), sizeof(v));
          return std::make_unique<literal_expression<int64_t>>(v, msg.is_null());
        }
        case cudf::type_id::FLOAT32: {
          float v;
          std::memcpy(&v, val.data(), sizeof(v));
          return std::make_unique<literal_expression<float>>(v, msg.is_null());
        }
        case cudf::type_id::FLOAT64: {
          double v;
          std::memcpy(&v, val.data(), sizeof(v));
          return std::make_unique<literal_expression<double>>(v, msg.is_null());
        }
        case cudf::type_id::STRING:
          return std::make_unique<literal_expression<std::string>>(val, msg.is_null());
        case cudf::type_id::TIMESTAMP_DAYS: {
          int32_t count;
          std::memcpy(&count, val.data(), sizeof(count));
          return std::make_unique<literal_expression<cudf::timestamp_D>>(
            cudf::timestamp_D{cudf::duration_D{count}}, msg.is_null());
        }
        default:
          throw std::logic_error("Unsupported literal data type for deserialization: " +
                                 std::to_string(static_cast<int>(dt.id())));
      }
    }
    case proto::Expression::kIsNull: {
      return std::make_unique<is_null_expression>(
        deserialize_expression_shared(proto.is_null().input()));
    }
    case proto::Expression::kCast: {
      auto const& msg = proto.cast();
      return std::make_unique<cast_expression>(deserialize_expression_shared(msg.input()),
                                               deserialize_data_type(msg.target_type()));
    }
    case proto::Expression::kBinaryOp: {
      auto const& msg = proto.binary_op();
      auto op         = static_cast<cudf::binary_operator>(msg.op());
      auto lhs        = deserialize_expression_shared(msg.left());
      auto rhs        = deserialize_expression_shared(msg.right());
      switch (op) {
        case cudf::binary_operator::ADD: return std::make_unique<add_expression>(lhs, rhs);
        case cudf::binary_operator::SUB: return std::make_unique<subtract_expression>(lhs, rhs);
        case cudf::binary_operator::MUL: return std::make_unique<multiply_expression>(lhs, rhs);
        case cudf::binary_operator::TRUE_DIV: return std::make_unique<divide_expression>(lhs, rhs);
        case cudf::binary_operator::NULL_LOGICAL_AND:
          return std::make_unique<logical_and_expression>(lhs, rhs);
        case cudf::binary_operator::NULL_LOGICAL_OR:
          return std::make_unique<logical_or_expression>(lhs, rhs);
        case cudf::binary_operator::EQUAL: return std::make_unique<equal_expression>(lhs, rhs);
        case cudf::binary_operator::NULL_EQUALS:
          return std::make_unique<nulls_equal_expression>(lhs, rhs);
        case cudf::binary_operator::NOT_EQUAL:
          return std::make_unique<not_equal_expression>(lhs, rhs);
        case cudf::binary_operator::LESS: return std::make_unique<less_expression>(lhs, rhs);
        case cudf::binary_operator::GREATER: return std::make_unique<greater_expression>(lhs, rhs);
        case cudf::binary_operator::LESS_EQUAL:
          return std::make_unique<less_equal_expression>(lhs, rhs);
        case cudf::binary_operator::GREATER_EQUAL:
          return std::make_unique<greater_equal_expression>(lhs, rhs);
        default:
          throw std::logic_error("Unsupported binary operator for deserialization: " +
                                 std::to_string(msg.op()));
      }
    }
    case proto::Expression::kUnaryOp: {
      auto const& msg = proto.unary_op();
      auto op         = static_cast<cudf::unary_operator>(msg.op());
      if (op == cudf::unary_operator::NOT) {
        return std::make_unique<not_expression>(deserialize_expression_shared(msg.input()));
      }
      throw std::logic_error("Unsupported unary operator for deserialization: " +
                             std::to_string(msg.op()));
    }
    case proto::Expression::kIfThenElse: {
      auto const& msg = proto.if_then_else();
      return std::make_unique<if_then_else_expression>(
        deserialize_expression_shared(msg.condition()),
        deserialize_expression_shared(msg.then_expr()),
        deserialize_expression_shared(msg.else_expr()));
    }
    case proto::Expression::kScalarFunction: {
      auto const& msg = proto.scalar_function();
      auto fn_kind    = static_cast<scalar_function_expression::function_kind>(msg.function_kind());

      std::vector<std::shared_ptr<expression>> args;
      for (auto const& arg : msg.arguments()) {
        args.push_back(deserialize_expression_shared(arg));
      }

      switch (fn_kind) {
        case scalar_function_expression::function_kind::datepart:
          return std::make_unique<datepart_expression>(
            std::move(args[0]),
            static_cast<cudf::datetime::datetime_component>(msg.datetime_component()));
        case scalar_function_expression::function_kind::like:
          return std::make_unique<like_expression>(
            std::move(args[0]), msg.pattern(), msg.escape_character(), msg.ignore_case());
        case scalar_function_expression::function_kind::round:
          return std::make_unique<round_expression>(std::move(args[0]), msg.decimal_places());
        case scalar_function_expression::function_kind::substr:
          return std::make_unique<substr_expression>(std::move(args[0]), msg.start(), msg.length());
        default:
          throw std::logic_error("Unknown scalar function kind: " +
                                 std::to_string(msg.function_kind()));
      }
    }
    case proto::Expression::kInPredicate: {
      auto const& msg = proto.in_predicate();
      std::vector<std::shared_ptr<expression>> needles;
      for (auto const& n : msg.needles()) {
        needles.push_back(deserialize_expression_shared(n));
      }
      return std::make_unique<in_predicate_expression>(
        std::move(needles), static_cast<cudf::size_type>(msg.relation_index()));
    }
    case proto::Expression::EXPR_NOT_SET: return nullptr;
    default:
      throw std::logic_error("Unknown expression type in protobuf: " +
                             std::to_string(proto.expr_case()));
  }
}

}  // namespace gqe::rpc
