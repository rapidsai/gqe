/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "join_predicate.hpp"

#include <gqe/executor/eval.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/cast.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/is_null.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/expression/utility.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace gqe::detail {

namespace {

[[nodiscard]] bool has_supported_ast_operator(
  std::variant<cudf::binary_operator, cudf::unary_operator> op)
{
  try {
    static_cast<void>(cudf_to_ast_operator(op));
    return true;
  } catch (std::logic_error const&) {
    return false;
  }
}

[[nodiscard]] bool is_supported_literal(expression const& predicate)
{
  return dynamic_cast<literal_expression<int32_t> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<int64_t> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<float> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<double> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<numeric::decimal32> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<numeric::decimal64> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<numeric::decimal128> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<std::string> const*>(&predicate) != nullptr ||
         dynamic_cast<literal_expression<cudf::timestamp_D> const*>(&predicate) != nullptr;
}

[[nodiscard]] bool is_supported_predicate(expression const& predicate)
{
  using expression_type             = expression::expression_type;
  auto const children               = predicate.children();
  auto const children_are_supported = [&] {
    return std::ranges::all_of(children, [](auto const* child) {
      return child != nullptr && is_supported_predicate(*child);
    });
  };

  switch (predicate.type()) {
    case expression_type::column_reference: return children.empty();
    case expression_type::literal: return children.empty() && is_supported_literal(predicate);
    case expression_type::binary_op: {
      auto const* binary = dynamic_cast<binary_op_expression const*>(&predicate);
      return binary != nullptr && children.size() == 2 &&
             has_supported_ast_operator(binary->binary_operator()) && children_are_supported();
    }
    case expression_type::unary_op: {
      auto const* unary = dynamic_cast<unary_op_expression const*>(&predicate);
      return unary != nullptr && children.size() == 1 &&
             has_supported_ast_operator(unary->unary_operator()) && children_are_supported();
    }
    case expression_type::is_null: return children.size() == 1 && children_are_supported();
    case expression_type::cast: {
      auto const* cast = dynamic_cast<cast_expression const*>(&predicate);
      return cast != nullptr && children.size() == 1 &&
             (cudf::is_integral(cast->out_type()) || cudf::is_floating_point(cast->out_type()) ||
              cudf::is_fixed_point(cast->out_type())) &&
             children_are_supported();
    }
    default: return false;
  }
}

/** @brief Builds an owned join AST, resolving references through optional side-local bindings. */
class predicate_rewriter : public expression_visitor {
 public:
  predicate_rewriter(cudf::size_type left_num_columns,
                     expression_input const* left,
                     expression_input const* right,
                     std::vector<join_predicate::owned_object>* objects,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
    : _left_num_columns(left_num_columns),
      _left(left),
      _right(right),
      _objects(objects),
      _stream(stream),
      _mr(mr)
  {
  }

  [[nodiscard]] cudf::ast::expression const* result() const noexcept { return _result; }

  void visit(column_reference_expression const* expression) override
  {
    auto const column_index = expression->column_idx();
    GQE_EXPECTS(column_index >= 0,
                "make_join_predicate: column reference must be non-negative",
                std::invalid_argument);
    auto const is_left       = column_index < _left_num_columns;
    auto const* input        = is_left ? _left : _right;
    auto const logical_index = is_left ? column_index : column_index - _left_num_columns;
    auto const physical_index =
      input == nullptr ? logical_index : input->physical_column_index(logical_index);
    retain(std::make_unique<cudf::ast::column_reference>(
      physical_index,
      is_left ? cudf::ast::table_reference::LEFT : cudf::ast::table_reference::RIGHT));
  }

  void visit(binary_op_expression const* expression) override
  {
    auto const children = expression->children();
    GQE_EXPECTS(children.size() == 2,
                "make_join_predicate: binary expression must have two children");
    auto const* lhs = rewrite(*children[0]);
    auto const* rhs = rewrite(*children[1]);
    retain(std::make_unique<cudf::ast::operation>(
      cudf_to_ast_operator(expression->binary_operator()), *lhs, *rhs));
  }

  void visit(unary_op_expression const* expression) override
  {
    auto const children = expression->children();
    GQE_EXPECTS(children.size() == 1, "make_join_predicate: unary expression must have one child");
    auto const* child = rewrite(*children[0]);
    retain(std::make_unique<cudf::ast::operation>(
      cudf_to_ast_operator(expression->unary_operator()), *child));
  }

  void visit(is_null_expression const* expression) override
  {
    auto const children = expression->children();
    GQE_EXPECTS(children.size() == 1,
                "make_join_predicate: IS NULL expression must have one child");
    auto const* child = rewrite(*children[0]);
    retain(std::make_unique<cudf::ast::operation>(cudf::ast::ast_operator::IS_NULL, *child));
  }

  void visit(cast_expression const* expression) override
  {
    auto const children = expression->children();
    GQE_EXPECTS(children.size() == 1, "make_join_predicate: cast expression must have one child");
    auto const* child = rewrite(*children[0]);

    auto const type = expression->out_type();
    if (cudf::is_integral(type)) {
      retain(
        std::make_unique<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_INT64, *child));
    } else if (cudf::is_floating_point(type) || cudf::is_fixed_point(type)) {
      retain(
        std::make_unique<cudf::ast::operation>(cudf::ast::ast_operator::CAST_TO_FLOAT64, *child));
    } else {
      throw std::logic_error("make_join_predicate: unsupported cast expression");
    }
  }

  void visit(literal_expression<int32_t> const* expression) override { retain_literal(expression); }
  void visit(literal_expression<int64_t> const* expression) override { retain_literal(expression); }
  void visit(literal_expression<float> const* expression) override { retain_literal(expression); }
  void visit(literal_expression<double> const* expression) override { retain_literal(expression); }
  void visit(literal_expression<numeric::decimal32> const* expression) override
  {
    retain_literal(expression);
  }
  void visit(literal_expression<numeric::decimal64> const* expression) override
  {
    retain_literal(expression);
  }
  void visit(literal_expression<numeric::decimal128> const* expression) override
  {
    retain_literal(expression);
  }
  void visit(literal_expression<std::string> const* expression) override
  {
    retain_literal(expression);
  }
  void visit(literal_expression<cudf::timestamp_D> const* expression) override
  {
    retain_literal(expression);
  }

 private:
  [[nodiscard]] cudf::ast::expression const* rewrite(expression const& expression)
  {
    predicate_rewriter child(_left_num_columns, _left, _right, _objects, _stream, _mr);
    expression.accept(child);
    GQE_EXPECTS(child.result() != nullptr,
                "make_join_predicate: expression visitor produced no AST node");
    return child.result();
  }

  void retain(std::unique_ptr<cudf::ast::expression> node)
  {
    _result = node.get();
    _objects->push_back(std::move(node));
  }

  template <typename T>
  void retain_literal(literal_expression<T> const* expression)
  {
    using scalar_type =
      std::conditional_t<std::is_same_v<T, std::string>,
                         cudf::string_scalar,
                         std::conditional_t<std::is_same_v<T, cudf::timestamp_D>,
                                            cudf::timestamp_scalar<cudf::timestamp_D>,
                                            std::conditional_t<cudf::is_fixed_point<T>(),
                                                               cudf::fixed_point_scalar<T>,
                                                               cudf::numeric_scalar<T>>>>;
    auto scalar =
      std::make_unique<scalar_type>(expression->value(), !expression->is_null(), _stream, _mr);
    auto node = std::make_unique<cudf::ast::literal>(*scalar);
    _result   = node.get();
    _objects->push_back(std::move(scalar));
    _objects->push_back(std::move(node));
  }

  cudf::size_type _left_num_columns;
  expression_input const* _left;  ///< Borrowed for conversion, or null for identity mapping.

  expression_input const* _right;  ///< Borrowed for conversion, or null for identity mapping.
  std::vector<join_predicate::owned_object>* _objects;
  cuda::stream_ref _stream;
  rmm::device_async_resource_ref _mr;
  cudf::ast::expression const* _result{};
};

}  // namespace

std::optional<join_predicate_columns> try_collect_join_predicate_columns(
  expression const& predicate, cudf::size_type left_width, cudf::size_type right_width)
{
  if (left_width < 0 || right_width < 0 ||
      left_width > std::numeric_limits<cudf::size_type>::max() - right_width) {
    return std::nullopt;
  }
  join_predicate_columns result;
  expression const* roots[] = {&predicate};
  for (auto index : collect_column_references(roots)) {
    if (index < 0 || index >= left_width + right_width) { return std::nullopt; }
    if (index < left_width) {
      result.left.push_back(index);
    } else {
      result.right.push_back(index - left_width);
    }
  }
  return result;
}

bool is_join_predicate_supported(expression const& predicate)
{
  return is_supported_predicate(predicate);
}

join_predicate::join_predicate(cudf::ast::expression const* root, std::vector<owned_object> objects)
  : _root{root}, _objects{std::move(objects)}
{
  GQE_EXPECTS(_root != nullptr, "join_predicate: root must not be null", std::invalid_argument);
}

cudf::ast::expression const& join_predicate::root() const
{
  GQE_EXPECTS(_root != nullptr, "join_predicate: root is null");
  return *_root;
}

join_predicate make_join_predicate(expression const& predicate,
                                   cudf::size_type left_num_columns,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  GQE_EXPECTS(left_num_columns >= 0,
              "make_join_predicate: left column count must be non-negative",
              std::invalid_argument);
  std::vector<join_predicate::owned_object> objects;
  predicate_rewriter rewriter(left_num_columns, nullptr, nullptr, &objects, stream, mr);
  predicate.accept(rewriter);
  GQE_EXPECTS(rewriter.result() != nullptr,
              "make_join_predicate: expression visitor produced no AST root");
  return join_predicate{rewriter.result(), std::move(objects)};
}

join_predicate make_join_predicate(expression const& predicate,
                                   cudf::size_type left_num_columns,
                                   expression_input const& left,
                                   expression_input const& right,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  std::vector<join_predicate::owned_object> objects;
  GQE_EXPECTS(left_num_columns >= 0,
              "make_join_predicate: left column count must be nonnegative",
              std::invalid_argument);
  predicate_rewriter rewriter(left_num_columns, &left, &right, &objects, stream, mr);
  predicate.accept(rewriter);
  GQE_EXPECTS(rewriter.result() != nullptr,
              "make_join_predicate: expression visitor produced no AST root");
  return join_predicate(rewriter.result(), std::move(objects));
}

}  // namespace gqe::detail
