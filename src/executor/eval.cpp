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

#include <gqe/executor/eval.hpp>

#include "masked_ast.hpp"

#include <gqe/executor/like.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

/**
 * @brief Integer power of 10 in the specified representation type.
 *
 * @tparam Rep Integer representation type for the result.
 * @param exponent Non-negative exponent.
 * @return 10^exponent as Rep.
 */
template <typename Rep = int64_t>
constexpr Rep int_pow10(int32_t exponent)
{
  Rep result = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

/**
 * @brief Whether a binary op must avoid the interpreted cuDF AST path for DECIMAL128.
 *
 * cuDF AST intermediates are stored in 8-byte slots, so DECIMAL128 arithmetic silently
 * produces zeros (see `DecimalArithmeticTypes` in cudf/cpp/tests/ast/transform_tests.cpp).
 * DECIMAL128 comparisons and other BOOL-producing ops remain AST-eligible.
 */
[[nodiscard]] constexpr bool is_decimal_arithmetic_op(cudf::binary_operator op)
{
  return op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
         op == cudf::binary_operator::MUL || op == cudf::binary_operator::DIV;
}

[[nodiscard]] constexpr bool decimal128_interpreted_ast_arithmetic_unsupported(
  cudf::binary_operator op, cudf::data_type const& result_dt)
{
  return result_dt.id() == cudf::type_id::DECIMAL128 && is_decimal_arithmetic_op(op);
}

/**
 * @brief Whether an operation reads fixed-point operand scales when consuming an intermediate.
 *
 * cuDF 26.06's interpreted AST corrupts the scale when reading a fixed-point intermediate.
 * Multiplication and division use only the representation and therefore do not expose this bug,
 * while addition, subtraction, and comparisons rescale their operands and do.
 */
[[nodiscard]] constexpr bool consumes_fixed_point_scale(cudf::binary_operator op)
{
  return op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
         op == cudf::binary_operator::EQUAL || op == cudf::binary_operator::NOT_EQUAL ||
         op == cudf::binary_operator::LESS || op == cudf::binary_operator::GREATER ||
         op == cudf::binary_operator::LESS_EQUAL || op == cudf::binary_operator::GREATER_EQUAL ||
         op == cudf::binary_operator::NULL_EQUALS;
}

// FIXME (breta): On a high-level, shouldn't we be passing the executor an "executable" plan? The
// scaling logic could already take place on the logical query plan. That would also mean that
// conditioning takes place once, instead of each time evaluate_expressions is called (i.e., for
// each partition).
class binary_op_data_conditioner {
 public:
  binary_op_data_conditioner(cudf::binary_operator op,
                             cudf::table_view const& table,
                             gqe::column_reference_expression const* lhs_ref,
                             gqe::expression_evaluator::evaluation_context const* lhs_context,
                             gqe::column_reference_expression const* rhs_ref,
                             gqe::expression_evaluator::evaluation_context const* rhs_context,
                             cudf::data_type const& result_data_type,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
    : _stream{stream}, _mr{mr}
  {
    cudf::data_type const lhs_type = set_input_and_get_type(LHS, table, lhs_ref, lhs_context);
    cudf::data_type const rhs_type = set_input_and_get_type(RHS, table, rhs_ref, rhs_context);

    _conditioned_binary_op = op;

    bool const has_fixed_point_inputs =
      cudf::is_fixed_point(lhs_type) || cudf::is_fixed_point(rhs_type);
    if (has_fixed_point_inputs) {
      // FIXED POINT NOTES:
      // 1) Binary ops require compatible decimal widths. Different scales are supported.
      // 2) cudf division
      //    a) TRUE_DIV is not allowed
      //    b) Dividend and divisor scales are subtracted for the result scale. This requires
      //       us to cast the dividend to a larger scale to account for the subtraction.
      // 3) cudf does not round. This must be done manually. (not implemented)

      // FIXME (breta): Should TRUE_DIV always be floating point?

      // TRUE_DIV is unsupported when producing fixed-point results.
      if (cudf::is_fixed_point(result_data_type) && op == cudf::binary_operator::TRUE_DIV) {
        throw std::logic_error("TRUE_DIV is not supported for fixed-point (decimal) types.");
      }

      int needed_lhs_scale = 0;
      // If dividing, calculate the scale for the dividend so that we promote to a large enough
      // decimal type. cudf computes: result_scale = lhs_scale - rhs_scale, so we need
      // lhs_cond_scale = result_scale + rhs_scale.
      if (_conditioned_binary_op == cudf::binary_operator::DIV) {
        needed_lhs_scale = result_data_type.scale() + rhs_type.scale();
      }

      // LHS and RSH can be different sizes or non-decimal, so we need to find a common decimal
      // type to promote to when conditioning.
      cudf::type_id const promotion_type =
        gqe::binary_op_decimal_promotion_type(needed_lhs_scale, lhs_type, rhs_type);
      cudf::type_id conditioned_type = promotion_type;
      if (cudf::is_fixed_point(result_data_type)) {
        conditioned_type = std::max(conditioned_type, result_data_type.id());
      }
      GQE_LOG_TRACE(
        "eval::binary_op_data_conditioner: op={} fixed-point conditioning enabled; lhs=({}), "
        "rhs=({}), result=({}), needed_lhs_scale={}, promotion_type_id={}, conditioned_type_id={}",
        static_cast<int>(op),
        gqe::utility::data_type_to_string(lhs_type),
        gqe::utility::data_type_to_string(rhs_type),
        gqe::utility::data_type_to_string(result_data_type),
        needed_lhs_scale,
        static_cast<int>(promotion_type),
        static_cast<int>(conditioned_type));

      condition_fixed_point_inputs(conditioned_type, result_data_type, rhs_type);
    }
  }

  [[nodiscard]] cudf::binary_operator binary_op() const noexcept { return _conditioned_binary_op; }

  [[nodiscard]] cudf::column_view const& lhs_col_view() const noexcept
  {
    return _conditioned_input[LHS].col_view;
  }

  [[nodiscard]] cudf::scalar const* lhs_scalar() const noexcept
  {
    return _conditioned_input[LHS].scalar;
  }

  [[nodiscard]] cudf::column_view const& rhs_col_view() const noexcept
  {
    return _conditioned_input[RHS].col_view;
  }

  [[nodiscard]] cudf::scalar const* rhs_scalar() const noexcept
  {
    return _conditioned_input[RHS].scalar;
  }

 private:
  cudf::data_type set_input_and_get_type(
    int input_idx,
    cudf::table_view const& table,
    gqe::column_reference_expression const* col_ref,
    gqe::expression_evaluator::evaluation_context const* eval_context) noexcept
  {
    if (col_ref) {
      _conditioned_input[input_idx].col_view = table.column(col_ref->column_idx());
      return _conditioned_input[input_idx].col_view.type();
    } else {
      _conditioned_input[input_idx].scalar = eval_context->cudf_scalar.value().get();
      return _conditioned_input[input_idx].scalar->type();
    }
  }

  void condition_fixed_point_inputs(cudf::type_id promotion_type,
                                    cudf::data_type const& result_data_type,
                                    cudf::data_type const& rhs_type)
  {
    // Get the LHS original scale before conditioning.
    auto const& lhs_input  = _conditioned_input[LHS];
    int32_t lhs_orig_scale = !lhs_input.col_view.is_empty() ? lhs_input.col_view.type().scale()
                                                            : lhs_input.scalar->type().scale();

    for (int input_idx = 0; input_idx < INPUT_COUNT; ++input_idx) {
      // The fixed-point scale delta for the conditioned input.
      // condition_fixed_point_input will add the input's original scale to this value.
      int64_t scale_delta = 0;
      // When applying the scale to an original input value, cudf::fixed_point_scalar will shift the
      // decimal point. For example, {2, -3} will become 0.002. Since, we want it to be 2.000, we
      // have to multiply the value such that we pass in {2000, -3}. The exponent is used instead of
      // a pre-computed multiplier so that the multiplication can be done in the target rep type
      // (e.g., __int128_t for DECIMAL128), avoiding integer overflow.
      int32_t multiplier_exp = 0;

      // If dividing, condition the dividend so that cudf produces the correct result scale.
      // cudf computes: result_scale = lhs_cond_scale - rhs_cond_scale
      // The conditioned RHS preserves its original scale, so rhs_cond_scale = rhs_original_scale.
      // We need: lhs_cond_scale = result_scale + rhs_original_scale
      // Since condition_fixed_point_input adds lhs_original_scale:
      //   lhs_cond_scale = scale_delta + lhs_original_scale
      //   scale_delta = result_scale + rhs_original_scale - lhs_original_scale
      if (input_idx == LHS && _conditioned_binary_op == cudf::binary_operator::DIV) {
        scale_delta    = result_data_type.scale() + rhs_type.scale() - lhs_orig_scale;
        multiplier_exp = static_cast<int32_t>(-scale_delta);
      }

      condition_fixed_point_input(input_idx, scale_delta, multiplier_exp, promotion_type);
    }
  }

  void condition_fixed_point_input(int input_idx,
                                   int64_t scale_delta,
                                   int32_t multiplier_exp,
                                   cudf::type_id promotion_type)
  {
    // TODO: Consider checks to avoid unneeded conditioning.
    auto& cond_input = _conditioned_input[input_idx];
    if (!cond_input.col_view.is_empty()) {
      auto const original_type = cond_input.col_view.type();
      auto const target_scale  = scale_delta + cond_input.col_view.type().scale();
      GQE_LOG_TRACE(
        "eval::condition_fixed_point_input: input={} kind=column original=({}) -> "
        "target=(type_id={}, scale={}), "
        "scale_delta={}, multiplier_exp={}",
        input_idx,
        gqe::utility::data_type_to_string(original_type),
        static_cast<int>(promotion_type),
        target_scale,
        scale_delta,
        multiplier_exp);
      cond_input.conditioned_col = cudf::cast(
        cond_input.col_view, cudf::data_type(promotion_type, target_scale), _stream, _mr);
      cond_input.col_view = cond_input.conditioned_col->view();
    } else {
      auto const original_type = cond_input.scalar->type();
      auto const target_scale  = scale_delta + cond_input.scalar->type().scale();
      GQE_LOG_TRACE(
        "eval::condition_fixed_point_input: input={} kind=scalar original=({}) -> "
        "target=(type_id={}, scale={}), "
        "scale_delta={}, multiplier_exp={}",
        input_idx,
        gqe::utility::data_type_to_string(original_type),
        static_cast<int>(promotion_type),
        target_scale,
        scale_delta,
        multiplier_exp);
      cond_input.conditioned_scalar =
        condition_scalar(cond_input.scalar, promotion_type, multiplier_exp, target_scale);
      cond_input.scalar = cond_input.conditioned_scalar.get();
    }
  }

  std::unique_ptr<cudf::scalar> condition_scalar(cudf::scalar* src,
                                                 cudf::type_id dst_type_id,
                                                 int32_t multiplier_exp,
                                                 int64_t scale)
  {
    auto const is_valid = src->is_valid(_stream);
    if (cudf::is_numeric(src->type())) {
      switch (src->type().id()) {
        case cudf::type_id::INT8: {
          auto val = static_cast<cudf::numeric_scalar<int8_t>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::INT16: {
          auto val = static_cast<cudf::numeric_scalar<int16_t>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::INT32: {
          auto val = static_cast<cudf::numeric_scalar<int32_t>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::INT64: {
          auto val = static_cast<cudf::numeric_scalar<int64_t>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::FLOAT32: {
          auto val = static_cast<cudf::numeric_scalar<float>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::FLOAT64: {
          auto val = static_cast<cudf::numeric_scalar<double>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        default: throw std::runtime_error("src is not a numeric scalar type.");
      }
    } else if (cudf::is_fixed_point(src->type())) {
      switch (src->type().id()) {
        case cudf::type_id::DECIMAL32: {
          auto val =
            static_cast<cudf::fixed_point_scalar<numeric::decimal32>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::DECIMAL64: {
          auto val =
            static_cast<cudf::fixed_point_scalar<numeric::decimal64>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        case cudf::type_id::DECIMAL128: {
          auto val =
            static_cast<cudf::fixed_point_scalar<numeric::decimal128>*>(src)->value(_stream);
          return created_conditioned_scalar(val, dst_type_id, multiplier_exp, scale, is_valid);
        }
        default: throw std::runtime_error("src is not a fixed point scalar type.");
      }
    } else {
      throw std::runtime_error("src is an unsupported scalar type.");
    }
  }

  template <typename T, typename Rep>
  [[nodiscard]] T scale_conditioned_value(T const& value, int32_t multiplier_exp)
  {
    if (multiplier_exp > 0) {
      return value * int_pow10<Rep>(multiplier_exp);
    } else if (multiplier_exp < 0) {
      return value / int_pow10<Rep>(-multiplier_exp);
    }
    return value;
  }

  template <typename T>
  std::unique_ptr<cudf::scalar> created_conditioned_scalar(
    T const& val, cudf::type_id dst_type_id, int32_t multiplier_exp, int64_t scale, bool is_valid)
  {
    if (dst_type_id == cudf::type_id::DECIMAL32) {
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        scale_conditioned_value<T, int32_t>(val, multiplier_exp),
        numeric::scale_type(scale),
        is_valid,
        _stream,
        _mr);
    } else if (dst_type_id == cudf::type_id::DECIMAL64) {
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        scale_conditioned_value<T, int64_t>(val, multiplier_exp),
        numeric::scale_type(scale),
        is_valid,
        _stream,
        _mr);
    } else if (dst_type_id == cudf::type_id::DECIMAL128) {
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        scale_conditioned_value<T, __int128_t>(val, multiplier_exp),
        numeric::scale_type(scale),
        is_valid,
        _stream,
        _mr);
    } else {
      throw std::runtime_error("dst_type_id must be a decimal type.");
    }
  }

  enum { LHS, RHS, INPUT_COUNT };

  struct _conditioned_input {
    // Placeholders for potential conditioned vales
    std::unique_ptr<cudf::column> conditioned_col;
    std::unique_ptr<cudf::scalar> conditioned_scalar;
    cudf::column_view col_view;
    cudf::scalar* scalar = nullptr;
  } _conditioned_input[INPUT_COUNT];

  cuda::stream_ref _stream;
  rmm::device_async_resource_ref _mr;
  cudf::binary_operator _conditioned_binary_op;
};

}  // namespace

namespace gqe {

[[nodiscard]] cudf::ast::ast_operator cudf_to_ast_operator(
  std::variant<cudf::binary_operator, cudf::unary_operator> op)
{
  static const std::unordered_map<std::variant<cudf::binary_operator, cudf::unary_operator>,
                                  cudf::ast::ast_operator>
    _operator_map = {
      {cudf::binary_operator::ADD, cudf::ast::ast_operator::ADD},
      {cudf::binary_operator::SUB, cudf::ast::ast_operator::SUB},
      {cudf::binary_operator::MUL, cudf::ast::ast_operator::MUL},
      {cudf::binary_operator::TRUE_DIV, cudf::ast::ast_operator::TRUE_DIV},
      {cudf::binary_operator::DIV, cudf::ast::ast_operator::DIV},
      {cudf::binary_operator::NULL_LOGICAL_AND, cudf::ast::ast_operator::NULL_LOGICAL_AND},
      {cudf::binary_operator::NULL_LOGICAL_OR, cudf::ast::ast_operator::NULL_LOGICAL_OR},
      {cudf::binary_operator::EQUAL, cudf::ast::ast_operator::EQUAL},
      {cudf::binary_operator::NOT_EQUAL, cudf::ast::ast_operator::NOT_EQUAL},
      {cudf::binary_operator::LESS, cudf::ast::ast_operator::LESS},
      {cudf::binary_operator::GREATER, cudf::ast::ast_operator::GREATER},
      {cudf::binary_operator::LESS_EQUAL, cudf::ast::ast_operator::LESS_EQUAL},
      {cudf::binary_operator::GREATER_EQUAL, cudf::ast::ast_operator::GREATER_EQUAL},
      {cudf::binary_operator::NULL_EQUALS, cudf::ast::ast_operator::NULL_EQUAL},
      {cudf::unary_operator::NOT,
       cudf::ast::ast_operator::NOT}};  ///> Emum mapper between cudf::{unary|binary}_op and
                                        /// cudf::ast::ast_operator.

  auto search = _operator_map.find(op);
  if (search == _operator_map.end()) {
    throw std::logic_error(
      "Unable to convert cudf::{unary|binary}_operator to cudf::ast::ast_operator");
  }

  return search->second;
}

[[nodiscard]] std::optional<cudf::ast::ast_operator> cast_to_ast_operator(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT64: return cudf::ast::ast_operator::CAST_TO_INT64;
    case cudf::type_id::UINT64: return cudf::ast::ast_operator::CAST_TO_UINT64;
    case cudf::type_id::FLOAT64: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
    default: return std::nullopt;
  }
}

std::string expression_evaluator::evaluation_context::to_string() const noexcept
{
  std::string s = "evaluation_context(" + this->gqe_expression->to_string() + ", ";
  s += (this->cudf_ast_expression.has_value()) ? "has" : "no";
  s += " cudf::ast equivalent, ";
  s += (this->column_idx.has_value())
         ? "column_offset = " + std::to_string(this->column_idx.value()) + ")"
         : "no task)";
  return s;
}

expression_binding::expression_binding(std::span<cudf::size_type const> physical_to_logical)
{
  constexpr auto max_columns = std::numeric_limits<cudf::size_type>::max();
  GQE_EXPECTS(physical_to_logical.size() <= static_cast<std::size_t>(max_columns),
              "expression_binding: too many physical columns",
              std::invalid_argument);
  cudf::size_type required_width = 0;
  for (auto const logical_index : physical_to_logical) {
    GQE_EXPECTS(logical_index >= 0 && logical_index < max_columns,
                "expression_binding: logical index must fit in a cuDF schema",
                std::invalid_argument);
    required_width = std::max(required_width, logical_index + 1);
  }
  _physical_column_count = static_cast<cudf::size_type>(physical_to_logical.size());
  constexpr auto missing = cudf::size_type{-1};
  _logical_to_physical.assign(static_cast<std::size_t>(required_width), missing);
  for (cudf::size_type physical_index = 0; physical_index < _physical_column_count;
       ++physical_index) {
    auto const logical_index = physical_to_logical[physical_index];
    GQE_EXPECTS(_logical_to_physical[logical_index] == missing,
                "expression_binding: mapped logical indices must be unique",
                std::invalid_argument);
    _logical_to_physical[logical_index] = physical_index;
  }
}

cudf::size_type expression_binding::physical_column_index(cudf::size_type logical_index) const
{
  GQE_EXPECTS(logical_index >= 0 && logical_index < required_logical_column_count() &&
                _logical_to_physical[logical_index] >= 0,
              "expression_binding: logical column reference is absent from the physical table",
              std::out_of_range);
  return _logical_to_physical[logical_index];
}

expression_input::expression_input(cudf::table_view table) noexcept
  : _table{table}, _logical_row_count{table.num_rows()}, _binding{nullptr}
{
}

expression_input::expression_input(cudf::table_view table,
                                   expression_binding const& binding,
                                   cudf::size_type logical_row_count)
  : _table{table}, _logical_row_count{logical_row_count}, _binding{&binding}
{
  GQE_EXPECTS(
    logical_row_count >= 0 && (table.num_columns() == 0 || logical_row_count == table.num_rows()),
    "expression_input: logical row count must match table geometry",
    std::invalid_argument);
  GQE_EXPECTS(binding.physical_column_count() == table.num_columns(),
              "expression_input: mapping size must equal the table column count",
              std::invalid_argument);
}

cudf::size_type expression_input::physical_column_index(cudf::size_type logical_index) const
{
  if (_binding != nullptr) { return _binding->physical_column_index(logical_index); }
  GQE_EXPECTS(logical_index >= 0 && logical_index < _table.num_columns(),
              "expression_input: logical column reference is out of range",
              std::out_of_range);
  return logical_index;
}

expression_evaluator::expression_evaluator(expression_input const& input,
                                           expression const* root_expression,
                                           bool use_ast_jit,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref result_mr)
  : _input{input},
    _table{input.table()},
    _root_expression{root_expression},
    _use_ast_jit{use_ast_jit},
    _stream{stream},
    _result_mr{result_mr},
    _temporary_mr{result_mr},
    _root_context{nullptr},
    _next_task{_table.num_columns()}
{
  if (root_expression == nullptr) {
    throw std::invalid_argument("expression_evaluator: root expression must not be null");
  }

  // _column_types is an empty member vector.
  _column_types.reserve(_table.num_columns());
  for (auto const& col : _table) {
    _column_types.push_back(col.type());
  }

  // traverse AST recursively starting at the root node
  _root_expression->accept(*this);

  // mark the root expression as to be evaluated as well
  auto& root_context = this->find_context(_root_expression);
  this->dispatch_task(root_context);
  _root_context = &root_context;
}

[[nodiscard]] std::pair<cudf::column_view, std::unique_ptr<cudf::column>>
expression_evaluator::evaluate(bool use_like_shift_and,
                               std::optional<cudf::table_view> input_override) const
{
  auto table = input_override.value_or(_table);
  if (input_override &&
      (table.num_rows() != _table.num_rows() || !cudf::have_same_types(table, _table))) {
    throw std::invalid_argument(
      "expression_evaluator: override must preserve input shape and types");
  }
  if (_root_expression->type() == expression::expression_type::column_reference) {
    auto const index = _root_context->column_idx;
    if (!index || *index < 0 || *index >= table.num_columns()) {
      throw std::logic_error("expression_evaluator: resolved reference is not a table column");
    }
    return {table.column(*index), nullptr};
  }
  if (_input.row_count() == 0) {
    auto output = cudf::make_empty_column(_root_context->data_type());
    auto view   = output->view();
    return {view, std::move(output)};
  }
  auto column_types = this->_column_types;

  // storage for intermediate result columns
  std::vector<std::unique_ptr<cudf::column>> intermediate_results;
  intermediate_results.reserve(_tasks.size());

  // helper function to append any newly generated column to the table
  auto append_result =
    [&table, &column_types, &intermediate_results](std::unique_ptr<cudf::column> result) {
      // All columns in a table_view must have the same row count, and a zero-column table_view
      // cannot represent nonzero row geometry. Reference-free expressions materialize their first
      // result from expression_input::row_count(), so use that column to establish the physical
      // table geometry instead of concatenating it with the empty, zero-row view.
      table = table.num_columns() == 0
                ? cudf::table_view{{result->view()}}
                : cudf::table_view({table, cudf::table_view{{result->view()}}});
      column_types.push_back(result->type());
      intermediate_results.push_back(std::move(result));
    };

  for (auto context : _tasks) {
    auto const output_mr = context == _root_context ? _result_mr : _temporary_mr;
    // cudf::compute_column only supports fixed-width output columns.
    // A zero-column table_view cannot carry non-zero row geometry. Ensure the first task in a
    // reference-free expression uses the fallback path, which expands its scalar to the logical
    // row count. Subsequent tasks can then use the materialized column's geometry.
    bool const has_input_geometry =
      table.num_columns() > 0 || table.num_rows() == _input.row_count();
    bool const can_use_cudf_ast = context->cudf_ast_expression.has_value() &&
                                  cudf::is_fixed_width(context->data_type()) && has_input_geometry;

    if (can_use_cudf_ast) {
      // evaluate the the `cudf::ast` (sub-)expression using `cudf::compute_column`
      if (_use_ast_jit) {
        append_result(cudf::compute_column_jit(
          table, *context->cudf_ast_expression.value(), _stream, output_mr));
      } else {
        append_result(
          cudf::compute_column(table, *context->cudf_ast_expression.value(), _stream, output_mr));
      }
    } else {
      // fallback evaluation method
      auto const expression = context->gqe_expression.get();
      auto const children   = expression->children();

      switch (expression->type()) {
        case gqe::expression::expression_type::column_reference: {
          throw std::logic_error("Evaluating " + expression->to_string() + " is not supported");
          // TODO either implement a deep copy of the input column to the output table
          // or share a reference-counted view.
          // With this, we can remove the WAR in the evaluate_expressions function
          // for when the input expressions is just a column reference
          break;
        }
        case gqe::expression::expression_type::literal: {
          // expand literal to a column
          append_result(cudf::make_column_from_scalar(
            *context->cudf_scalar.value().get(), _input.row_count(), _stream, output_mr));
          break;
        }
        case gqe::expression::expression_type::binary_op: {
          auto const binop = dynamic_cast<gqe::binary_op_expression const*>(expression);

          auto const lhs_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);
          auto const rhs_ref = dynamic_cast<gqe::column_reference_expression const*>(children[1]);

          auto const& lhs_context = context->child_contexts[0];
          auto const& rhs_context = context->child_contexts[1];

          // children must either be a column reference or literal expression
          if (!lhs_ref && children[0]->type() != expression::expression_type::literal) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " in the fallback path requires an LHS child which is either a "
                                   "column reference or literal expression but is " +
                                   children[0]->to_string() + ".");
          }

          if (!rhs_ref && children[1]->type() != expression::expression_type::literal) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " in the fallback path requires an LHS child which is either a "
                                   "column reference or literal expression but is " +
                                   children[1]->to_string() + ".");
          }

          auto const binary_op        = binop->binary_operator();
          auto const result_data_type = binop->data_type(column_types);

          binary_op_data_conditioner data_conditioner(binary_op,
                                                      table,
                                                      lhs_ref,
                                                      lhs_context,
                                                      rhs_ref,
                                                      rhs_context,
                                                      result_data_type,
                                                      _stream,
                                                      _temporary_mr);

          // If the expression is a binary op we can evaluate it using `cudf::binary_operation`.
          // Children can only be of type column_reference or literal -> 4 combinations.
          if (lhs_ref && rhs_ref) {
            append_result(cudf::binary_operation(data_conditioner.lhs_col_view(),
                                                 data_conditioner.rhs_col_view(),
                                                 data_conditioner.binary_op(),
                                                 result_data_type,
                                                 _stream,
                                                 output_mr));
          } else if (lhs_ref && !rhs_ref) {
            append_result(cudf::binary_operation(data_conditioner.lhs_col_view(),
                                                 *data_conditioner.rhs_scalar(),
                                                 data_conditioner.binary_op(),
                                                 result_data_type,
                                                 _stream,
                                                 output_mr));
          } else if (!lhs_ref && rhs_ref) {
            append_result(cudf::binary_operation(*data_conditioner.lhs_scalar(),
                                                 data_conditioner.rhs_col_view(),
                                                 data_conditioner.binary_op(),
                                                 result_data_type,
                                                 _stream,
                                                 output_mr));
          } else if (!lhs_ref && !rhs_ref) {
            throw std::logic_error("Evaluating " + binop->to_string() +
                                   " on two scalar inputs in the fallback path is not supported");
          }
          break;
        }
        case gqe::expression::expression_type::unary_op: {
          auto const op      = dynamic_cast<gqe::unary_op_expression const*>(expression);
          auto const col_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          if (!col_ref) {
            throw std::logic_error(
              "Input of `cudf::unary_operation` must be a column reference but is " +
              children[0]->to_string());
          }

          append_result(cudf::unary_operation(
            table.column(col_ref->column_idx()), op->unary_operator(), _stream, output_mr));
          break;
        }
        case gqe::expression::expression_type::if_then_else: {
          auto const if_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          if (!if_ref) {
            throw std::logic_error(
              "Input predicate of `cudf::copy_if_else` must be a column reference but is " +
              children[0]->to_string());
          }

          auto const then_ref = dynamic_cast<gqe::column_reference_expression const*>(children[1]);
          auto const else_ref = dynamic_cast<gqe::column_reference_expression const*>(children[2]);

          auto const& then_context = context->child_contexts[1];
          auto const& else_context = context->child_contexts[2];

          // `cudf::copy_if_else` supports THEN and ELSE children to be either
          // of type column_reference or literal -> 4 combinations.
          if (then_ref && else_ref) {
            append_result(cudf::copy_if_else(table.column(then_ref->column_idx()),
                                             table.column(else_ref->column_idx()),
                                             table.column(if_ref->column_idx()),
                                             _stream,
                                             output_mr));
          }

          if (!then_ref && else_ref) {
            append_result(cudf::copy_if_else(*(then_context->cudf_scalar.value().get()),
                                             table.column(else_ref->column_idx()),
                                             table.column(if_ref->column_idx()),
                                             _stream,
                                             output_mr));
          }

          if (then_ref && !else_ref) {
            append_result(cudf::copy_if_else(table.column(then_ref->column_idx()),
                                             *(else_context->cudf_scalar.value().get()),
                                             table.column(if_ref->column_idx()),
                                             _stream,
                                             output_mr));
          }

          if (!then_ref && !else_ref) {
            append_result(cudf::copy_if_else(*(then_context->cudf_scalar.value().get()),
                                             *(else_context->cudf_scalar.value().get()),
                                             table.column(if_ref->column_idx()),
                                             _stream,
                                             output_mr));
          }
          break;
        }
        case gqe::expression::expression_type::cast: {
          auto const child_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);
          GQE_EXPECTS(
            child_ref != nullptr,
            "Input of `cudf::cast` must be a column reference but is " + children[0]->to_string());
          auto const out_type = dynamic_cast<gqe::cast_expression*>(expression)->out_type();

          append_result(
            cudf::cast(table.column(child_ref->column_idx()), out_type, _stream, output_mr));
          break;
        }
        case gqe::expression::expression_type::scalar_function: {
          auto const scalar_func = dynamic_cast<gqe::scalar_function_expression*>(expression);
          auto const col_ref = dynamic_cast<gqe::column_reference_expression const*>(children[0]);

          // Scalar function expressions are always evaluated in the fallback path.
          // Thus, the child expression must always be a column reference.
          if (!col_ref) {
            throw std::logic_error(
              "Input of `gqe::scalar_function_expression` must be a column reference but is " +
              children[0]->to_string());
          }

          switch (scalar_func->fn_kind()) {
            case gqe::scalar_function_expression::function_kind::like: {
              auto const like_expr = dynamic_cast<gqe::like_expression*>(scalar_func);

              if (like_expr->ignore_case()) {
                throw std::logic_error("Evaluating ILIKE expression not implemented");
              }

              cudf::string_scalar const escape_char{
                like_expr->escape_character(), true, _stream, _temporary_mr};

              if (use_like_shift_and) {
                // if filter_use_like_shift_and is true,
                // we use the like_shift_and kernel for better performance
                append_result(gqe::like(table.column(col_ref->column_idx()),
                                        like_expr->pattern(),
                                        escape_char,
                                        _stream,
                                        output_mr));
              } else {
                append_result(cudf::strings::like(table.column(col_ref->column_idx()),
                                                  std::string_view{like_expr->pattern()},
                                                  std::string_view{like_expr->escape_character()},
                                                  _stream,
                                                  output_mr));
              }
              break;
            }
            case gqe::scalar_function_expression::function_kind::substr: {
              auto const substr_expr = dynamic_cast<gqe::substr_expression*>(scalar_func);

              auto const start = std::optional{substr_expr->start()};
              auto const end   = std::optional{substr_expr->start() + substr_expr->length()};
              auto const step  = std::optional<cudf::size_type>{1};

              append_result(cudf::strings::slice_strings(
                table.column(col_ref->column_idx()), start, end, step, _stream, output_mr));
              break;
            }
            case gqe::scalar_function_expression::function_kind::datepart: {
              auto const dp_expr = dynamic_cast<gqe::datepart_expression*>(scalar_func);

              append_result(cudf::datetime::extract_datetime_component(
                table.column(col_ref->column_idx()), dp_expr->component(), _stream, output_mr));
              break;
            }
            default:
              throw std::logic_error("Unable to evaluate `scalar_function_expression` " +
                                     expression->to_string());
          }
          break;
        }
        default:
          throw std::logic_error("Cannot evaluate expression in the fallback path: " +
                                 expression->to_string());
      }
    }
  }

  // cuDF's AST module might evaluate expression to a different type than GQE expression's data
  // type. For example, the AST module might evaluate an expression to int32_t, while GQE promotes
  // the result to int64_t. In such case, we cast the result to GQE expression's data type.
  auto const root_search = this->_evaluation_contexts.find(_root_expression);
  if (root_search == this->_evaluation_contexts.end()) {
    throw std::logic_error("Unable to find evaluation context for root expression " +
                           _root_expression->to_string());
  }
  auto const expected_type = root_search->second.data_type();
  if (intermediate_results.back()->type() != expected_type) {
    append_result(
      cudf::cast(intermediate_results.back()->view(), expected_type, _stream, _result_mr));
  }

  return std::make_pair(intermediate_results.back()->view(),
                        std::move(intermediate_results.back()));
}

std::unique_ptr<cudf::column> expression_evaluator::try_evaluate_masked_filter(
  cudf::column_view selection) const
{
  return try_evaluate_masked(selection, masked_evaluation_mode::filter);
}

std::unique_ptr<cudf::column> expression_evaluator::try_evaluate_masked_projection(
  cudf::column_view selection) const
{
  return try_evaluate_masked(selection, masked_evaluation_mode::projection);
}

std::unique_ptr<cudf::column> expression_evaluator::try_evaluate_masked(
  cudf::column_view selection, masked_evaluation_mode mode) const
{
  if (_use_ast_jit || _tasks.size() != 1 || !_root_context->cudf_ast_expression ||
      _root_expression->type() == expression::expression_type::column_reference ||
      (mode == masked_evaluation_mode::filter &&
       _root_context->data_type().id() != cudf::type_id::BOOL8) ||
      !cudf::is_fixed_width(_root_context->data_type()) || _table.num_columns() == 0)
    return nullptr;
  if (mode == masked_evaluation_mode::filter) {
    return detail::masked_filter_ast(_table,
                                     **_root_context->cudf_ast_expression,
                                     selection,
                                     _root_context->data_type(),
                                     _stream,
                                     _result_mr);
  }
  return detail::masked_projection_ast(_table,
                                       **_root_context->cudf_ast_expression,
                                       selection,
                                       _root_context->data_type(),
                                       _stream,
                                       _result_mr);
}

void expression_evaluator::visit(column_reference_expression const* expression)
{
  auto const column_id = _input.physical_column_index(expression->column_idx());
  auto [context, is_new] =
    this->emplace_context(expression, std::make_unique<column_reference_expression>(column_id));
  if (is_new) {
    context.cudf_ast_expression = std::make_unique<cudf::ast::column_reference>(column_id);
    context.column_idx          = column_id;
    context.result_data_type    = this->_column_types[column_id];
  }
}

void expression_evaluator::visit(literal_expression<int8_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<int32_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<int64_t> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<float> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<double> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<numeric::decimal32> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<numeric::decimal64> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<numeric::decimal128> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<std::string> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(literal_expression<cudf::timestamp_D> const* expression)
{
  this->create_literal_context(expression);
}

void expression_evaluator::visit(unary_op_expression const* expression)
{
  // Emplace context in the context map
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const child = expression->_children[0];
    // compute AST for the child expression
    child->accept(*this);

    // Store a reference to the child context in the parent context
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);

    bool const child_is_column = child_context.column_idx.has_value();
    bool const child_has_ast   = child_context.cudf_ast_expression.has_value();
    // Only `not_expression` derives from `unary_op_expression` today and its data_type
    // is BOOL8, but read it from the raw expression rather than hardcoding to keep this
    // open to future unary ops. Safe because comparison/logical ops don't recurse.
    context.result_data_type = expression->data_type(this->_column_types);
    // Gate on the child actually having an AST-usable representation: subqueries,
    // scalar functions, etc. populate only cudf_scalar / require a materialized
    // column.
    bool const can_use_cudf_ast =
      cudf::is_fixed_width(context.result_data_type) && (child_is_column || child_has_ast);

    if (can_use_cudf_ast) {
      auto const op = cudf_to_ast_operator(expression->unary_operator());

      if (child_is_column) {
        auto const child_column_idx = child_context.column_idx.value();
        context.gqe_expression->_children[0] =
          std::make_shared<column_reference_expression>(child_column_idx);
        auto new_child = std::make_unique<cudf::ast::column_reference>(child_column_idx);
        context.cudf_ast_expression = std::make_unique<cudf::ast::operation>(op, *new_child);
        context.cudf_ast_dependencies.emplace_back(std::move(new_child));
      } else {
        context.gqe_expression->_children[0] = child_context.gqe_expression;
        context.cudf_ast_expression =
          std::make_unique<cudf::ast::operation>(op, *child_context.cudf_ast_expression.value());
      }
    } else {
      auto const child_column_idx = this->dispatch_task(child_context);
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      this->dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(is_null_expression const* expression)
{
  // Emplace context in the context map
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const child = expression->_children[0];
    // compute AST for the child expression
    child->accept(*this);

    // Store a reference to the child context in the parent context
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);

    bool const child_is_column = child_context.column_idx.has_value();
    bool const child_has_ast   = child_context.cudf_ast_expression.has_value();
    auto const op              = cudf::ast::ast_operator::IS_NULL;
    context.result_data_type   = cudf::data_type(cudf::type_id::BOOL8);

    // IS NULL expression always returns a BOOL8, which is fixed width so we can use cudf ast
    // same logic as in visit(unary_op_expression const*)
    if (child_is_column) {
      auto const child_column_idx = child_context.column_idx.value();
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      auto new_child              = std::make_unique<cudf::ast::column_reference>(child_column_idx);
      context.cudf_ast_expression = std::make_unique<cudf::ast::operation>(op, *new_child);
      context.cudf_ast_dependencies.emplace_back(std::move(new_child));
    } else if (child_has_ast) {
      context.gqe_expression->_children[0] = child_context.gqe_expression;
      context.cudf_ast_expression =
        std::make_unique<cudf::ast::operation>(op, *child_context.cudf_ast_expression.value());
    } else {
      // Child produced only a cudf_scalar (e.g. a scalar_function result). Materialize it
      // as a column and take the column-ref branch so IS NULL still evaluates via the AST
      // path.
      auto const child_column_idx = this->dispatch_task(child_context);
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      auto new_child              = std::make_unique<cudf::ast::column_reference>(child_column_idx);
      context.cudf_ast_expression = std::make_unique<cudf::ast::operation>(op, *new_child);
      context.cudf_ast_dependencies.emplace_back(std::move(new_child));
    }
  }
}

void expression_evaluator::visit(binary_op_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const lhs = expression->_children[0];
    auto const rhs = expression->_children[1];
    // compute AST for the child expressions
    lhs->accept(*this);
    rhs->accept(*this);

    auto& lhs_context = this->find_context(lhs.get());
    auto& rhs_context = this->find_context(rhs.get());

    context.child_contexts.emplace_back(&lhs_context);
    context.child_contexts.emplace_back(&rhs_context);

    auto const lhs_dt   = lhs_context.data_type();
    auto const rhs_dt   = rhs_context.data_type();
    auto const op       = expression->binary_operator();
    bool const is_arith = op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
                          op == cudf::binary_operator::MUL || op == cudf::binary_operator::DIV;
    auto const result_dt     = is_arith ? gqe::arithmetic_output_type(op, lhs_dt, rhs_dt)
                                        : expression->data_type(this->_column_types);
    context.result_data_type = result_dt;
    // Force the fallback path for fixed-point DIV. cuDF AST decimal division applies the
    // natural rule `result_scale = lhs_scale - rhs_scale` (e.g. decimal64:-4 / decimal64:-4
    // → decimal64:0), which is integer-division semantics on the underlying rep and yields
    // 0 for ratios with magnitude < 1. The Spark-style wider output scale that GQE plans
    // expect (e.g. decimal128:-20) is only produced by `binary_op_data_conditioner`, which
    // pre-widens the dividend so that cuDF's division leaves the result at the wider scale.
    bool const fixed_point_width_compatible =
      cudf::is_fixed_point(lhs_dt) && cudf::is_fixed_point(rhs_dt) && lhs_dt.id() == rhs_dt.id();
    bool const needs_fixed_point_conditioning =
      cudf::is_fixed_point(result_dt) &&
      (op == cudf::binary_operator::DIV || !cudf::is_fixed_point(lhs_dt) ||
       !cudf::is_fixed_point(rhs_dt) || result_dt.id() != lhs_dt.id() ||
       result_dt.id() != rhs_dt.id());
    bool const exact_type_match = lhs_dt == rhs_dt;
    bool const decimal128_interpreted_ast_unsupported =
      !_use_ast_jit && decimal128_interpreted_ast_arithmetic_unsupported(op, result_dt);
    auto const is_fixed_point_ast_intermediate = [](gqe::expression const& child,
                                                    evaluation_context const& child_context) {
      return child.type() == gqe::expression::expression_type::binary_op &&
             !child_context.column_idx.has_value() &&
             child_context.cudf_ast_expression.has_value() &&
             cudf::is_fixed_point(child_context.data_type());
    };
    bool const fixed_point_intermediate_scale_unsupported =
      !_use_ast_jit && consumes_fixed_point_scale(op) &&
      (is_fixed_point_ast_intermediate(*lhs, lhs_context) ||
       is_fixed_point_ast_intermediate(*rhs, rhs_context));
    bool const can_use_cudf_ast =
      cudf::is_fixed_width(result_dt) && (exact_type_match || fixed_point_width_compatible) &&
      !needs_fixed_point_conditioning && !decimal128_interpreted_ast_unsupported &&
      !fixed_point_intermediate_scale_unsupported;
    GQE_LOG_TRACE(
      "eval::visit(binary_op): expr='{}' op={} lhs=({}) rhs=({}) result=({}) can_use_ast={} "
      "exact_type_match={} fixed_point_width_compatible={} needs_fixed_point_conditioning={} "
      "decimal128_interpreted_ast_unsupported={} "
      "fixed_point_intermediate_scale_unsupported={} use_ast_jit={}",
      expression->to_string(),
      static_cast<int>(op),
      gqe::utility::data_type_to_string(lhs_dt),
      gqe::utility::data_type_to_string(rhs_dt),
      gqe::utility::data_type_to_string(result_dt),
      can_use_cudf_ast,
      exact_type_match,
      fixed_point_width_compatible,
      needs_fixed_point_conditioning,
      decimal128_interpreted_ast_unsupported,
      fixed_point_intermediate_scale_unsupported,
      _use_ast_jit);

    if (can_use_cudf_ast) {
      auto const op = cudf_to_ast_operator(expression->binary_operator());

      auto process_child = [&](auto& context,
                               auto const child_idx) -> cudf::ast::expression const& {
        auto const& child_context = context.child_contexts[child_idx];
        bool const is_task        = child_context->column_idx.has_value();
        if (is_task) {
          auto const column_idx = child_context->column_idx.value();
          context.gqe_expression->_children[child_idx] =
            std::make_shared<column_reference_expression>(column_idx);
          auto new_child = std::make_unique<cudf::ast::column_reference>(column_idx);
          return *context.cudf_ast_dependencies.emplace_back(std::move(new_child));
        } else {
          context.gqe_expression->_children[child_idx] = child_context->gqe_expression;
          return *child_context->cudf_ast_expression.value();
        }
      };

      auto const& cudf_ast_lhs = process_child(context, 0);
      auto const& cudf_ast_rhs = process_child(context, 1);

      context.cudf_ast_expression =
        std::make_unique<cudf::ast::operation>(op, cudf_ast_lhs, cudf_ast_rhs);
    } else {
      auto process_child = [&](auto& context, int child_idx) {
        auto const& child_context = context.child_contexts[child_idx];
        bool const is_scalar      = child_context->cudf_scalar.has_value();

        if (is_scalar) {
          context.gqe_expression->_children[child_idx] = child_context->gqe_expression;
        } else {
          context.gqe_expression->_children[child_idx] =
            std::make_shared<column_reference_expression>(dispatch_task(*child_context));
        }
      };

      bool const lhs_is_scalar = lhs_context.cudf_scalar.has_value();
      bool const rhs_is_scalar = rhs_context.cudf_scalar.has_value();

      if (lhs_is_scalar && rhs_is_scalar) {
        throw std::logic_error(
          "Unable to evaluate cudf::binary_operation on two cudf::scalars in " +
          expression->to_string());
      }

      process_child(context, 0);
      process_child(context, 1);

      dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(if_then_else_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    for (std::size_t child_id = 0; child_id < expression->_children.size(); ++child_id) {
      auto const& child = expression->_children[child_id];
      child->accept(*this);
      auto& child_context = this->find_context(child.get());
      context.child_contexts.emplace_back(&child_context);
    }

    auto const cond_dt = context.child_contexts[0]->data_type();
    auto const then_dt = context.child_contexts[1]->data_type();
    auto const else_dt = context.child_contexts[2]->data_type();

    if (!cudf::is_boolean(cond_dt)) {
      throw std::logic_error("Cannot convert " + expression->_children[0]->to_string() +
                             " to boolean");
    }

    if (then_dt != else_dt) {
      throw std::logic_error("Column types of " + expression->_children[1]->to_string() + " and " +
                             expression->_children[2]->to_string() + " must be equal");
    }

    context.result_data_type = then_dt;

    for (std::size_t child_id = 0; child_id < expression->_children.size(); ++child_id) {
      auto& child_context = *context.child_contexts[child_id];

      // IF, THEN, and ELSE columns need to be evaluated.
      // However, cudf::copy_if_else supports scalar arguments for THEN and ELSE
      if (child_id == 0 || !child_context.cudf_scalar.has_value()) {
        auto const child_column_idx = this->dispatch_task(child_context);
        context.gqe_expression->_children[child_id] =
          std::make_shared<column_reference_expression>(child_column_idx);
      }
    }

    this->dispatch_task(context);
  }
}

void expression_evaluator::visit(cast_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const& child = expression->_children[0];
    child->accept(*this);

    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);
    bool const child_is_column = child_context.column_idx.has_value();
    bool const child_has_ast   = child_context.cudf_ast_expression.has_value();

    auto const cudf_ast_operator = cast_to_ast_operator(expression->out_type());

    context.result_data_type = expression->out_type();
    bool const can_use_cudf_ast =
      cudf_ast_operator.has_value() && (child_is_column || child_has_ast);

    if (can_use_cudf_ast) {
      // same logic as in visit(unary_op_expression const*)
      if (child_is_column) {
        auto const child_column_idx = child_context.column_idx.value();
        context.gqe_expression->_children[0] =
          std::make_shared<column_reference_expression>(child_column_idx);
        auto new_child = std::make_unique<cudf::ast::column_reference>(child_column_idx);
        context.cudf_ast_expression =
          std::make_unique<cudf::ast::operation>(*cudf_ast_operator, *new_child);
        context.cudf_ast_dependencies.emplace_back(std::move(new_child));
      } else {
        context.gqe_expression->_children[0] = child_context.gqe_expression;
        context.cudf_ast_expression          = std::make_unique<cudf::ast::operation>(
          *cudf_ast_operator, *child_context.cudf_ast_expression.value());
      }
    } else {
      auto const child_column_idx = dispatch_task(child_context);
      context.gqe_expression->_children[0] =
        std::make_shared<column_reference_expression>(child_column_idx);
      dispatch_task(context);
    }
  }
}

void expression_evaluator::visit(scalar_function_expression const* expression)
{
  auto [context, is_new] = this->emplace_context(expression, expression->clone());

  if (is_new) {
    auto const& child = expression->_children[0];
    child->accept(*this);
    auto& child_context = this->find_context(child.get());
    context.child_contexts.emplace_back(&child_context);
    auto const child_column_idx = dispatch_task(child_context);
    context.gqe_expression->_children[0] =
      std::make_shared<column_reference_expression>(child_column_idx);
    std::vector<cudf::data_type> single_child_types(child_column_idx + 1);
    single_child_types[child_column_idx] = child_context.data_type();
    context.result_data_type             = context.gqe_expression->data_type(single_child_types);

    // scalar function expressions are always evaluated in the fallback path
    dispatch_task(context);
  }
}

void expression_evaluator::visit(subquery_expression const* expression)
{
  expression_visitor::visit(expression);
}

cudf::size_type expression_evaluator::dispatch_task(evaluation_context& context) noexcept
{
  // is already a task
  if (context.column_idx.has_value()) {
    return context.column_idx.value();
  } else {
    // make it a new task and emit column index of the result
    auto const column_idx = _next_task++;
    context.column_idx    = column_idx;

    this->_tasks.emplace_back(&context);
    return column_idx;
  }
}

template <typename... Args>
std::pair<expression_evaluator::evaluation_context&, bool> expression_evaluator::emplace_context(
  expression const* expression, Args&&... args) noexcept
{
  auto [iter, is_new] =
    this->_evaluation_contexts.try_emplace(expression, std::forward<Args>(args)...);
  return {iter->second, is_new};
}

[[nodiscard]] expression_evaluator::evaluation_context& expression_evaluator::find_context(
  expression const* expression)
{
  auto search = this->_evaluation_contexts.find(expression);
  if (search == this->_evaluation_contexts.end()) {
    throw std::logic_error("Unable to find evaluation context for " + expression->to_string());
  }

  return search->second;
}

template <typename T>
void expression_evaluator::create_literal_context(literal_expression<T> const* expression) noexcept
{
  using cudf_scalar_type =
    std::conditional_t<std::is_same_v<T, std::string>,
                       cudf::string_scalar,
                       std::conditional_t<std::is_same_v<T, cudf::timestamp_D>,
                                          cudf::timestamp_scalar<cudf::timestamp_D>,
                                          std::conditional_t<cudf::is_fixed_point<T>(),
                                                             cudf::fixed_point_scalar<T>,
                                                             cudf::numeric_scalar<T>>>>;
  auto [context, is_new] = this->emplace_context(expression, expression->clone());
  if (is_new) {
    auto scalar = std::make_unique<cudf_scalar_type>(
      expression->value(), !expression->is_null(), _stream, _temporary_mr);
    context.cudf_ast_expression = std::make_unique<cudf::ast::literal>(*scalar);
    context.cudf_scalar         = std::move(scalar);
    context.result_data_type    = expression->data_type({});
  }
}

evaluated_expressions evaluate_expressions(gqe::optimization_parameters const& parameters,
                                           expression_input const& input,
                                           std::span<expression const* const> exprs,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref result_mr)
{
  utility::nvtx_scoped_range eval_expr_range("evaluate_expressions");
  auto const table = input.table();
  evaluated_expressions results;
  results.evaluated_result.reserve(exprs.size());
  results.owned_result.reserve(exprs.size());

  for (auto expr : exprs) {
    if (expr == nullptr) {
      throw std::invalid_argument("evaluate_expressions: root expression must not be null");
    }
    if (expr->type() == expression::expression_type::column_reference) {
      auto const* reference = static_cast<column_reference_expression const*>(expr);
      auto const column_idx = input.physical_column_index(reference->column_idx());
      results.evaluated_result.push_back(table.column(column_idx));
      results.owned_result.push_back(nullptr);
      continue;
    }

    expression_evaluator evaluator(input, expr, parameters.use_ast_jit, stream, result_mr);
    auto [result, cache] = evaluator.evaluate(parameters.filter_use_like_shift_and);
    if (cache == nullptr) {
      throw std::logic_error("evaluate_expressions: computed expression has no owning result");
    }
    results.evaluated_result.push_back(result);
    results.owned_result.push_back(std::move(cache));
  }

  return results;
}

}  // namespace gqe
