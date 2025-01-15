/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/binary_op.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/utilities/traits.hpp>

namespace gqe {

namespace {

// https://github.com/apache/spark/blob/v3.1.1/sql/catalyst/src/main/scala/org/apache/spark/sql/types/DecimalType.scala
struct scala_decimal {
  static constexpr int MAX_PRECISION          = 38;
  static constexpr int MAX_SCALE              = 38;
  static constexpr int MINIMUM_ADJUSTED_SCALE = 6;

  [[nodiscard]] static int adjust_scale(int precision, int scale)
  {
    assert(precision >= scale);

    if (precision <= MAX_PRECISION) {
      // Adjustment only needed when we exceed max precision
      return scale;
    } else if (scale < 0) {
      // Decimal can have negative scale (SPARK-24468). In this case, we cannot allow a precision
      // loss since we would cause a loss of digits in the integer part.
      // In this case, we are likely to meet an overflow.
      return scale;
    } else {
      // Precision/scale exceed maximum precision. Result must be adjusted to max_precision.
      int int_digits = precision - scale;
      // If original scale is less than min_adjusted_scale, use original scale value; otherwise
      // preserve at least min_adjusted_scale fractional digits
      int min_scale_value = std::min(scale, MINIMUM_ADJUSTED_SCALE);
      // The resulting scale is the maximum between what is available without causing a loss of
      // digits for the integer part of the decimal and the minimum guaranteed scale, which is
      // computed above
      return std::max(MAX_PRECISION - int_digits, min_scale_value);
    }
  }

  [[nodiscard]] static int bound_scale(int precision, int scale)
  {
    return std::min(scale, MAX_SCALE);
  }
};

[[nodiscard]] int get_decimal_precision(cudf::data_type type)
{
  // floor(log10(2^(bits-1)))
  if (size_of(type) == 4)
    return 9;
  else if (size_of(type) == 8)
    return 18;
  else if (size_of(type) == 16)
    return 38;
  else
    return 0;
}

// `lhs_type` or `rhs_type` must be a fixed point value.
// Notes on allow_precision_loss:
// https://stackoverflow.com/questions/67756929/controlling-decimal-precision-overflow-in-spark
[[nodiscard]] int calc_decimal_division_scale(cudf::data_type lhs_type,
                                              cudf::data_type rhs_type,
                                              bool allow_precision_loss = true)
{
  // Spark's algorithm for calculating the result scale for a decimal division

  assert(cudf::is_fixed_point(lhs_type) || cudf::is_fixed_point(rhs_type));
  int const pl = get_decimal_precision(lhs_type);
  int const pr = get_decimal_precision(rhs_type);

  // Spark's scale space is opposite cudf.
  int const sl = -lhs_type.scale();
  int const sr = -rhs_type.scale();

  int scale;

  if (allow_precision_loss) {
    // Precision: p1 - s1 + s2 + max(6, s1 + p2 + 1)
    // Scale: max(6, s1 + p2 + 1)
    int int_dig = pl - sl + sr;
    scale       = std::max(scala_decimal::MINIMUM_ADJUSTED_SCALE, sl + pr + 1);
    int prec    = int_dig + scale;
    scale       = scala_decimal::adjust_scale(prec, scale);
  } else {
    int int_dig    = std::min(scala_decimal::MAX_SCALE, pl - sl + sr);
    int dec_dig    = std::min(scala_decimal::MAX_SCALE,
                           std::max(scala_decimal::MINIMUM_ADJUSTED_SCALE, sl + pr + 1));
    int const diff = (int_dig + dec_dig) - scala_decimal::MAX_SCALE;
    if (diff > 0) {
      dec_dig -= diff / 2 + 1;
      int_dig = scala_decimal::MAX_SCALE - dec_dig;
    }
    scale = scala_decimal::bound_scale(int_dig + dec_dig, dec_dig);
  }

  // Convert back to cudf scale space.
  return -scale;
}

[[nodiscard]] cudf::type_id get_decimal_type_for_scale(int scale)
{
  scale = -scale;
  if (scale < 9)
    return cudf::type_id::DECIMAL32;
  else if (scale < 18)
    return cudf::type_id::DECIMAL64;
  else if (scale < 38)
    return cudf::type_id::DECIMAL128;
  else
    throw std::runtime_error("Decimal Overflow");
}

}  // namespace

cudf::type_id binary_op_decimal_promotion_type(int required_scale,
                                               cudf::data_type left_type,
                                               cudf::data_type right_type)

{
  cudf::type_id const need_scale_type_id = get_decimal_type_for_scale(required_scale);

  std::size_t const lhs_type_size       = cudf::size_of(left_type);
  std::size_t const rhs_type_size       = cudf::size_of(right_type);
  std::size_t const promotion_type_size = std::max(lhs_type_size, rhs_type_size);

  cudf::type_id promotion_type_id;

  if (promotion_type_size == 4)
    promotion_type_id = cudf::type_id::DECIMAL32;
  else if (promotion_type_size == 8)
    promotion_type_id = cudf::type_id::DECIMAL64;
  else if (promotion_type_size >= 16)
    promotion_type_id = cudf::type_id::DECIMAL128;
  else
    throw std::runtime_error("Unknown type size.");

  return std::max(promotion_type_id, need_scale_type_id);
}

cudf::data_type arithmetic_output_type(cudf::binary_operator op,
                                       cudf::data_type left_type,
                                       cudf::data_type right_type)
{
  // FIXME: Right now, we promote the result to either INT64 or FLOAT64. Is this necessary?
  if (cudf::is_integral(left_type) && cudf::is_integral(right_type)) {
    return cudf::data_type(cudf::type_id::INT64);
  } else if (cudf::is_fixed_point(left_type) || cudf::is_fixed_point(right_type)) {
    // First, calculate the output scale based on Spark.
    int output_scale = 0;
    if (op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB) {
      // min since negative scales increase decimal digit.
      output_scale = std::min(left_type.scale(), right_type.scale());
    } else if (op == cudf::binary_operator::MUL) {
      output_scale = left_type.scale() + right_type.scale();
    } else if (op == cudf::binary_operator::DIV) {
      output_scale = calc_decimal_division_scale(left_type, right_type);
    } else {
      // Default to cuDF.
      return cudf::binary_operation_fixed_point_output_type(op, left_type, right_type);
    }
    // Second, build the promoted output type.
    auto output_type = binary_op_decimal_promotion_type(output_scale, left_type, right_type);
    // Lastly, construct the full output data type from the type and scale.
    return cudf::data_type(output_type, numeric::scale_type(output_scale));
  } else if (cudf::is_floating_point(left_type) || cudf::is_floating_point(right_type)) {
    return cudf::data_type(cudf::type_id::FLOAT64);
  } else {
    throw std::logic_error("Encountered unsupported types for arithmatic binary expressions");
  }
}

cudf::data_type add_expression::data_type(std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(cudf::binary_operator::ADD,
                                child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

cudf::data_type subtract_expression::data_type(
  std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(cudf::binary_operator::SUB,
                                child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

cudf::data_type multiply_expression::data_type(
  std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(cudf::binary_operator::MUL,
                                child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

cudf::data_type divide_expression::data_type(std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(cudf::binary_operator::DIV,
                                child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

bool binary_op_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_binary_expr = dynamic_cast<const binary_op_expression*>(&other);
  if (this->binary_operator() != other_binary_expr->binary_operator()) { return false; }
  // Recursively compare children
  return utility::compare_pointer_vectors(this->children(), other_binary_expr->children());
}

}  // namespace gqe
