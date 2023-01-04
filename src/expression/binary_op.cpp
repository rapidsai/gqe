/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/binary_op.hpp>

#include <cudf/utilities/traits.hpp>

#include <type_traits>

namespace gqe {

namespace {

// FIXME: Replace with cudf::is_integral once migrating to cuDF 22.12
struct is_integral_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return std::is_integral<T>();
  }
};

constexpr bool is_integral(cudf::data_type type)
{
  return cudf::type_dispatcher(type, is_integral_impl{});
}

cudf::data_type arithmetic_output_type(cudf::data_type left_type, cudf::data_type right_type)
{
  // FIXME: Right now, we promote the result to either INT64 or FLOAT64. Is this necessary?
  if (is_integral(left_type) && is_integral(right_type)) {
    return cudf::data_type(cudf::type_id::INT64);
  } else if (cudf::is_floating_point(left_type) || cudf::is_floating_point(right_type) ||
             cudf::is_fixed_point(left_type) || cudf::is_fixed_point(right_type)) {
    return cudf::data_type(cudf::type_id::FLOAT64);
  } else {
    throw std::logic_error("Encountered unsupported types for arithmatic binary expressions");
  }
}

}  // namespace

cudf::data_type add_expression::data_type(std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

cudf::data_type subtract_expression::data_type(
  std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

cudf::data_type multiply_expression::data_type(
  std::vector<cudf::data_type> const& column_types) const
{
  auto const child_exprs = children();
  assert(child_exprs.size() == 2);
  return arithmetic_output_type(child_exprs[0]->data_type(column_types),
                                child_exprs[1]->data_type(column_types));
}

}  // namespace gqe
