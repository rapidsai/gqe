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

#include <gqe/expression/binary_op.hpp>

#include <gqe/utility/helpers.hpp>

#include <cudf/utilities/traits.hpp>

namespace gqe {

namespace {

cudf::data_type arithmetic_output_type(cudf::data_type left_type, cudf::data_type right_type)
{
  // FIXME: Right now, we promote the result to either INT64 or FLOAT64. Is this necessary?
  if (cudf::is_integral(left_type) && cudf::is_integral(right_type)) {
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

bool binary_op_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_binary_expr = dynamic_cast<const binary_op_expression*>(&other);
  if (this->binary_operator() != other_binary_expr->binary_operator()) { return false; }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_binary_expr->children());
}

}  // namespace gqe
