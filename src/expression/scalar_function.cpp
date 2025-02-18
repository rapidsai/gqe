/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/scalar_function.hpp>
#include <gqe/utility/helpers.hpp>

namespace {
/**
 * @brief Return the string representation of datetime `component`
 */
std::string datetime_component_str(cudf::datetime::datetime_component component)
{
  std::string s;
  switch (component) {
    case cudf::datetime::datetime_component::YEAR: s = "YEAR"; break;
    case cudf::datetime::datetime_component::MONTH: s = "MONTH"; break;
    case cudf::datetime::datetime_component::DAY: s = "DAY"; break;
    case cudf::datetime::datetime_component::WEEKDAY: s = "WEEKDAY"; break;
    case cudf::datetime::datetime_component::HOUR: s = "HOUR"; break;
    case cudf::datetime::datetime_component::MINUTE: s = "MINUTE"; break;
    case cudf::datetime::datetime_component::SECOND: s = "SECOND"; break;
    case cudf::datetime::datetime_component::MILLISECOND: s = "MILLISECOND"; break;
    case cudf::datetime::datetime_component::MICROSECOND: s = "MICROSECOND"; break;
    case cudf::datetime::datetime_component::NANOSECOND: s = "NANOSECOND"; break;
  }
  return s;
}
}  // namespace

[[nodiscard]] std::string gqe::datepart_expression::to_string() const noexcept
{
  auto child_exprs = children();
  assert(child_exprs.size() == 1);
  return "date_part(" + child_exprs[0]->to_string() + ", " + datetime_component_str(_component) +
         ")";
}

[[nodiscard]] std::string gqe::like_expression::to_string() const noexcept
{
  auto child_exprs = children();
  assert(child_exprs.size() == 1);
  std::string fn_name = _ignore_case ? "ilike" : "like";
  return fn_name + "(" + child_exprs[0]->to_string() + ", '" + _pattern + "', '" +
         _escape_character + "')";
}

bool gqe::datepart_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_datepart_expr = dynamic_cast<const datepart_expression&>(other);
  // Compare attributes
  if (this->fn_kind() != other_datepart_expr.fn_kind() ||
      this->component() != other_datepart_expr.component()) {
    return false;
  }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_datepart_expr.children());
}

bool gqe::like_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_like_expr = dynamic_cast<const like_expression&>(other);
  // Compare attributes
  if (this->pattern() != other_like_expr.pattern() ||
      this->escape_character() != other_like_expr.escape_character() ||
      this->ignore_case() != other_like_expr.ignore_case()) {
    return false;
  }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_like_expr.children());
}

bool gqe::round_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_round_expr = dynamic_cast<const round_expression&>(other);
  // Compare attributes
  if (this->decimal_places() != other_round_expr.decimal_places()) { return false; }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_round_expr.children());
}

bool gqe::substr_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_substr_expr = dynamic_cast<const substr_expression&>(other);
  // Compare attributes
  if (this->start() != other_substr_expr.start() || this->length() != other_substr_expr.length()) {
    return false;
  }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_substr_expr.children());
}
