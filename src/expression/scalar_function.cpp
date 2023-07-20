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

namespace {
/**
 * @brief Return the string representation of datetime `component`
 */
std::string datetime_component_str(gqe::datepart_expression::datetime_component component)
{
  std::string s;
  switch (component) {
    case gqe::datepart_expression::datetime_component::year: s = "YEAR"; break;
    case gqe::datepart_expression::datetime_component::month: s = "MONTH"; break;
    case gqe::datepart_expression::datetime_component::day: s = "DAY"; break;
    case gqe::datepart_expression::datetime_component::weekday: s = "WEEKDAY"; break;
    case gqe::datepart_expression::datetime_component::hour: s = "HOUR"; break;
    case gqe::datepart_expression::datetime_component::minute: s = "MINUTE"; break;
    case gqe::datepart_expression::datetime_component::second: s = "SECOND"; break;
    case gqe::datepart_expression::datetime_component::millisecond: s = "MILLISECOND"; break;
    case gqe::datepart_expression::datetime_component::nanosecond: s = "NANOSECOND"; break;
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
