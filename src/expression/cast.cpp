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

#include <gqe/expression/cast.hpp>

#include <cassert>

namespace gqe {
bool cast_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_cast_expr = dynamic_cast<const cast_expression&>(other);
  if (this->out_type() != other_cast_expr.out_type()) { return false; }
  // Recuresively compare child
  auto this_children  = this->children();
  auto other_children = other_cast_expr.children();
  assert(this_children.size() == 1 && other_children.size() == 1);
  return *this_children[0] == *other_children[0];
}
}  // namespace gqe
