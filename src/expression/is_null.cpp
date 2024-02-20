/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/expression/is_null.hpp>

#include <cassert>
namespace gqe {
bool is_null_expression::operator==(const expression& other) const
{
  if (auto other_is_null_expr = dynamic_cast<const is_null_expression*>(&other);
      other_is_null_expr != nullptr) {
    auto this_children  = this->children();
    auto other_children = other_is_null_expr->children();
    assert(this_children.size() == 1 && other_children.size() == 1);
    return *this_children[0] == *other_children[0];
  }
  return false;
}
}  // namespace gqe