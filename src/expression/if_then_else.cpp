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

#include <gqe/expression/if_then_else.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {

bool if_then_else_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }
  auto other_if_then_expr = dynamic_cast<const if_then_else_expression&>(other);
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_if_then_expr.children());
}

}  // namespace gqe
