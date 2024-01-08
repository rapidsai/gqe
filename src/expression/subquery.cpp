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

#include <gqe/expression/subquery.hpp>
#include <gqe/utility/helpers.hpp>

namespace gqe {
bool subquery_expression::operator==(const expression& other) const
{
  if (this->type() != other.type()) { return false; }

  auto other_subquery_expr = dynamic_cast<const subquery_expression*>(&other);
  if (this->subquery_type() != other_subquery_expr->subquery_type()) { return false; }
  // Compare attributes
  if (this->relation_index() != other_subquery_expr->relation_index()) { return false; }
  // Recuresively compare children
  return utility::compare_pointer_vectors(this->children(), other_subquery_expr->children());
}
}  // namespace gqe
