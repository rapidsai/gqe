/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
